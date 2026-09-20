"""Independent object tokens: all valid points reach masked max/mean pooling."""
import torch
from torch import nn
from data.object_centric import ROLES


class ObjectPointEncoder(nn.Module):
    def __init__(self, in_channels, output_dim):
        super().__init__()
        self.point_mlp = nn.Sequential(nn.Linear(in_channels, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU())
        self.projection = nn.Linear(256, output_dim)

    def forward(self, points, mask):
        mask = mask.bool()
        # Mask before the MLP too: padded NaNs/large values must not leak into gradients.
        features = self.point_mlp(torch.where(mask[..., None], points, 0.))
        count = mask.sum(-1, keepdim=True)
        mean = (features * mask[..., None]).sum(-2) / count.clamp_min(1)
        maximum = features.masked_fill(~mask[..., None], -torch.inf).amax(-2)
        maximum = torch.where(count > 0, maximum, 0.)
        pooled = self.projection(torch.cat([maximum, mean], -1))
        return torch.where(count > 0, pooled, 0.)


class ObjectCentricEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dim=256, use_state=True,
                 state_dim=16, variant='points', state_skip=False):
        super().__init__()
        if output_dim % 4 or variant not in ('points', 'centers'):
            raise ValueError('Require output_dim divisible by 4 and points/centers variant')
        if not isinstance(state_skip, bool):
            raise ValueError('state_skip must be boolean')
        if state_skip and (not use_state or state_dim < 1):
            raise ValueError('state_skip requires use_state=True and positive state_dim')
        self.use_state, self.variant = use_state, variant
        self.object_encoder = ObjectPointEncoder(in_channels, output_dim)
        self.context_encoder = ObjectPointEncoder(in_channels, output_dim)
        # center(3), extent(3), validity(1), occupancy(1)
        self.geometry = nn.Sequential(nn.Linear(8, output_dim), nn.GELU(),
                                      nn.Linear(output_dim, output_dim))
        self.role = nn.Embedding(len(ROLES), output_dim)
        self.state_encoder = nn.Linear(state_dim, output_dim) if use_state else None
        self.token_type = nn.Parameter(torch.zeros(3, output_dim))
        self.readout = nn.Parameter(torch.zeros(1, 1, output_dim))
        nn.init.normal_(self.token_type, std=.02)
        nn.init.normal_(self.readout, std=.02)
        layer = nn.TransformerEncoderLayer(output_dim, 4, 2*output_dim,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(output_dim)
        self.state_projection = None
        if state_skip:
            # Start as the legacy readout, without shifting downstream model RNG.
            # The zero-initialized state columns are trainable from the first update.
            with torch.random.fork_rng(devices=[]):
                self.state_projection = nn.Linear(output_dim + state_dim, output_dim)
            with torch.no_grad():
                self.state_projection.weight.zero_()
                self.state_projection.weight[:, :output_dim].copy_(torch.eye(output_dim))
                self.state_projection.bias.zero_()

    def forward(self, obs):
        valid = obs['object_valid'].bool()
        mask = obs['object_point_mask'].bool() & valid[..., None]
        center = torch.where(valid[..., None], obs['object_centers'], 0.)
        extent = torch.where(valid[..., None], obs['object_extents'], 0.)
        occupancy = mask.float().mean(-1, keepdim=True)
        if self.variant == 'centers':
            # Visible-centroid ablation, not a simulator-pose oracle.
            extent, occupancy = torch.zeros_like(extent), torch.zeros_like(occupancy)
        geometry = torch.cat([center, extent, valid[..., None].float(), occupancy], -1)
        objects = self.geometry(geometry) + self.role(obs['object_roles'].long()) + self.token_type[0]
        if self.variant == 'points':
            objects = objects + self.object_encoder(obs['object_points'], mask)
        b = objects.shape[0]
        # Missing slots remain explicit role/validity tokens with zero geometry.
        tokens = [self.readout.expand(b, -1, -1), objects]
        if self.variant == 'points':
            context = self.context_encoder(obs['context_points'], obs['context_point_mask'])
            tokens.append((context + self.token_type[1])[:, None])
        if self.use_state:
            tokens.append((self.state_encoder(obs['state']) + self.token_type[2])[:, None])
        fused = self.fusion(torch.cat(tokens, dim=1))
        condition = self.output_norm(fused[:, 0])
        if self.state_projection is not None:
            # Same normalized robot state used by the existing token; no new features.
            condition = self.state_projection(torch.cat([condition, obs['state']], dim=-1))
        return condition
