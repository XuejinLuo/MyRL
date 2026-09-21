"""Global PointNeXt with an optional identity-initialized relational branch."""
import torch
from torch import nn
from models.encoders.pointnext import PointNeXtEncoder


class RelationalPointNeXtEncoder(PointNeXtEncoder):
    def __init__(self, in_channels, output_dim, use_state, state_dim,
                 object_feature_dim=23, object_feature_hidden_dim=64):
        super().__init__(in_channels, output_dim, use_state, state_dim)
        if object_feature_dim != 23 or object_feature_hidden_dim < 1:
            raise ValueError('Relational encoder requires 23 features and positive hidden dim')
        # Do not shift the RNG used to initialize the unchanged Flow and critics.
        with torch.random.fork_rng(devices=[]):
            self.object_feature_encoder = nn.Sequential(
                nn.Linear(object_feature_dim, object_feature_hidden_dim), nn.GELU(),
                nn.Linear(object_feature_hidden_dim, object_feature_hidden_dim), nn.GELU())
            self.relational_fusion = nn.Linear(output_dim + object_feature_hidden_dim, output_dim)
        with torch.no_grad():
            self.relational_fusion.weight.zero_()
            self.relational_fusion.weight[:, :output_dim].copy_(torch.eye(output_dim))
            self.relational_fusion.bias.zero_()

    def forward(self, obs_dict):
        features = obs_dict.get('object_features')
        if features is None or features.ndim != 2 or features.shape[-1] != 23:
            raise ValueError('Relational encoder requires batched 23-D object_features')
        global_features = super().forward(obs_dict)  # Includes robot state once.
        object_features = self.object_feature_encoder(features)
        return self.relational_fusion(torch.cat([global_features, object_features], dim=-1))
