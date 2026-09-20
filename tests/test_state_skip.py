"""Direct state conditioning, legacy checkpoint loading and opt-in validation."""
import copy
import io

import pytest
import torch
from omegaconf import OmegaConf

from models.encoders.object_centric import ObjectCentricEncoder
from models.factory import build_base
from tests.test_object_centric import build, tensor_obs, object_config
from tests.test_stages import config
from utils.config import validate_common


@pytest.mark.parametrize('variant', ['points', 'centers'])
def test_identity_initialization_preserves_baseline_and_model_rng(tmp_path, variant):
    cfg = object_config(tmp_path/'unused.h5')
    OmegaConf.update(cfg, 'env.observation.encoder_variant', variant, force_add=True)
    torch.manual_seed(42)
    baseline = build_base(cfg, 'cpu').eval()
    baseline_rng = torch.get_rng_state().clone()
    OmegaConf.update(cfg, 'env.observation.state_skip', True, force_add=True)
    torch.manual_seed(42)
    modified = build_base(cfg, 'cpu').eval()
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    # Existing encoder AND downstream Flow weights keep identical initialization.
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(modified.state_dict()[name], value, rtol=0, atol=0)
    obs = tensor_obs(build(), state_dim=cfg.model.state_dim)
    obs['state'] = torch.rand_like(obs['state']) * 2 - 1
    torch.testing.assert_close(modified.encode(obs), baseline.encode(obs))


@pytest.mark.parametrize('variant', ['points', 'centers'])
@pytest.mark.parametrize('state_dim', [3, 16])
def test_state_route_learns_even_when_token_cannot_see_state(variant, state_dim):
    torch.manual_seed(4)
    encoder = ObjectCentricEncoder(variant=variant, state_dim=state_dim, state_skip=True).eval()
    # Isolate the new path: the original state token is constant for any state.
    with torch.no_grad():
        encoder.state_encoder.weight.zero_()
    obs = tensor_obs(build(0, 0, 0), state_dim=state_dim)
    obs['state'].fill_(.5)
    alternative = {**obs, 'state': -obs['state']}
    torch.testing.assert_close(encoder(obs), encoder(alternative))
    optimizer = torch.optim.SGD(encoder.state_projection.parameters(), lr=.1)
    encoder(obs).square().mean().backward()
    grad = encoder.state_projection.weight.grad[:, -state_dim:]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    optimizer.step()
    a, b = encoder(obs), encoder(alternative)
    assert a.shape == (1, 256) and torch.isfinite(a).all()
    assert not torch.allclose(a, b)


@pytest.mark.parametrize('enabled', [False, True])
def test_checkpoint_roundtrip_and_legacy_default(tmp_path, enabled):
    cfg = object_config(tmp_path/'unused.h5')
    if enabled:
        OmegaConf.update(cfg, 'env.observation.state_skip', True, force_add=True)
    base = build_base(cfg, 'cpu').eval()
    saved = io.BytesIO()
    torch.save(dict(config=OmegaConf.to_container(cfg, resolve=True),
                    model_state_dict=base.state_dict()), saved)
    saved.seek(0)
    cp = torch.load(saved, weights_only=True)
    restored_cfg = OmegaConf.create(cp['config'])
    restored = build_base(restored_cfg, 'cpu').eval()
    restored.load_state_dict(cp['model_state_dict'], strict=True)
    obs = tensor_obs(build(), state_dim=cfg.model.state_dim)
    torch.testing.assert_close(restored.encode(obs), base.encode(obs))
    if not enabled:
        assert not any('state_projection' in key for key in cp['model_state_dict'])
        restored_cfg.env.observation.state_skip = False
        build_base(restored_cfg, 'cpu').load_state_dict(cp['model_state_dict'], strict=True)
    else:
        assert restored.encoder.state_projection is not None
        restored_cfg.env.observation.state_skip = False
        with pytest.raises(RuntimeError):
            build_base(restored_cfg, 'cpu').load_state_dict(cp['model_state_dict'], strict=True)


@pytest.mark.parametrize('mode,use_state,flag', [
    ('global_object_budget', True, True), ('object_centric', False, True),
    ('object_centric', True, 'true'), ('object_centric', True, 1)])
def test_bad_skip_settings_fail_before_training(tmp_path, mode, use_state, flag):
    cfg = config('offline', tmp_path)
    cfg.env.observation.mode = mode
    OmegaConf.update(cfg, 'env.observation.state_skip', flag, force_add=True)
    cfg.env.num_points = 1024
    cfg.model.use_state = use_state
    cfg.model.cond_dim = 32
    with pytest.raises(ValueError, match='state_skip'):
        validate_common(cfg)


def test_comparison_requires_opt_in_for_state_skip(tmp_path):
    from evaluation.compare import comparison_protocol
    cfg = OmegaConf.to_container(object_config(tmp_path/'unused.h5'), resolve=True)
    modified = copy.deepcopy(cfg)
    modified['env']['observation']['state_skip'] = True
    assert comparison_protocol(cfg, {}, .7, .0067) != comparison_protocol(modified, {}, .7, .0067)
    assert comparison_protocol(cfg, {}, .7, .0067, True) == comparison_protocol(modified, {}, .7, .0067, True)
