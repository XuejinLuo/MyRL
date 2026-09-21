"""One encoder selection contract for actor and both independent critics."""
from data.observations import observation_mode, relational_enabled


def encoder_options(cfg):
    mode = observation_mode(cfg.env)
    relational = relational_enabled(cfg.env)
    state_skip = (cfg.env.get('observation') or {}).get('state_skip', False)
    if not isinstance(state_skip, bool):
        raise ValueError('env.observation.state_skip must be boolean')
    if state_skip and (mode != 'object_centric' or not cfg.model.use_state or cfg.model.state_dim < 1):
        raise ValueError('state_skip requires object_centric, use_state=True and positive state_dim')
    if mode == 'object_centric':
        return dict(encoder_type='object_centric',
                    encoder_variant=cfg.env.observation.get('encoder_variant', 'points'),
                    state_skip=state_skip)
    if cfg.model.encoder_type != 'pointnext':
        raise ValueError('Global observation modes require the pointnext encoder')
    if relational:
        from numbers import Integral
        dim = cfg.model.get('object_feature_dim', 23)
        hidden = cfg.model.get('object_feature_hidden_dim', 64)
        if (isinstance(dim, bool) or not isinstance(dim, Integral) or dim != 23
                or isinstance(hidden, bool) or not isinstance(hidden, Integral) or hidden < 1):
            raise ValueError('Relational object_feature_dim must be 23 and hidden dim a positive integer')
        return dict(encoder_type='pointnext', relational_features=True,
                    object_feature_dim=dim, object_feature_hidden_dim=hidden)
    return dict(encoder_type=cfg.model.encoder_type)


def build_encoder(encoder_type, in_channels, output_dim, use_state, state_dim,
                  encoder_variant='points', state_skip=False, relational_features=False,
                  object_feature_dim=23, object_feature_hidden_dim=64):
    if relational_features:
        if encoder_type != 'pointnext' or state_skip:
            raise ValueError('Relational features require the global pointnext encoder')
        from models.encoders.relational import RelationalPointNeXtEncoder
        return RelationalPointNeXtEncoder(in_channels, output_dim, use_state, state_dim,
                                          object_feature_dim, object_feature_hidden_dim)
    if encoder_type == 'object_centric':
        from models.encoders.object_centric import ObjectCentricEncoder
        return ObjectCentricEncoder(in_channels, output_dim, use_state, state_dim,
                                    encoder_variant, state_skip)
    if state_skip:
        raise ValueError('state_skip requires the object_centric encoder')
    if encoder_type == 'pointnext':
        from models.encoders.pointnext import PointNeXtEncoder
        return PointNeXtEncoder(in_channels, output_dim, use_state, state_dim)
    raise ValueError(f'Unsupported encoder: {encoder_type}')


def encoder_observation(obs, state=None):
    if not isinstance(obs, dict):
        return dict(point_cloud=obs, **({'state': state} if state is not None else {}))
    if 'pc' in obs:
        return {('point_cloud' if k == 'pc' else k): v for k, v in obs.items()}
    return obs
