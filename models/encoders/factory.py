"""One encoder selection contract for actor and both independent critics."""
from data.observations import observation_mode


def encoder_options(cfg):
    if observation_mode(cfg.env) == 'object_centric':
        return dict(encoder_type='object_centric',
                    encoder_variant=cfg.env.observation.get('encoder_variant', 'points'))
    if cfg.model.encoder_type != 'pointnext':
        raise ValueError('Global observation modes require the pointnext encoder')
    return dict(encoder_type=cfg.model.encoder_type)


def build_encoder(encoder_type, in_channels, output_dim, use_state, state_dim,
                  encoder_variant='points'):
    if encoder_type == 'object_centric':
        from models.encoders.object_centric import ObjectCentricEncoder
        return ObjectCentricEncoder(in_channels, output_dim, use_state, state_dim, encoder_variant)
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
