"""Stream a few fixed-seed episodes at primitive control frequency."""
from pathlib import Path
import warnings
import numpy as np
import gymnasium as gym
from utils.online_eval import preserve_rng


class EvalVideo(gym.Wrapper):
    def __init__(self, env, directory, episodes=2):
        super().__init__(env)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.limit, self.index, self.writer = episodes, -1, None
        self.disabled = False

    def _finish(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _frame(self):
        if self.writer is None:
            return
        try:
            # Rendering must not consume policy/point-cloud random numbers.
            with preserve_rng():
                frame = self.env.render()
            if hasattr(frame, 'detach'):
                frame = frame.detach().cpu().numpy()
            frame = np.asarray(frame)
            if frame.ndim == 4 and frame.shape[0] == 1:
                frame = frame[0]
            if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                raise ValueError(f'Unsupported video frame: {frame.shape}')
            if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1:
                frame = frame * 255
            self.writer.append_data(np.clip(frame[..., :3], 0, 255).astype(np.uint8))
        except Exception as exc:
            self._finish()
            self.disabled = True
            (self.directory / 'video_error.txt').write_text(str(exc))
            warnings.warn(f'Video disabled; evaluation continues: {exc}')

    def reset(self, **kwargs):
        self._finish()
        result = self.env.reset(**kwargs)
        self.index += 1
        if self.index < self.limit and not self.disabled:
            try:
                import imageio.v2 as imageio
                self.writer = imageio.get_writer(
                    str(self.directory / f'episode_{self.index:03d}_seed{kwargs.get("seed")}.mp4'),
                    fps=int(getattr(self.unwrapped, 'control_freq', 20)))
            except Exception as exc:
                self.disabled = True
                (self.directory / 'video_error.txt').write_text(str(exc))
                warnings.warn(f'Video disabled: {exc}')
        self._frame()
        return result

    def step(self, action):
        result = self.env.step(action)
        self._frame()
        if result[2] or result[3]:
            self._finish()
        return result

    def close(self):
        try:
            self._finish()
        finally:
            self.env.close()
