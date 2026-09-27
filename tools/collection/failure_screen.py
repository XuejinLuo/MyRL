"""Unattended screening using exactly the human collector's clipped policy rule."""
import numpy as np

from data.episodes import digest
from tools.collection.human_takeover import CollectorUI
from utils.experiment import write_json
from workflows.failure_queue import state_vector, takeover_step


class FailureScreen(CollectorUI):
    def __init__(self, args, cfg, actor, encode, normalizer, output):
        self.init_session(args, cfg, actor, encode, normalizer, output)

    def start_recording(self):
        # Successful episodes leave only a summary row; no MP4/point-cloud dump.
        self.video_path, self.video_error = None, None

    def draw(self, record=False):
        pass

    def refresh_status(self):
        pass

    def scan(self, provenance):
        horizon = int(self.cfg.env.max_episode_steps)
        if not 1 <= self.args.recovery_reserve < horizon:
            raise ValueError('recovery-reserve must be positive and smaller than the episode horizon')
        queue = dict(provenance, schema='myrl_failure_queue_v1', failures=[], results=[], complete=False,
                     success_rule='any primitive success',
                     execution='collector frozen-range clipping; not an evaluation benchmark',
                     recovery_reserve=self.args.recovery_reserve, rewind_steps=self.args.rewind_steps)
        (self.output/'failures').mkdir()
        for _ in range(self.args.episodes):
            self.next_episode()
            schema, initial = state_vector(self.env.unwrapped.get_state_dict())
            states, success_any = [initial], False
            self.control.switch('policy')
            while not self.control.done:
                self.control.advance()
                current_schema, state = state_vector(self.env.unwrapped.get_state_dict())
                if current_schema != schema:
                    raise ValueError('Simulator state schema changed during screening')
                states.append(state)
                success_any |= bool(self.control.info.get('success', False))
            c = self.control
            result = dict(seed=self.seed, steps=c.steps, success_any=success_any,
                          success_final=bool(c.info.get('success', False)))
            queue['results'].append(result)
            if not success_any:
                step = takeover_step(c.events, c.steps, horizon, self.args.recovery_reserve,
                                     self.args.rewind_steps, self.args.takeover_step)
                path = self.output/'failures'/f'seed_{self.seed}.npz'
                np.savez_compressed(path, actions=np.asarray([e['executed_action'] for e in c.events], np.float32),
                                    states=np.asarray(states))
                queue['failures'].append(dict(result, path=str(path.relative_to(self.output)), sha256=digest(path),
                    takeover_step=step, state_schema=schema,
                    hints=[dict(step=e['step'], hints=e['hints']) for e in c.events if e['hints']]))
            self.active = False  # screening data never becomes an approved correction
            write_json(self.output/'failure_queue.json', queue)
            print(f"Screen {self.index+1}/{self.args.episodes}: seed={self.seed}, success={success_any}; "
                  f"queued failures={len(queue['failures'])}", flush=True)
        queue['complete'] = True
        write_json(self.output/'failure_queue.json', queue)
        print(f"Failure queue: {self.output/'failure_queue.json'} ({len(queue['failures'])} failures)", flush=True)
        return queue
