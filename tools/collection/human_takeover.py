"""Collect StackCube corrections with SAPIEN drag targets or keyboard/buttons."""
import argparse
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.corrections import held_out_seeds
from data.episodes import digest, save_episode
from evaluation.grasp_trace import GraspTrace
from evaluation.runner import seed_all, preserve_rng
from models.checkpoint import policy_weights
from models.factory import build_base, observation_encoder
from models.online_policy import FlowPPOPolicy
from utils.experiment import write_json
from utils.normalizer import MinMaxNormalizer
from workflows.collection import PrimitiveRecorder
from workflows.human_collection import TakeoverController


def rgb_image(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[-1] not in (3, 4) or not np.isfinite(value).all():
        raise ValueError(f'Invalid RGB image {value.shape}')
    if np.issubdtype(value.dtype, np.floating) and value.max() <= 1:
        value = value*255
    return np.clip(value[..., :3], 0, 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='outputs/StackCube-v1/oc_budget/iterative/checkpoints/best.pth')
    p.add_argument('--output', default='outputs/StackCube-v1/oc_budget/human_session_01')
    p.add_argument('--seed-start', type=int, default=12000)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--exclude-seeds', type=int, nargs='*', default=[])
    p.add_argument('--sampler', choices=['cps', 'ode'], default='cps')
    p.add_argument('--device', default='cuda')
    p.add_argument('--auto-pause', action='store_true', help='Pause once per sustained hint type per episode')
    p.add_argument('--policy-delay-ms', type=int, default=100, help='Wall-clock delay between primitive steps')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--ui', choices=['sapien', 'buttons'], default='sapien')
    return p.parse_args()


class CollectorUI:
    def init_session(self, args, cfg, actor, encode, normalizer, output):
        self.args, self.cfg, self.actor, self.encode = args, cfg, actor, encode
        self.normalizer, self.output = normalizer, output
        self.index, self.active, self.running = -1, False, False
        self.env, self.writer, self.events_handle = None, None, None
        self.notice, self.review = '', dict(schema='myrl_human_review_v1', episodes=[])

    def __init__(self, root, args, cfg, actor, encode, normalizer, output):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.init_session(args, cfg, actor, encode, normalizer, output)
        root.title('MyRL — Human corrective collection')
        self.image_label = ttk.Label(root)
        self.image_label.pack()
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, wraplength=1150).pack(fill='x')
        self.hint_label = ttk.Label(root, foreground='#a34000', wraplength=1150)
        self.hint_label.pack(fill='x')
        bar = ttk.Frame(root)
        bar.pack()
        for label, callback in [('Run policy [P]', self.run_policy), ('Pause [Space]', self.pause),
                                ('Take over [H]', self.takeover), ('Policy step [N]', self.policy_step),
                                ('Finish/save [F]', self.finish), ('Next seed', self.next_episode)]:
            ttk.Button(bar, text=label, command=lambda cb=callback: self.safe(cb)).pack(side='left')
        controls = ttk.Frame(root)
        controls.pack()
        self.magnitude, self.repeat = tk.StringVar(value='0.05'), tk.StringVar(value='1')
        ttk.Label(controls, text='Controller delta (NOT metres):').pack(side='left')
        ttk.Combobox(controls, textvariable=self.magnitude, values=['0.02', '0.05', '0.1', '0.2'], width=6, state='readonly').pack(side='left')
        ttk.Label(controls, text='Primitive steps per click:').pack(side='left')
        ttk.Combobox(controls, textvariable=self.repeat, values=['1', '5', '10', '20'], width=4, state='readonly').pack(side='left')
        move = ttk.Frame(root)
        move.pack()
        self.keys = {}
        for axis, (name, positive, negative) in enumerate([
            ('X', 'd', 'a'), ('Y', 'w', 's'), ('Z', 'e', 'q'),
            ('Rx', 'l', 'j'), ('Ry', 'i', 'k'), ('Rz', 'u', 'o')]):
            for sign, key in ((1, positive), (-1, negative)):
                callback = lambda a=axis, s=sign: self.move(a, s)
                self.keys[key] = callback
                ttk.Button(move, text=f'{name}{"+" if sign > 0 else "-"} [{key.upper()}]',
                           command=lambda cb=callback: self.safe(cb)).grid(row=axis//3, column=(axis%3)*2+(sign < 0))
        grip = ttk.Frame(root)
        grip.pack()
        for text, callback in [('Open [G]', lambda: self.move(gripper=1.)),
                               ('Close [B]', lambda: self.move(gripper=-1.)),
                               ('Hold/settle [T]', self.move)]:
            ttk.Button(grip, text=text, command=lambda cb=callback: self.safe(cb)).pack(side='left')
        self.keys.update(p=self.run_policy, space=self.pause, h=self.takeover, n=self.policy_step,
                         f=self.finish, g=lambda: self.move(gripper=1.), b=lambda: self.move(gripper=-1.), t=self.move)
        ttk.Label(root, text='Click a button or press/release a key. Human actions require H first. '
                  'Pause/idle never advances physics. Axes follow the validated robot-root controller frame.').pack()
        self.pressed = set()
        root.bind('<KeyPress>', self.key_press)
        root.bind('<KeyRelease>', lambda event: self.pressed.discard(event.keysym.lower()))
        root.bind('<FocusOut>', lambda event: self.pressed.clear())
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.report_callback_exception = self.callback_error
        root.after(max(10, args.policy_delay_ms), self.tick)

    def key_press(self, event):
        key = event.keysym.lower()
        if key not in self.pressed and key in self.keys:
            self.pressed.add(key)
            self.safe(self.keys[key])

    def safe(self, callback):
        try:
            callback()
        except Exception as exc:
            self.callback_error(type(exc), exc, exc.__traceback__)

    def callback_error(self, kind, error, tb):
        import traceback
        from tkinter import messagebox
        self.running = False
        self.faulted = True
        self.notice = f'ERROR: {kind.__name__}: {error}; episode excluded, restart collector.'
        traceback.print_exception(kind, error, tb)
        write_json(self.output/'error.json', dict(error=self.notice, seed=getattr(self, 'seed', None)))
        messagebox.showerror('Collection stopped', self.notice)
        # A failed env.step may have advanced physics without returning an observation.
        # Never save such a trajectory as valid training data.
        self.active = False
        self.refresh_status()

    def next_episode(self):
        if self.active:
            self.notice = 'Finish/save the current episode before moving to the next seed.'
            self.refresh_status()
            return
        if getattr(self, 'faulted', False):
            return
        self.close_resources()
        self.index += 1
        if self.index >= self.args.episodes:
            self.notice = 'Session complete. Close window, inspect videos, then review/import.'
            self.refresh_status()
            return
        from envs.factory import make_env
        self.seed = self.args.seed_start+self.index
        seed_all(self.seed)
        holder = []
        def wrap(env):
            recorder = PrimitiveRecorder(env)
            holder.append(recorder)
            return recorder
        self.env = make_env(self.cfg, primitive_wrapper=wrap)
        if self.env.use_ensembling:
            raise ValueError('Takeover currently supports non-ensembled prefix execution')
        raw = self.env.unwrapped
        self.telemetry = GraspTrace(holder[0], self.output, self.normalizer, self.cfg.env.exec_steps)
        def snapshot(obs, info, contacts=True):
            with preserve_rng():
                return self.telemetry.snapshot(obs, info, contacts)
        def sample(obs):
            return self.actor.sample(self.encode(obs), self.cfg.model.num_inference_steps)[0].cpu().numpy()
        self.control = TakeoverController(holder[0], sample, self.normalizer, self.cfg.env.exec_steps,
                                          snapshot, float(raw.control_freq))
        self.control.reset(self.seed)
        # Do not silently replace the wrist-camera robot with a plain Panda.
        if raw.agent.uid != 'panda_wristcam':
            raise ValueError(f'Expected panda_wristcam, got {raw.agent.uid}; use a matching environment/checkpoint')
        arm = raw.agent.controller.controllers['arm'].config
        if not arm.use_delta or arm.use_target or not arm.normalize_action or arm.frame != 'root_translation:root_aligned_body_rotation':
            raise ValueError('Unsupported controller semantics for keyboard deltas')
        self.telemetry.initial_height = self.control.before['cubeA_pose_world'][2]
        self.active, self.running = True, False
        self.warned, self.notice = set(), 'Paused. P runs policy; H takes over.'
        self.path = self.output/f'seed_{self.seed}.npz'
        self.events_handle = self.path.with_suffix('.events.jsonl').open('w', encoding='utf-8')
        self.video_path, self.video_error = None, None
        if not self.args.no_video:
            try:
                import imageio.v2 as imageio
                self.video_path = str(self.path.with_suffix('.mp4'))
                self.writer = imageio.get_writer(self.video_path, fps=int(raw.control_freq))
            except Exception as exc:
                self.video_error = str(exc)
                self.writer = None
        self.draw(record=True)

    def run_policy(self):
        if self.active:
            self.control.switch('policy')
            self.running = True
            self.notice = 'Policy running. Space pauses; H takes over.'
            self.refresh_status()

    def pause(self):
        if self.active:
            self.running = False
            self.control.switch('paused')
            self.notice = 'Paused; physics and episode clock stopped. H for human, P to resume.'
            self.refresh_status()

    def takeover(self):
        if self.active:
            self.running = False
            self.control.switch('human')
            self.notice = 'Human mode. Position/rotation buttons, G open, B close, T settle.'
            self.refresh_status()

    def policy_step(self):
        if self.active:
            self.running = False
            self.control.switch('policy')
            self.advance()

    def tick(self):
        if self.active and self.running:
            self.safe(self.advance)
        self.root.after(max(10, self.args.policy_delay_ms), self.tick)

    def move(self, axis=None, sign=1, gripper=None):
        if not self.active:
            return
        if self.control.mode != 'human':
            self.notice = 'Press H / Take over before human movement.'
            self.refresh_status()
            return
        action = self.control.human_command(axis, sign*float(self.magnitude.get()), gripper)
        for _ in range(int(self.repeat.get())):
            if not self.active:
                break
            self.advance(action)

    def advance(self, action=None):
        if not self.control.advance(action):
            return
        self.events_handle.write(json.dumps(self.control.events[-1], allow_nan=False)+'\n')
        self.events_handle.flush()
        if self.control.mode == 'policy' and self.args.auto_pause:
            new = set(self.control.hints)-self.warned
            if new:
                self.warned.update(new)
                self.pause()
                self.notice = 'Sustained hint: '+', '.join(sorted(new))+'. H take over, P continue; not a definitive diagnosis.'
        self.draw(record=True)
        if self.control.done:
            self.finish()

    def render_panel(self):
        from PIL import Image, ImageDraw
        with preserve_rng():
            frames = [('Overview (operator only)', rgb_image(self.env.render()))]
            # Same configured cameras as policy; never replace the wrist stream with overview.
            images = self.env.unwrapped.get_sensor_images()
            for name, values in images.items():
                if 'rgb' in values:
                    frames.append((name, rgb_image(values['rgb'])))
        if len(frames) < 3:
            raise ValueError('Expected overview + base/wrist RGB streams; cannot silently hide missing cameras')
        frames = frames[:4]
        panel = Image.new('RGB', (384*len(frames), 416), '#202020')
        draw = ImageDraw.Draw(panel)
        for i, (name, frame) in enumerate(frames):
            picture = Image.fromarray(frame)
            picture.thumbnail((384, 384))
            panel.paste(picture, (i*384, 24))
            draw.text((i*384+5, 5), name, fill='white')
        draw.text((5, 395), f'seed={self.seed} step={self.control.steps} source={self.control.mode} '
                  f'hints={",".join(self.control.hints)}', fill='white')
        return panel

    def draw(self, record=False):
        from PIL import ImageTk
        panel = self.render_panel()
        self.photo = ImageTk.PhotoImage(panel)
        self.image_label.configure(image=self.photo)
        self.record_panel(panel, record)
        self.refresh_status()

    def record_panel(self, panel, record):
        if record and self.writer is not None:
            try:
                self.writer.append_data(np.asarray(panel))
            except Exception as exc:
                self.video_error = str(exc)
                try:
                    self.writer.close()
                finally:
                    self.writer = None

    def refresh_status(self):
        if hasattr(self, 'control'):
            c = self.control
            self.status.set(f'Seed {self.seed} | step {c.steps}/{self.cfg.env.max_episode_steps} | '
                f'mode={c.mode} | success={bool(c.info.get("success", False))} | {self.notice}')
            self.hint_label.configure(text='Suspected failure: '+(', '.join(c.hints) or 'none')+
                '. Hints use simulator telemetry, never policy inputs. '+
                (f'VIDEO ERROR: {self.video_error}' if getattr(self, 'video_error', None) else ''))

    def finish(self):
        if not self.active:
            return
        self.running = False
        c = self.control
        if c.steps:
            ep = c.episode()
            save_episode(self.path, ep)
            meta = dict(seed=self.seed, steps=c.steps, final_success=bool(ep['success'][-1]),
                        end_reason='environment_end' if c.done else 'operator_stop',
                        segments=c.segments, robot_uid=self.env.unwrapped.agent.uid,
                        controller_config=repr(getattr(self.env.unwrapped.agent.controller, 'configs', None)),
                        action_low=c.low.tolist(), action_high=c.high.tolist(),
                        failure_hints=sorted({h for e in c.events for h in e['hints']}),
                        diagnostic_errors=self.telemetry.errors, video=self.video_path,
                        video_error=self.video_error)
            write_json(self.path.with_suffix('.json'), meta)
            self.review['episodes'].append(dict(path=self.path.name, sha256=digest(self.path),
                metadata_sha256=digest(self.path.with_suffix('.json')), decision='pending',
                segments=[dict(s, decision='pending') for s in c.segments]))
            write_json(self.output/'review.json', self.review)
        self.active = False
        self.close_resources(close_env=False)
        self.notice = 'Saved for review (not yet expert data). Click Next seed or close window.'
        self.refresh_status()

    def close_resources(self, close_env=True):
        if self.events_handle:
            self.events_handle.close()
            self.events_handle = None
        if self.writer:
            self.writer.close()
            self.writer = None
        if self.env and close_env:
            self.env.close()
            self.env = None

    def close(self):
        from tkinter import messagebox
        if self.active and not messagebox.askyesno('Save and exit?', 'Save current partial episode as an operator timeout and exit?'):
            return
        self.safe(self.finish)
        self.close_resources()
        self.root.destroy()


def run(args):
    if args.episodes < 1 or args.policy_delay_ms < 0:
        raise ValueError('Invalid episode count/delay')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'{output}: use a new output directory')
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    cp = torch.load(checkpoint, map_location=args.device, weights_only=True)
    config = cp['config']
    cfg = OmegaConf.create(config)
    if cfg.env.env_id != 'StackCube-v1' or cfg.env.control_mode != 'pd_ee_delta_pose' or cfg.model.action_dim != 7:
        raise ValueError('Collector supports StackCube-v1 / Panda / pd_ee_delta_pose only')
    seeds = set(range(args.seed_start, args.seed_start+args.episodes))
    if seeds & (held_out_seeds(config) | set(args.exclude_seeds)):
        raise ValueError('Collection seeds overlap evaluation/excluded seeds')
    cfg.device = args.device
    normalizer = MinMaxNormalizer()
    normalizer.stats = cp['normalizer']
    base = build_base(cfg, args.device)
    base.load_state_dict(policy_weights(cp), strict=True)
    actor = FlowPPOPolicy(base, num_steps=cfg.model.num_inference_steps,
                         noise_level=cfg.get('noise_level', cfg.algo.get('noise_level', .7)),
                         min_std=cfg.get('min_std', cfg.algo.get('min_std', .0067)), eval_mode=args.sampler)
    actor.eval()
    encode = observation_encoder(cfg, actor, normalizer, args.device)
    root = None
    if args.ui == 'buttons':
        import tkinter as tk
        root = tk.Tk()
    else:
        # Fail before creating an output directory if official planning dependencies are missing.
        from tools.collection.sapien_takeover import SapienCollector
        from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    output.mkdir(parents=True)
    packages = {}
    for package in ('mani_skill', 'sapien', 'mplib', 'torch', 'Pillow'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    write_json(output/'session.json', dict(schema='myrl_human_session_v1', config=config,
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), normalizer=cp['normalizer'],
        sampler=args.sampler, ui=args.ui, versions=packages, seeds=sorted(seeds), excluded_seeds=args.exclude_seeds,
        weight_key='ema_model_state_dict' if 'ema_model_state_dict' in cp else 'model_state_dict',
        intervention_success_is_not_autonomous_evaluation=True))
    write_json(output/'review.json', dict(schema='myrl_human_review_v1', episodes=[]))
    ui = None
    try:
        ui = (CollectorUI(root, args, cfg, actor, encode, normalizer, output) if root is not None else
              SapienCollector(args, cfg, actor, encode, normalizer, output))
        ui.next_episode()
        if root is not None:
            root.mainloop()
        else:
            ui.loop()
    finally:
        if ui is not None:
            ui.close_resources()
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass  # Window was already closed by the operator.
    print(f'Collected session: {output}; review.json must be approved before import.')


if __name__ == '__main__':
    run(parse_args())
