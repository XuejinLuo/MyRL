"""Interactively inspect one evaluation episode.

Shows:
- base_camera RGB
- hand_camera RGB
- global environment RGB render
- base_camera RGB-colored point cloud before camera fusion
- hand_camera RGB-colored point cloud before camera fusion
- the final sampled point cloud actually passed to the policy
- bottom slider: move through policy-observation frames

Example (uses defaults):
python -m tools.diagnostics.inspect_eval_episode

Override defaults when needed:
python -m tools.diagnostics.inspect_eval_episode \
    --checkpoint outputs/StackCube-v1/oc_budget/offline/checkpoints/best.pth \
    --seed 3000 \
    --sampler cps
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import torch
from omegaconf import OmegaConf

from envs.factory import make_env
from evaluation.runner import seed_all
from models.checkpoint import policy_weights
from models.factory import build_base, observation_encoder
from models.online_policy import FlowPPOPolicy
from utils.normalizer import MinMaxNormalizer


# ============================================================
# Default settings
# You can run this script directly without passing any arguments.
# Command-line arguments can still override these values.
# ============================================================
DEFAULT_CHECKPOINT = "outputs/StackCube-v1/oc_budget/offline/checkpoints/best.pth"
DEFAULT_SEED = 3000
DEFAULT_SAMPLER = "cps"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SAVE = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        type=str,
        help=f"Path to best.pth (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        type=int,
        help=f"Evaluation seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--sampler",
        default=DEFAULT_SAMPLER,
        choices=["cps", "ode"],
        help=f"Evaluation sampler (default: {DEFAULT_SAMPLER})",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Torch device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--save",
        default=DEFAULT_SAVE,
        help="Optional .npz path to save captured frames",
    )
    return parser.parse_args()


def prepare_rgb(frame):
    """Convert RGB image to H x W x 3 uint8."""

    if frame is None:
        return None

    # ------------------------------------------------------------
    # ManiSkill sensor images may be CUDA torch tensors.
    # Move them to CPU before converting to numpy.
    # ------------------------------------------------------------
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)

    # ManiSkill often returns:
    # [1, H, W, C]
    # Remove leading batch dimensions of size 1.
    while frame.ndim > 3 and frame.shape[0] == 1:
        frame = frame[0]

    if frame.ndim != 3:
        print(
            f"[viewer] Unexpected RGB shape: {frame.shape}"
        )
        return None

    # Color texture may contain RGBA.
    # Keep RGB only.
    if frame.shape[-1] > 3:
        frame = frame[..., :3]

    # ------------------------------------------------------------
    # Convert to uint8 [0, 255]
    # ------------------------------------------------------------
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)

        if frame.size > 0:
            frame_min = float(frame.min())
            frame_max = float(frame.max())

            # Common ManiSkill RGB range: [0, 1]
            if frame_min >= 0.0 and frame_max <= 1.0:
                frame = frame * 255.0

        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame

def _as_numpy(value):
    """Convert torch / array-like values to NumPy on CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _remove_single_batch(arr):
    """Remove leading batch dimensions of size 1."""
    arr = _as_numpy(arr)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def capture_camera_observations(env):
    """
    Capture each observation camera's RGB image and its RGB-colored point cloud.

    The per-camera point clouds are reconstructed from ManiSkill's raw
    camera-space position texture, transformed into the world frame using
    cam2world_gl, and colored with that same camera's RGB pixels.

    These are the *pre-fusion* camera clouds. The actual policy input remains
    obs["point_cloud"], which may subsequently be fused/cropped/sampled by
    the environment wrappers.

    Returns:
        camera_rgbs:
            {camera_name: H x W x 3 uint8}
        camera_pointclouds:
            {camera_name: N x 6 float32 [X,Y,Z,R,G,B]}
    """
    camera_rgbs = {}
    camera_pointclouds = {}

    try:
        # Use one sensor capture so RGB and geometry are frame-aligned.
        sensor_data = env.unwrapped._get_obs_sensor_data()
        sensor_params = env.unwrapped.get_sensor_params()
    except Exception as exc:
        print(f"[viewer] Failed to get raw sensor data: {exc}")
        return camera_rgbs, camera_pointclouds

    for camera_name, camera_data in sensor_data.items():
        if not isinstance(camera_data, dict):
            continue

        # --------------------------------------------------------
        # RGB
        # --------------------------------------------------------
        rgb_raw = None
        for key in ("rgb", "Color", "color"):
            if key in camera_data:
                rgb_raw = camera_data[key]
                break

        rgb = prepare_rgb(rgb_raw) if rgb_raw is not None else None
        if rgb is not None:
            camera_rgbs[camera_name] = rgb

        # --------------------------------------------------------
        # Camera-space position + validity / segmentation
        # ManiSkill's standard "position" texture stores XYZ in mm.
        # Older/minimal forms may expose PositionSegmentation.
        # --------------------------------------------------------
        pos_raw = None
        seg_raw = None

        if "position" in camera_data:
            pos_raw = camera_data["position"]
        elif "Position" in camera_data:
            pos_raw = camera_data["Position"]
        elif "PositionSegmentation" in camera_data:
            pos_raw = camera_data["PositionSegmentation"]

        if "segmentation" in camera_data:
            seg_raw = camera_data["segmentation"]
        elif "Segmentation" in camera_data:
            seg_raw = camera_data["Segmentation"]

        if pos_raw is None or rgb is None:
            continue

        pos = _remove_single_batch(pos_raw)
        original_dtype = pos.dtype

        if pos.ndim != 3 or pos.shape[-1] < 3:
            print(
                f"[viewer] Unexpected position shape for {camera_name}: "
                f"{pos.shape}"
            )
            continue

        xyz_cam = pos[..., :3].astype(np.float32)

        # ManiSkill position texture is normally in millimeters.
        # Keep a float fallback for compatibility with custom sensors.
        finite_xyz = xyz_cam[np.isfinite(xyz_cam)]
        scale_to_m = np.issubdtype(original_dtype, np.integer)
        if finite_xyz.size > 0 and not scale_to_m:
            scale_to_m = float(np.max(np.abs(finite_xyz))) > 20.0
        if scale_to_m:
            xyz_cam /= 1000.0

        # Valid pixels: segmentation != 0 when available.
        if seg_raw is not None:
            seg = _remove_single_batch(seg_raw)
            if seg.ndim == 3:
                valid = seg[..., 0] != 0
            else:
                valid = seg != 0
        elif pos.shape[-1] >= 4:
            valid = pos[..., 3] != 0
        else:
            valid = np.isfinite(xyz_cam).all(axis=-1)
            valid &= np.linalg.norm(xyz_cam, axis=-1) > 1e-8

        valid &= np.isfinite(xyz_cam).all(axis=-1)

        # RGB and position should correspond pixel-for-pixel.
        if rgb.shape[:2] != xyz_cam.shape[:2]:
            print(
                f"[viewer] RGB/position size mismatch for {camera_name}: "
                f"rgb={rgb.shape}, position={xyz_cam.shape}"
            )
            continue

        xyz_flat = xyz_cam.reshape(-1, 3)
        rgb_flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
        valid_flat = valid.reshape(-1)

        xyz_flat = xyz_flat[valid_flat]
        rgb_flat = rgb_flat[valid_flat]

        # --------------------------------------------------------
        # Camera frame -> world frame, matching ManiSkill's pointcloud
        # construction before different cameras are concatenated.
        # --------------------------------------------------------
        params = sensor_params.get(camera_name, {})
        cam2world = params.get("cam2world_gl") if isinstance(params, dict) else None

        if cam2world is None:
            print(
                f"[viewer] cam2world_gl not found for {camera_name}; "
                "showing camera-frame point cloud."
            )
            xyz_world = xyz_flat
        else:
            cam2world = _remove_single_batch(cam2world).astype(np.float32)
            if cam2world.shape != (4, 4):
                print(
                    f"[viewer] Unexpected cam2world shape for {camera_name}: "
                    f"{cam2world.shape}"
                )
                xyz_world = xyz_flat
            else:
                ones = np.ones((xyz_flat.shape[0], 1), dtype=np.float32)
                xyz_h = np.concatenate([xyz_flat, ones], axis=1)
                xyz_world = (xyz_h @ cam2world.T)[:, :3]

        camera_pointclouds[camera_name] = np.concatenate(
            [xyz_world, rgb_flat],
            axis=1,
        ).astype(np.float32)

    return camera_rgbs, camera_pointclouds


def capture_episode(checkpoint, seed, sampler, device):
    checkpoint = Path(checkpoint).expanduser().resolve()

    print(f"Loading checkpoint: {checkpoint}")

    cp = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if "config" not in cp or "normalizer" not in cp:
        raise ValueError(
            "Checkpoint must contain embedded config and normalizer."
        )

    # ------------------------------------------------------------
    # Reconstruct the exact config stored in the checkpoint.
    # This is important for budget vs relational.
    # ------------------------------------------------------------
    cfg = OmegaConf.create(cp["config"])
    cfg.device = device

    cfg.noise_level = cfg.get(
        "noise_level",
        cfg.algo.get("noise_level", 0.7),
    )
    cfg.min_std = cfg.get(
        "min_std",
        cfg.algo.get("min_std", 0.0067),
    )

    cfg.eval.sampler = sampler

    print("\nObservation configuration:")
    print(OmegaConf.to_yaml(cfg.env))

    # ------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------
    base = build_base(cfg, device)

    cp_device = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )

    base.load_state_dict(
        policy_weights(cp_device),
        strict=True,
    )

    normalizer = MinMaxNormalizer()
    normalizer.stats = cp["normalizer"]

    actor = FlowPPOPolicy(
        base,
        num_steps=cfg.model.num_inference_steps,
        noise_level=cfg.noise_level,
        min_std=cfg.min_std,
        eval_mode=sampler,
    )

    actor.eval()

    encode = observation_encoder(
        cfg,
        actor,
        normalizer,
        device,
    )

    # Match evaluation.runner reproducibility as closely as possible.
    seed_all(seed)

    env = make_env(cfg)

    seed_all(seed)
    obs, _ = env.reset(seed=seed)

    print("\n==== Observation nested structure ====")
    print_nested("", obs)

    frames = []
    printed_camera_names = False

    done = False
    truncated = False
    policy_step = 0

    print("\nFirst observation:")
    for key, value in obs.items():
        arr = np.asarray(value)
        print(
            f"  {key:20s}"
            f" shape={str(arr.shape):15s}"
            f" dtype={arr.dtype}"
        )

    while not (done or truncated):

        # --------------------------------------------------------
        # THIS is the observation visible to the policy.
        # --------------------------------------------------------
        point_cloud = np.array(
            obs["point_cloud"],
            copy=True,
        )

        state = np.array(
            obs["state"],
            copy=True,
        )

        object_features = None
        if "object_features" in obs:
            object_features = np.array(
                obs["object_features"],
                copy=True,
            )

        # Environment RGB is only a visual reference.
        rgb_frame = prepare_rgb(env.render())

        # Extract per-camera RGB + the pre-fusion RGB-colored point clouds.
        camera_rgbs, camera_pointclouds = capture_camera_observations(env)

        if not printed_camera_names:
            print("\n==== ManiSkill observation cameras ====")
            camera_names = sorted(
                set(camera_rgbs.keys()) | set(camera_pointclouds.keys())
            )
            for name in camera_names:
                image = camera_rgbs.get(name)
                cam_pc = camera_pointclouds.get(name)
                image_desc = "None" if image is None else (
                    f"shape={image.shape}, dtype={image.dtype}"
                )
                pc_desc = "None" if cam_pc is None else (
                    f"shape={cam_pc.shape}, dtype={cam_pc.dtype}"
                )
                print(f"{name}: RGB[{image_desc}]  pointcloud[{pc_desc}]")
            printed_camera_names = True

        frames.append(
            {
                "point_cloud": point_cloud,
                "state": state,
                "object_features": object_features,
                "rgb": rgb_frame,                 # global render
                "camera_rgbs": camera_rgbs,       # per-camera RGB views
                "camera_pointclouds": camera_pointclouds,  # pre-fusion XYZRGB
                "policy_step": policy_step,
                "primitive_step": int(getattr(env, "global_step", policy_step)),
            }
        )

        # --------------------------------------------------------
        # Same policy inference as evaluation.runner
        # --------------------------------------------------------
        with torch.no_grad():
            features = encode(obs)

            actions = actor.sample(
                features,
                num_steps=cfg.model.num_inference_steps,
            )[0].cpu().numpy()

        actions = normalizer.unnormalize(
            actions,
            "action",
        )

        obs, reward, done, truncated, info = env.step(actions)

        policy_step += 1

    env.close()

    print(
        f"\nCaptured {len(frames)} policy-observation frames "
        f"for seed={seed}."
    )

    return frames, cfg


def save_frames(frames, path):
    """Optional archive for later analysis."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    pcs = np.stack(
        [f["point_cloud"] for f in frames],
        axis=0,
    )

    states = np.stack(
        [f["state"] for f in frames],
        axis=0,
    )

    kwargs = {
        "point_cloud": pcs,
        "state": states,
    }

    if all(f["rgb"] is not None for f in frames):
        try:
            kwargs["rgb"] = np.stack(
                [f["rgb"] for f in frames],
                axis=0,
            )
        except ValueError:
            pass

    if all(f["object_features"] is not None for f in frames):
        kwargs["object_features"] = np.stack(
            [f["object_features"] for f in frames],
            axis=0,
        )

    np.savez_compressed(path, **kwargs)
    print(f"Saved capture to: {path}")


def show_viewer(frames, cfg):
    if not frames:
        raise ValueError("No frames captured.")

    bounds = np.asarray(
        cfg.env.workspace_bounds,
        dtype=np.float32,
    )

    # =========================
    # Global font settings
    # =========================
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    # 2 x 3 dashboard:
    # [Base RGB] [Hand RGB] [Global render]
    # [Base PCD] [Hand PCD] [Actual policy PCD]
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.15],
        height_ratios=[0.82, 1.18],
    )

    ax_base_rgb = fig.add_subplot(gs[0, 0])
    ax_hand_rgb = fig.add_subplot(gs[0, 1])
    ax_global = fig.add_subplot(gs[0, 2])

    ax_base_pc = fig.add_subplot(gs[1, 0], projection="3d")
    ax_hand_pc = fig.add_subplot(gs[1, 1], projection="3d")
    ax_policy_pc = fig.add_subplot(gs[1, 2], projection="3d")

    plt.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.91,
        bottom=0.12,
        wspace=0.10,
        hspace=0.12,
    )

    slider_ax = fig.add_axes([0.18, 0.045, 0.64, 0.032])

    slider = Slider(
        ax=slider_ax,
        label="Frame",
        valmin=0,
        valmax=max(len(frames) - 1, 1),
        valinit=0,
        valstep=1,
    )

    slider.label.set_fontsize(16)
    slider.valtext.set_fontsize(16)

    def _setup_3d_axis(ax):
        ax.set_xlim(bounds[0, 0], bounds[1, 0])
        ax.set_ylim(bounds[0, 1], bounds[1, 1])
        ax.set_zlim(bounds[0, 2], bounds[1, 2])
        ax.set_xlabel("X", fontsize=11, labelpad=6)
        ax.set_ylabel("Y", fontsize=11, labelpad=6)
        ax.set_zlabel("Z", fontsize=11, labelpad=6)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="z", labelsize=9)

        # Make workspace geometry less distorted.
        try:
            span = bounds[1] - bounds[0]
            span = np.maximum(span, 1e-6)
            ax.set_box_aspect(span)
        except Exception:
            pass

    def _draw_camera_rgb(ax, image, title, missing_name):
        ax.clear()
        if image is not None:
            ax.imshow(image)
        else:
            ax.text(
                0.5,
                0.5,
                f"{missing_name} not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(title, fontsize=16, pad=8)
        ax.axis("off")

    def _draw_pointcloud(ax, pc, title, point_size):
        ax.clear()

        if pc is None or len(pc) == 0:
            ax.text2D(
                0.5,
                0.5,
                "point cloud not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=15, pad=10)
            _setup_3d_axis(ax)
            return

        xyz = pc[:, :3]

        scatter_kwargs = {
            "s": point_size,
            "depthshade": False,
            "linewidths": 0,
        }

        if pc.shape[1] >= 6:
            scatter_kwargs["c"] = np.clip(pc[:, 3:6], 0.0, 1.0)

        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            **scatter_kwargs,
        )

        ax.set_title(
            f"{title}\nN={pc.shape[0]:,}, shape={pc.shape}",
            fontsize=15,
            pad=10,
        )
        _setup_3d_axis(ax)

    def draw_frame(index):
        index = int(index)
        frame = frames[index]

        policy_pc = frame["point_cloud"]
        rgb_image = frame["rgb"]
        camera_rgbs = frame.get("camera_rgbs", {})
        camera_pointclouds = frame.get("camera_pointclouds", {})

        base_image = camera_rgbs.get("base_camera")
        hand_image = camera_rgbs.get("hand_camera")
        base_pc = camera_pointclouds.get("base_camera")
        hand_pc = camera_pointclouds.get("hand_camera")

        # ========================================================
        # Top row: RGB views
        # ========================================================
        _draw_camera_rgb(
            ax_base_rgb,
            base_image,
            "Base Camera RGB\n(observation sensor)",
            "base_camera",
        )
        _draw_camera_rgb(
            ax_hand_rgb,
            hand_image,
            "Hand Camera RGB\n(wrist observation sensor)",
            "hand_camera",
        )

        ax_global.clear()
        if rgb_image is not None:
            ax_global.imshow(rgb_image)
        else:
            ax_global.text(
                0.5,
                0.5,
                "env.render() returned no RGB image",
                ha="center",
                va="center",
                transform=ax_global.transAxes,
            )
        ax_global.set_title(
            "Global Environment Render\n(visual reference only)",
            fontsize=16,
            pad=8,
        )
        ax_global.axis("off")

        # ========================================================
        # Bottom row: per-camera pre-fusion PCDs + actual policy PCD
        # All are displayed in the same world/workspace coordinates.
        # ========================================================
        _draw_pointcloud(
            ax_base_pc,
            base_pc,
            "Base Camera RGB Point Cloud\n(pre-fusion, world frame)",
            point_size=2,
        )
        _draw_pointcloud(
            ax_hand_pc,
            hand_pc,
            "Hand Camera RGB Point Cloud\n(pre-fusion, world frame)",
            point_size=2,
        )

        policy_title = "Actual Policy Point Cloud\n(fused/cropped/sampled)"
        if frame["object_features"] is not None:
            policy_title += " + relational[23]"

        _draw_pointcloud(
            ax_policy_pc,
            policy_pc,
            policy_title,
            point_size=6,
        )

        fig.suptitle(
            f"Policy Observation Inspection | "
            f"Frame {index}/{len(frames)-1} | "
            f"Policy Step {frame['policy_step']}",
            fontsize=21,
            fontweight="bold",
        )

        fig.canvas.draw_idle()

    slider.on_changed(draw_frame)
    draw_frame(0)

    print(
        "\nViewer controls:\n"
        "  slider       : change frame\n"
        "  left mouse   : rotate each 3D point cloud\n"
        "  scroll       : zoom\n"
        "  close window : exit\n"
    )

    plt.show()


def print_nested(prefix, obj, depth=0, max_depth=3):
    indent = "  " * depth
    if depth > max_depth:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{indent}{prefix}{k}: {type(v)}")
            print_nested(prefix="", obj=v, depth=depth + 1, max_depth=max_depth)
    else:
        try:
            arr = np.asarray(obj)
            print(f"{indent}shape={arr.shape}, dtype={arr.dtype}")
        except Exception:
            pass
def main():
    args = parse_args()

    frames, cfg = capture_episode(
        checkpoint=args.checkpoint,
        seed=args.seed,
        sampler=args.sampler,
        device=args.device,
    )

    if args.save:
        save_frames(
            frames,
            args.save,
        )

    show_viewer(
        frames,
        cfg,
    )


if __name__ == "__main__":
    main()