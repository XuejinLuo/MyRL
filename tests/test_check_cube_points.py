import h5py
import numpy as np


H5_PATH = (
    "/home/luo/.maniskill/demos/StackCube-v1/"
    "motionplanning/"
    "trajectory.pointcloud.pd_ee_delta_pose.physx_cpu.h5"
)

NUM_POINTS = 1024
MAX_EPISODES = 100
FRAME_STEP = 5

# 改成你上面实际打印出来的 ID
CUBE_A_ID = 18
CUBE_B_ID = 19

WORKSPACE_BOUNDS = np.array(
    [
        [-0.5, -0.5, 0.0],
        [ 0.5,  0.5, 0.5],
    ],
    dtype=np.float32,
)


def sample_points_with_seg(
    xyzw,
    segmentation,
    num_points=1024,
):
    """
    完全模拟当前 preprocess_points 的：
    valid -> workspace crop -> random sample

    同时保留 segmentation label。
    """

    xyzw = np.asarray(xyzw)
    seg = np.asarray(segmentation).reshape(-1)

    xyz = xyzw[..., :3].reshape(-1, 3)
    valid_w = xyzw[..., 3].reshape(-1)

    # ManiSkill 有效点
    mask = valid_w > 0

    # finite
    mask &= np.isfinite(xyz).all(axis=-1)

    # workspace crop
    mask &= (
        (xyz >= WORKSPACE_BOUNDS[0])
        & (xyz <= WORKSPACE_BOUNDS[1])
    ).all(axis=-1)

    xyz = xyz[mask]
    seg = seg[mask]

    n_before = len(xyz)

    if n_before == 0:
        return (
            np.zeros((num_points, 3)),
            np.zeros(num_points, dtype=seg.dtype),
            0,
        )

    indices = np.random.choice(
        n_before,
        num_points,
        replace=n_before < num_points,
    )

    return xyz[indices], seg[indices], n_before


def describe(name, values):
    values = np.asarray(values)

    print(f"\n{name}")
    print("-" * 55)

    for key, value in [
        ("mean", values.mean()),
        ("std", values.std()),
        ("min", values.min()),
        ("p01", np.percentile(values, 1)),
        ("p05", np.percentile(values, 5)),
        ("p10", np.percentile(values, 10)),
        ("median", np.median(values)),
        ("p90", np.percentile(values, 90)),
        ("max", values.max()),
    ]:
        print(f"{key:8s}: {value:.2f}")


def main():

    cube_a_before = []
    cube_b_before = []

    cube_a_after = []
    cube_b_after = []

    total_before = []

    with h5py.File(H5_PATH, "r") as f:

        episodes = list(f.keys())[:MAX_EPISODES]

        total_frames = 0

        for ep_name in episodes:

            cloud = f[ep_name]["obs"]["pointcloud"]

            xyzw_all = cloud["xyzw"]
            seg_all = cloud["segmentation"]

            T = len(xyzw_all)

            for t in range(0, T, FRAME_STEP):

                xyzw = np.asarray(xyzw_all[t])
                seg = np.asarray(
                    seg_all[t]
                ).reshape(-1)

                xyz = xyzw[..., :3].reshape(-1, 3)
                w = xyzw[..., 3].reshape(-1)

                # =====================================================
                # 下采样前：同样先 valid + workspace crop
                # =====================================================

                mask = w > 0

                mask &= np.isfinite(
                    xyz
                ).all(axis=-1)

                mask &= (
                    (xyz >= WORKSPACE_BOUNDS[0])
                    & (xyz <= WORKSPACE_BOUNDS[1])
                ).all(axis=-1)

                seg_crop = seg[mask]

                a_before = np.sum(
                    seg_crop == CUBE_A_ID
                )

                b_before = np.sum(
                    seg_crop == CUBE_B_ID
                )

                cube_a_before.append(a_before)
                cube_b_before.append(b_before)

                # =====================================================
                # 随机 1024
                # =====================================================

                _, seg_sampled, n_before = (
                    sample_points_with_seg(
                        xyzw,
                        seg,
                        NUM_POINTS,
                    )
                )

                a_after = np.sum(
                    seg_sampled == CUBE_A_ID
                )

                b_after = np.sum(
                    seg_sampled == CUBE_B_ID
                )

                cube_a_after.append(a_after)
                cube_b_after.append(b_after)

                total_before.append(n_before)

                total_frames += 1

    print("=" * 70)
    print("StackCube TRUE segmentation statistics")
    print("=" * 70)

    print("episodes     :", len(episodes))
    print("frames       :", total_frames)
    print("sample size  :", NUM_POINTS)

    describe(
        "Total cropped points BEFORE sampling",
        total_before,
    )

    describe(
        "Cube A BEFORE sampling",
        cube_a_before,
    )

    describe(
        "Cube B BEFORE sampling",
        cube_b_before,
    )

    describe(
        "Cube A AFTER random 1024",
        cube_a_after,
    )

    describe(
        "Cube B AFTER random 1024",
        cube_b_after,
    )

    a = np.asarray(cube_a_after)
    b = np.asarray(cube_b_after)

    print("\nLow-point frames AFTER 1024 sampling")
    print("-" * 55)

    for n in [1, 5, 10, 20, 30, 50, 100]:

        print(
            f"< {n:3d}: "
            f"cubeA={np.mean(a < n)*100:6.2f}%  "
            f"cubeB={np.mean(b < n)*100:6.2f}%"
        )

    print("\nZero point frames")
    print("-" * 55)

    print(
        "cubeA:",
        np.mean(a == 0) * 100,
        "%"
    )

    print(
        "cubeB:",
        np.mean(b == 0) * 100,
        "%"
    )


if __name__ == "__main__":
    main()