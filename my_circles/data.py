# visualization
import matplotlib.pyplot as plt
import torch

from scipy.spatial.transform import Rotation


# x1: shortest roation from start to goal
# x0: random orientaions (later: fixed start (and goal))


def inf_gt_gen(batch_size=20, length=20):
    starts = Rotation.random(batch_size)
    goals = Rotation.random(batch_size)

    delta = starts.inv() * goals

    # intermetidate = [starts + (i / length) * delta for i in range(length + 1)]

    out = torch.stack(
        [
            torch.tensor(
                (starts * delta ** (i / (length - 1))).as_matrix(), dtype=torch.float32
            ).flatten(-2, -1)
            for i in range(length)
        ],
        dim=1,
    )

    return out


def inf_noise_gen(batch_size=20, length=20):
    rots = Rotation.random(batch_size * length)
    return torch.tensor(rots.as_matrix(), dtype=torch.float32).reshape(
        batch_size, length, 9
    )


def inf_joint_gen(batch_size=20, length=20):
    starts = Rotation.random(batch_size)
    goals = Rotation.random(batch_size)

    delta = starts.inv() * goals

    # intermetidate = [starts + (i / length) * delta for i in range(length + 1)]

    x_1 = torch.stack(
        [
            torch.tensor(
                (starts * delta ** (i / (length - 1))).as_matrix(), dtype=torch.float32
            ).flatten(-2, -1)
            for i in range(length)
        ],
        dim=1,
    )

    rots = Rotation.random(batch_size * (length - 2))
    x_0 = torch.cat(
        [
            torch.tensor(starts.as_matrix(), dtype=torch.float32).flatten(-2, -1)[
                :, None
            ],
            torch.tensor(rots.as_matrix(), dtype=torch.float32).reshape(
                batch_size, (length - 2), 9
            ),
            torch.tensor(goals.as_matrix(), dtype=torch.float32).flatten(-2, -1)[
                :, None
            ],
        ],
        dim=1,
    )

    return x_0, x_1


def plot_orientations(orientations, ax=None, offset=0.001, type="quaternion"):
    # very small such that the axis scaling is not affected

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    for i, orientation in enumerate(orientations):
        if i == 0 or i == len(orientations) - 1:
            linewidth = 2.0
        else:
            linewidth = 0.5
        if type == "quaternion":
            r = Rotation.from_quat(orientation)
        elif type == "matrix":
            r = Rotation.from_matrix(orientation.reshape(-1, 3, 3))[0]
        length = 0.05
        x, y, z = r.apply([1, 0, 0]), r.apply([0, 1, 0]), r.apply([0, 0, 1])
        for axis, color in zip([x, y, z], ["r", "g", "b"]):
            ax.quiver(
                i * offset,
                0,
                0,
                axis[0],
                axis[1],
                axis[2],
                color=color,
                length=length,
                linewidth=linewidth,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Orientations in 3D")
    limit = 0.04
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect
    # plt.show()


# torch.manual_seed(42)

# print(inf_gt_gen(batch_size=2, length=20))
# plot_orientations(inf_gt_gen(batch_size=2, length=20)[0])
# plot_orientations(inf_noise_gen(batch_size=2, length=20)[0])


# # tensor([[-0.1049,  0.4089, -0.3777,  0.8241],
# # tensor([[-0.6336,  0.1415, -0.7258,  0.2276],
# length = 20
# start = Rotation.from_quat([-0.1049, 0.4089, -0.3777, 0.8241])
# goal = Rotation.from_quat([-0.6336, 0.1415, -0.7258, 0.2276])
# # start * delta = goal
# delta = start.inv() * goal
# # intermetidate = [starts + (i / length) * delta for i in range(length + 1)]
# out = torch.stack(
#     [
#         torch.tensor((start * delta ** (i / (length - 1))).as_quat())
#         for i in range(length)
#     ],
#     dim=0,
# )
# plot_orientations(out, offset=0.0)
# print(out)
# plt.show()
