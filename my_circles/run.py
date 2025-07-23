import time

import torch

# flow_matching
from flow_matching.path import SO3ProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import SO3

from my_circles.data import inf_gt_gen, inf_joint_gen, inf_noise_gen, plot_orientations

from torch import nn, Tensor


DEBUG_ORIG = True
if DEBUG_ORIG:

    def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
        x1 = torch.rand(batch_size, device=device) * 2 - 2
        x3 = torch.rand(batch_size, device=device) * 0.01 - 2
        x2_ = (
            torch.rand(batch_size, device=device)
            - torch.randint(high=2, size=(batch_size,), device=device) * 2
        )
        x2 = x2_ + (torch.floor(x1) % 2)

        data = torch.cat([x1[:, None], x2[:, None], x3[:, None]], dim=1)

        return data.float()

    def wrap(manifold, samples):
        center = torch.cat(
            [torch.zeros_like(samples), torch.ones_like(samples[..., 0:1])], dim=-1
        )
        samples = torch.cat([samples, torch.zeros_like(samples[..., 0:1])], dim=-1) / 2

        return manifold.expmap(center, samples)


trajectory_length = 20

from matplotlib import cm

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using gpu")
else:
    device = "cpu"
    print("Using cpu.")

torch.manual_seed(42)


# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        time_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        if output_dim is None:
            output_dim = input_dim

        self.input_layer = nn.Linear(input_dim + time_dim, hidden_dim)

        self.main = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.shape[:-1]
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        h = self.input_layer(h)
        output = self.main(h)

        return output.reshape(*sz, -1)


# training arguments
lr = 0.001
batch_size = 1024  # 4096
iterations = 5001  # 5001
# iterations = 1
print_every = 100
manifold = SO3()
hidden_dim = 512

# velocity field model init
vf = MLP(
    input_dim=9 * trajectory_length,
    hidden_dim=hidden_dim,
    output_dim=3 * trajectory_length,
)

vf.to(device)

# instantiate an affine path object
path = SO3ProbPath(scheduler=CondOTScheduler())
# init optimizer
optim = torch.optim.Adam(vf.parameters(), lr=lr)

# train
start_time = time.time()
for i in range(iterations):
    optim.zero_grad()

    x_1 = inf_gt_gen(batch_size=batch_size, length=trajectory_length).to(device)
    x_0 = inf_noise_gen(batch_size=batch_size, length=trajectory_length).to(device)

    x_0, x_1 = inf_joint_gen(batch_size=batch_size, length=trajectory_length)
    x_0.to(device)
    x_1.to(device)

    # sample time (user's responsibility)
    t = torch.rand(batch_size).to(device)

    # Use GeodesicProbPath to generate trajectory sample
    # test_x0 = x_0.clone()[:20]
    # test_x1 = x_1.clone()[:20]
    # test_x0[:] = test_x0[0]
    # test_x1[:] = test_x1[0]
    # test_t = torch.linspace(0, 1, 20).to(device)
    # test_samples = path.sample(t=test_t, x_0=test_x0, x_1=test_x1)

    # sample probability path
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    # flow matching l2 loss
    loss = torch.pow(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()

    # optimizer step
    loss.backward()  # backward
    optim.step()  # update

    # log loss
    if (i + 1) % print_every == 0:
        elapsed = time.time() - start_time
        print(
            "| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ".format(
                i + 1, elapsed * 1000 / print_every, loss.item()
            )
        )
        start_time = time.time()


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x=x, t=t)


wrapped_vf = WrappedModel(vf)

# step size for ode solver
step_size = 0.01
N = 6

norm = cm.colors.Normalize(vmax=50, vmin=0)

batch_size = 5  # 50000  # batch size
eps_time = 1e-2
T = torch.linspace(0, 1, N)  # sample times
T = T.to(device=device)

# initial conditions
x_init = inf_noise_gen(batch_size=batch_size, length=trajectory_length).to(device)
gt_samples = inf_gt_gen(batch_size=batch_size, length=trajectory_length)
x_init, gt_samples = inf_joint_gen(batch_size=batch_size, length=trajectory_length)

solver = RiemannianODESolver(
    velocity_model=wrapped_vf, manifold=manifold
)  # create an ODESolver class
sol = solver.sample(
    x_init=x_init,
    step_size=step_size,
    method="midpoint",
    return_intermediates=True,
    time_grid=T,
    verbose=True,
)

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# plot_orientations(sol[-1, 0], type="matrix")

# plt.show()


samples = torch.cat([sol, gt_samples[None]], dim=0).numpy()


for j in range(batch_size):
    _, axs = plt.subplots(1, N + 1, figsize=(20, 3.2), subplot_kw={"projection": "3d"})
    for i in range(N + 1):

        # plot_orientations(samples[i], axs[i], offset=0.0)

        plot_orientations(
            Rotation.from_matrix(samples[i, j].reshape(-1, 3, 3)).as_quat(),
            axs[i],
            offset=0.0,
        )

        # Set the aspect ratio to equal for better visualization of a sphere
        axs[i].set_box_aspect([1, 1, 1])
        axs[i].view_init(elev=130, azim=0)
        axs[i].axis("off")

        print(i)

    plt.tight_layout()
    plt.show()
    plt.close()
