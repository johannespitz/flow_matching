import time

import torch

# flow_matching
from flow_matching.path import CompositeGeodesicProbPath, GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import RiemannianODESolver
from flow_matching.utils import ModelWrapper
from flow_matching.utils.manifolds import Composite, Manifold, Sphere

from my_circles.data import inf_gt_gen, inf_noise_gen, plot_orientations

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
    ):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(input_dim + time_dim, hidden_dim)

        self.main = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        h = self.input_layer(h)
        output = self.main(h)

        return output.reshape(*sz)


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield: nn.Module, manifold: Manifold):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.manifold.projx(x)
        v = self.vecfield(x, t)
        v = self.manifold.proju(x, v)
        return v


# training arguments
lr = 0.001
batch_size = 1024  # 4096
iterations = 5  # 5001
print_every = 1000
manifold = Composite([Sphere()] * trajectory_length)
if DEBUG_ORIG:
    manifold = Sphere()
dim = 4
hidden_dim = 512

# velocity field model init
vf = ProjectToTangent(  # Ensures we can just use Euclidean divergence.
    MLP(  # Vector field in the ambient space.
        input_dim=dim,
        hidden_dim=hidden_dim,
    ),
    manifold=manifold,
)
vf.to(device)

# instantiate an affine path object
path = CompositeGeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)
if DEBUG_ORIG:
    path = GeodesicProbPath(scheduler=CondOTScheduler(), manifold=manifold)

# init optimizer
optim = torch.optim.Adam(vf.parameters(), lr=lr)

# train
start_time = time.time()
for i in range(iterations):
    optim.zero_grad()

    x_1 = inf_gt_gen(batch_size=batch_size, length=trajectory_length).to(device)
    x_0 = inf_noise_gen(batch_size=batch_size, length=trajectory_length).to(device)

    if DEBUG_ORIG:
        # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
        x_1 = inf_train_gen(batch_size=batch_size, device=device)  # sample data
        x_0 = torch.randn_like(x_1).to(device)
        x_1 = wrap(manifold, x_1)
        x_0 = wrap(manifold, x_0)

    # sample time (user's responsibility)
    t = torch.rand(x_1.shape[0]).to(device)

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

plot_orientations(sol)
