import torch

from manitorch.manifolds import SO3
from scipy.spatial.transform import Rotation

xs = Rotation.random(3)
ys = Rotation.random(3)

xs_t = torch.Tensor(xs.as_matrix()).flatten(-2, -1)[:, :6]
ys_t = torch.Tensor(ys.as_matrix()).flatten(-2, -1)[:, :6]
xs_t = torch.Tensor(xs.as_matrix()).flatten(-2, -1)
ys_t = torch.Tensor(ys.as_matrix()).flatten(-2, -1)

print("xs:", xs_t)
print("ys:", ys_t)

manifold = SO3()

deltas = manifold.boxminus(ys_t, xs_t)

out = manifold.boxplus(xs_t, deltas)

print("xs:", xs_t)
print("ys:", ys_t)
print("deltas:", deltas)
print("out:", out)
print("out - ys:", (out - ys_t).abs().max())
