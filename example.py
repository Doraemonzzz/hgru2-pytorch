import torch

from hgru2_pytorch import Hgru2_1d

n = 1024
b = 4
d = 1024
x_1d = torch.randn(b, n, d).cuda().to(torch.bfloat16)
hgru2_1d = Hgru2_1d(embed_dim=d).cuda().to(torch.bfloat16)
print(hgru2_1d)

y_1d = hgru2_1d(x_1d)
# y_2d = hgru_2d(x_2d)

print(y_1d.shape)
