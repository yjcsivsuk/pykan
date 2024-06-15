import torch
from kan import KAN

torch.set_default_dtype(torch.float64)
model=KAN(width=[3,3,1])
dx=torch.randn(4,6,14,14)
dy=torch.randn(4,6,14,14)
dxdy=torch.randn(4,6,14,14)
X=[]
X.append(dx.flatten())
X.append(dy.flatten())
X.append(dxdy.flatten())
input=X
output=model(input)
print(output.shape)