import torch
import numpy as np

x = torch.empty(2, 2, 2, 3)
x = torch.rand(2, 2)
x = torch.ones(2, 2, dtype=torch.float16)
print(x.dtype)
print(x.size())
x = torch.tensor([2.5, 0.1])
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
z = x + y
z = torch.add(x, y)
print(z)
y.add_(x) # modify the value of y
print(y)
z = torch.sub(x, y) # equal to z = x - y
z = torch.mul(x, y)
z = torch.div(x, y)
x = torch.rand(5, 3)
print(x)
print(x[:, 0])
print(x[1, :])
print(x[1,1].item()) # get only the value
# reshape tensors
x = torch.rand(4, 4)
print(x)
y = x.view(16)
y = x.view(-1, 8)
print(y.size())
print(y)
# convert numpy to torch
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))
# if on CPU the above a and b share the same memory, which means if one is changed the other one is also changed.
# convert numpy to torch
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
# gradient calculation is needed
x = torch.ones(5, requires_grad=True)