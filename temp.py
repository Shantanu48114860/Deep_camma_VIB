import torch
from torch.distributions import MultivariateNormal

m = MultivariateNormal(torch.zeros(2), torch.eye(2))
print(m.covariance_matrix)
v = m.sample()
print(v)
print(m.log_prob(v))