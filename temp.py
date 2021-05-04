import torch
from torch.distributions import Normal

# m = MultivariateNormal(torch.zeros(2), torch.eye(2))
# print(m.covariance_matrix)
# v = m.sample()
# print(v)
# print(m.log_prob(v))

mean = torch.tensor([[0.0, 1.0], [1.0, 10], [11.0, 88]])
var = torch.tensor([[123.0, 1.0], [1.0, 10], [11.0, 11]])

print(mean.size())
m = Normal(mean, var)
v = m.sample()
log_prob = m.log_prob(v)
prob = log_prob.exp()

print(v)
print(log_prob)
print(prob)

mean = torch.tensor([0.0])
var = torch.tensor([123.0])

m1 = Normal(mean, var)
v1 = v[0][0]
log_prob1 = m1.log_prob(v1)
prob1 = log_prob1.exp()

print(v1)
print(log_prob1)
print(prob1)
print("---------")

mean = torch.tensor([1.0])
var = torch.tensor([1.0])
#
# m1 = Normal(mean, var)
# v1 = v[0][1]
# log_prob1 = m1.log_prob(v1)
# prob1 = log_prob1.exp()
#
# print(v1)
# print(log_prob1)
# print(prob1)
# print("---------")
#
# mean = torch.tensor([10.0])
# var = torch.tensor([10.0])
#
# m1 = Normal(mean, var)
# v1 = v[1][1]
# log_prob1 = m1.log_prob(v1)
# prob1 = log_prob1.exp()
#
# print(v1)
# print(log_prob1)
# print(prob1)
#
#
# mean = torch.tensor([[0.0, 1.0], [1.0, 10], [11.0, 88]])
# var = torch.tensor([[123.0, 1.0], [1.0, 10], [11.0, 11]])
#
#
# print(mean.size())
# m = Normal(mean, var)
# v = m.sample()
# log_prob = m.log_prob(v)
# prob = log_prob.exp()
#
# print(v)
# print(log_prob)
# print(prob)
#
#
# mean = torch.tensor([[1.0, 1.0], [1.0, 10.0], [11.0, 12.0]])
# print(torch.prod(mean, dim=1))
#
# x = torch.tensor([10, 4, 15])
# y = torch.tensor([5, 2, 5])
# print(x/y)

x = []

for i in range(10):
    a = torch.zeros(100, 4)
    print(a)
    x.append(a)

t_b_output = torch.cat(x, dim=0)
print(t_b_output)
print(t_b_output.size())