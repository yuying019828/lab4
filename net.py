import torch
import torch.nn as nn
import torch.nn.functional as F
#??????
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # ?????????conv1 kenrnel size=5*5????? 6
        self.conv1 = nn.Conv2d(1, 6, 5)
        # conv2 kernel size=5*5, ???? 16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # ????
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # max-pooling ???? (2,2) ?????
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # ?(kernel)?????????????????? (2,2) ? 2 ??
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        # ?? batch ????????
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
params = list(net.parameters())
print('The number of parameters: ', len(params))
# conv1.weight
print('The size of conv1.weight: ', params[0].size())
# ????????????
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
# ??????????????????????????
net.zero_grad()
out.backward(torch.randn(1, 10))

# ??????
output = net(input)
# ?????
target = torch.randn(10)
# ???????? output ??? size
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
# MSELoss
print(loss.grad_fn)
# Linear layer
print(loss.grad_fn.next_functions[0][0])
# Relu
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# ????
# ???????????
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# ????
# ???????????
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
import torch.optim as optim
# ?????
optimizer = optim.SGD(net.parameters(), lr=0.01)

# ????????????
optimizer.zero_grad() # ??????
output = net(input)
loss = criterion(output, target)
loss.backward()
# ????
optimizer.step()