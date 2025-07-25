import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1,stride=1)
    self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=1)
    self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
    self.fc1 = nn.Linear(in_features=32*7*7,out_features=128)
    self.fc2= nn.Linear(in_features=128,out_features=10)

  def forward(self,x):
    x= self.pool(F.relu(self.conv1(x)))
    x= self.pool(F.relu(self.conv2(x)))

    x= x.view(-1, 32*7*7)
    x = F.relu(self.fc1(x))
    x= self.fc2(x)
    return x
