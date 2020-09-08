# hunkim's inception module + modify(add module)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from pathlib import Path


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Training settings
batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                       download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#%matplotlib inline
# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(train_loader)
images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid(images[0]))
#print(' '.join('%5s' % classes[labels[j]] for j in range(2)))

class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 48, kernel_size=1) #16*3, 24*3

        self.branch5x5_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(48, 72, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 48, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(48, 72, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 72, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 30, kernel_size=5) #10*3
        self.conv2 = nn.Conv2d(30, 60, kernel_size=5) #88*3
        self.conv3 = nn.Conv2d(264, 60, kernel_size=5) #88*3

        self.incept1 = InceptionA(in_channels=60)
        self.incept2 = InceptionA(in_channels=264)

        self.mp1 = nn.MaxPool2d(1)
        self.mp2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(264, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp2(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        x = self.incept1(x)
        x = self.incept2(x)
        x = F.relu(self.mp1(self.conv3(x)))
        x = self.incept1(x)
        x = self.incept2(x)
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()
load_model = Path("./model/inception_cifar_CNN_modify")
if load_model.is_file():
    model.load_state_dict(torch.load('./model/inception_cifar_CNN_modify'))
    print("load saved model file")

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())) #data[0] -> item()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item() #data[0] -> item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
    torch.save(model.state_dict(), "./model/inception_cifar_CNN_modify")
    print("model saved complete")
