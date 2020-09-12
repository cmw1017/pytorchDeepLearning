import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob
import PIL
from PIL import Image
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
import random

# CUDA 사용시 device setting
print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# set hyper-parameter
batch_size = 64
validation_ratio = 0.1
random_seed = 10
initial_lr = 0.1
num_epoch = 300

# 데이터 normalize
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)

# print(trainset.data.shape)  # train_data => data로 변경됨

train_data_mean = trainset.data.mean(axis=(0, 1, 2))
train_data_std = trainset.data.std(axis=(0, 1, 2))

# print(train_data_mean)
# print(train_data_std)

train_data_mean = train_data_mean / 255
train_data_std = train_data_std / 255

# Normalize value
print(train_data_mean)
print(train_data_std)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# batch_norm + relu + convolution
class bn_relu_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride, padding, bias=False):
        super(bn_relu_conv, self).__init__()
        self.batch_norm = nn.BatchNorm2d(nin)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.batch_norm(x)
        out = self.relu(out)
        out = self.conv(out)

        return out


# ResNet의 Bottleneck과 동일 개념
class bottleneck_layer(nn.Sequential):
    def __init__(self, nin, growth_rate, drop_rate=0.2):
        super(bottleneck_layer, self).__init__()

        self.add_module('conv_1x1',
                        bn_relu_conv(nin=nin, nout=growth_rate * 4, kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('conv_3x3',
                        bn_relu_conv(nin=growth_rate * 4, nout=growth_rate, kernel_size=3, stride=1, padding=1,
                                     bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        bottleneck_output = super(bottleneck_layer, self).forward(x)
        if self.drop_rate > 0:
            bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)

        bottleneck_output = torch.cat((x, bottleneck_output), 1)

        return bottleneck_output


class Transition_layer(nn.Sequential):
    def __init__(self, nin, theta=0.5):
        super(Transition_layer, self).__init__()

        self.add_module('conv_1x1',
                        bn_relu_conv(nin=nin, nout=int(nin * theta), kernel_size=1, stride=1, padding=0, bias=False))
        self.add_module('avg_pool_2x2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))


class DenseBlock(nn.Sequential):
    def __init__(self, nin, num_bottleneck_layers, growth_rate, drop_rate=0.2):
        super(DenseBlock, self).__init__()

        for i in range(num_bottleneck_layers):
            nin_bottleneck_layer = nin + growth_rate * i
            self.add_module('bottleneck_layer_%d' % i,
                            bottleneck_layer(nin=nin_bottleneck_layer, growth_rate=growth_rate, drop_rate=drop_rate))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, num_layers=100, theta=0.5, drop_rate=0.2, num_classes=10):
        super(DenseNet, self).__init__()

        assert (num_layers - 4) % 6 == 0

        # (num_layers-4)//6
        num_bottleneck_layers = (num_layers - 4) // 6

        # 32 x 32 x 3
        # --> 32 x 32 x (growth_rate*2)
        self.dense_init = nn.Conv2d(3, growth_rate * 2, kernel_size=3, stride=1, padding=1, bias=True)

        # 32 x 32 x (growth_rate*2)
        # --> 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]
        self.dense_block_1 = DenseBlock(nin=growth_rate * 2, num_bottleneck_layers=num_bottleneck_layers,
                                        growth_rate=growth_rate, drop_rate=drop_rate)

        # 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]
        # --> 16 x 16 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]*theta
        nin_transition_layer_1 = (growth_rate * 2) + (growth_rate * num_bottleneck_layers)
        self.transition_layer_1 = Transition_layer(nin=nin_transition_layer_1, theta=theta)

        # 16 x 16 x nin_transition_layer_1*theta
        # --> 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_2 = DenseBlock(nin=int(nin_transition_layer_1 * theta),
                                        num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate,
                                        drop_rate=drop_rate)

        # 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]
        # --> 8 x 8 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]*theta
        nin_transition_layer_2 = int(nin_transition_layer_1 * theta) + (growth_rate * num_bottleneck_layers)
        self.transition_layer_2 = Transition_layer(nin=nin_transition_layer_2, theta=theta)

        # 8 x 8 x nin_transition_layer_2*theta
        # --> 8 x 8 x [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_3 = DenseBlock(nin=int(nin_transition_layer_2 * theta),
                                        num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate,
                                        drop_rate=drop_rate)

        nin_fc_layer = int(nin_transition_layer_2 * theta) + (growth_rate * num_bottleneck_layers)

        # [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)]
        # --> num_classes
        self.fc_layer = nn.Linear(nin_fc_layer, num_classes)

    def forward(self, x):
        dense_init_output = self.dense_init(x)

        dense_block_1_output = self.dense_block_1(dense_init_output)
        transition_layer_1_output = self.transition_layer_1(dense_block_1_output)

        dense_block_2_output = self.dense_block_2(transition_layer_1_output)
        transition_layer_2_output = self.transition_layer_2(dense_block_2_output)

        dense_block_3_output = self.dense_block_3(transition_layer_2_output)

        global_avg_pool_output = F.adaptive_avg_pool2d(dense_block_3_output, (1, 1))
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)

        output = self.fc_layer(global_avg_pool_output_flat)

        return output


def DenseNetBC_100_12():
    return DenseNet(growth_rate=12, num_layers=100, theta=0.5, drop_rate=0.2, num_classes=10)


def DenseNetBC_250_24():
    return DenseNet(growth_rate=24, num_layers=250, theta=0.5, drop_rate=0.2, num_classes=10)


def DenseNetBC_190_40():
    return DenseNet(growth_rate=40, num_layers=190, theta=0.5, drop_rate=0.2, num_classes=10)


DenseNet_N = DenseNetBC_100_12()
# DenseNet_N = DenseNetBC_190_40()
# DenseNet_N = DenseNetBC_250_24()
DenseNet_N.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(DenseNet_N.parameters(), lr=initial_lr, momentum=0.9)
# 이 예제는 StepLR이 아닌 MultiStepLR 사용
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(num_epoch * 0.5), int(num_epoch * 0.75)], gamma=0.1, last_epoch=-1)


def acc_check(net, test_set, epoch, save=1):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (100 * correct / total)
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)
    if save:
        torch.save(net.state_dict(), "./model/DenseNet_100_12_epoch_{}_acc_{}.pth".format(epoch, int(acc)))
    return acc


print(len(trainloader))
epochs = 60

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = DenseNet_N(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:  # print every 5 mini-batches
            print('[%d, %5d, %d%%] loss: %.3f' %
                  (epoch + 1, i + 1, 100 * (i + 1) / len(trainloader), running_loss / 5))
            running_loss = 0.0
    lr_scheduler.step()
    # Check Accuracy
    if epoch % 20 == 19:
        acc = acc_check(DenseNet_N, testloader, epoch, save=1)
    else:
        acc = acc_check(DenseNet_N, testloader, epoch, save=0)

print('Finished Training')