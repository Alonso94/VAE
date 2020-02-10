import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
print(torch.__version__)

import torchvision
import torchvision.transforms as TF

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. load the CIFAR data set and normalize
# compose several transforms together
# ToTensor() convert a PIL image or np.ndarray to tensor
# Normalize(mean,std)
# ps: we can use trasnform.resize, Grayscale
# transforms.lambda(lambda) functional transform
transform=TF.Compose([TF.ToTensor(),TF.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# show samples of the images
def imshow(img):
    img=img/2+0.5 # unnormalize
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

# get some random training images
# dataiter=iter(trainloader)
# images,labels=dataiter.next()
# show images and labels
# imshow(torchvision.utils.make_grid(images))
# print(' '.join("%s" % classes[labels[j]] for j in range(4)))

# define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

net=Net().to(device)

Train=False
path='./cifar_net.pth'
if Train:
    # define the loss
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

    # train the network
    for epoch in range(20):
        running_loss=0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels=data[0].to(device),data[1].to(device)
            optimizer.zero_grad()

            output=net(inputs)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            if i%2000==1999:
                print("[%5d,%5d] loss : %.3f" % (epoch+1,i+1,running_loss/2000))
                running_loss=0.0

    print("The end")
    # save the model
    torch.save(net.state_dict(),path)

else:
    dataiter=iter(testloader)
    images,labels=dataiter.next()[0].to(device),dataiter.next()[1].to(device)

    imshow(torchvision.utils.make_grid(images.cpu()))
    print('Ground truth:',' '.join("%5s" % classes[labels[j]]for j in range(4)))

    net.load_state_dict(torch.load(path))
    output=net(images)
    _,predicted=torch.max(output,1)
    print("predicted",' '.join("%5s" % classes[labels[j]]for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device),data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))