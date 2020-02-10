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

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(True),
            nn.Conv2d(6,16,5),
            nn.ReLU(True)
        )
        self.fc1=nn.Linear(16*5*5,64)
        self.fc2=nn.Linear(64,16*5*5)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(6,16,5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,5),
            nn.ReLU(True)
        )

    def forward(self,x):
        x=self.encoder(x)
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=x.view(16,5,5)
        x=self.decoder(x)
        return x