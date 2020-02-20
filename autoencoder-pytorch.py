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
transform = TF.Compose([TF.ToTensor(), TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# show samples of the images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.image_dim=32*32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True)
        )
        self.fc1 = nn.Linear(16 * 3 * self.image_dim, 512)
        self.fc2 = nn.Linear(512, 16 * 3 * self.image_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, 5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, 5),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        shape=x.shape
        x = x.view(-1, 16 * 3 * self.image_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(shape)
        x = self.decoder(x)
        return x

model=autoencoder().to(device)
# print(model)
mse=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),weight_decay=1e-5)
for epoch in range(10):
    for data in trainloader:
        img=data[0].to(device)
        output=model(img)
        loss=mse(output,img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch %d : loss %.4f" %(epoch+1,loss.item()))
    output = model(img)
    output = output[0].cpu()
    img = img[0].cpu()
    imshow(img)
    imshow(output)