from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import pyperlin
import matplotlib.pyplot as plt

from model import _netG

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/val', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)

###############################################
###              MASK OPTIONS               ###
### Uncomment the masks to use for testing  ###
###############################################

# RANDOM RECTANGLE
# x_one = np.random.randint(1, 128)
# x_two = np.random.randint(1, 128)
# x1 = min(x_one, x_two)
# x2 = max(x_one, x_two)

# y_one = np.random.randint(1, 128)
# y_two = np.random.randint(1, 128)
# y1 = min(y_one, y_two)
# y2 = max(y_one, y_two)

# HARDCODED MASK
# regular_square = [int(opt.imageSize/4), int(opt.imageSize/4+opt.imageSize/2), int(opt.imageSize/4), int(opt.imageSize/4+opt.imageSize/2)]
# small_central_square = [int(opt.imageSize/3), int((2/3) * opt.imageSize),  int(opt.imageSize/3),  int((2/3) * opt.imageSize)]
# big_rectangle = [30, 80, 20, 110]
# long_rectangle = [60, 75, 30, 100]
# small_square = [60, 75, 30, 45]

# options = {"regular_square": regular_square, "small_central_square": small_central_square, "big_rectangle": big_rectangle, "long_rectangle": long_rectangle, "small_square": small_square}

# #Set this variable to choose the shape
# option = options["long_rectangle"]

# x1 = option[0]
# x2 = option[1]
# y1 = option[2]
# y2 = option[3]


# mask = np.zeros((128, 128))
# mask[x1:x2, y1:y2] = 1
            
# RANDOM MASK
# shape_mask = (128, 128) # Size of the masks. Use powers of 2.
# num_masks = 1 # Num of different masks
# persistance = .4 # Controls the smoothness of the stains' boundaries. Should be float > 0. In practice, < 1
# threshold = .8 # More or less controls the area of the stains 

# # Mask generation
# output_size = (num_masks, shape_mask[0], shape_mask[1])
# generator = torch.Generator()
# generator.manual_seed(0)

# octaves = 5 # Controls level of detail. Should be integer 1-9, depending on the mask shape
# resolutions = [(2 ** i, 2 ** i) for i in range(1, octaves + 1)]
# factors = [persistance ** i for i in range(octaves)]
# fp = pyperlin.FractalPerlin2D(output_size, resolutions, factors, generator=generator)
# noise = fp().numpy()

# mask = np.zeros((128, 128))
# generation = noise[0]
# generation = generation - np.min(generation)
# generation = generation/np.max(generation)
# generation_t = (generation > .8).astype(np.uint8)
# mask = generation_t

# CIRCLE MASK
mask = np.zeros((128, 128))
center_x = 64
center_y = 64
r = 20

for i in range(128):
    for j in range(128):
        if (i - center_x)**2 + (j - center_y)**2 < r**2:
            mask[i][j] = 1

# Initialize generator 
netG = _netG(opt)
# Loading the trained state into the generator t
netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
netG.eval()

# Creating noise image?? 
transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

 
input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)


# Defining the MSE loss 
criterionMSE = nn.MSELoss()

# Setting up CUDA if using it 
if opt.cuda:
    netG.cuda()
    input_real, input_cropped = input_real.cuda(),input_cropped.cuda()
    criterionMSE.cuda()
    # real_center = real_center.cuda()

# Creating variables 
input_real = Variable(input_real)
input_cropped = Variable(input_cropped)

# Creating an iterable for the data
dataiter = iter(dataloader)
real_cpu, _ = next(dataiter)

# Copying data to the CPU
input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)

# Cropping image with the mask 
input_cropped.data[:,:, mask==1] = 1.0
        
# Generate a fake image using a cropped input
fake = netG(input_cropped, mask)

# Error of the Generator 
errG = criterionMSE(fake,input_real)

# Reconstructed image is a clone of the cropped image 
recon_image = fake.clone()

# Save image results
vutils.save_image(real_cpu,'val_real_samples.png',normalize=True)
vutils.save_image(input_cropped.data,'val_cropped_samples.png',normalize=True)
vutils.save_image(recon_image,'val_recon_samples.png',normalize=True)
p=0
l1=0
l2=0
fake = fake.data.numpy()
input_real = input_real.data.numpy()
from psnr import psnr
import numpy as np

t = input_real - fake
l2 = np.mean(np.square(t))
l1 = np.mean(np.abs(t))
input_real = (input_real+1)*127.5
fake = (fake+1)*127.5


for i in range(opt.batchSize):
    p = p + psnr(input_real[i] , fake[i])

print(l2)

print(l1)

print(p/opt.batchSize)
