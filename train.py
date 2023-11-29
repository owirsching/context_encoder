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
import matplotlib.pyplot as plt
from torchsummary import summary 
import pyperlin

from model import _netlocalD,_netG
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
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
parser.add_argument('--wtl2',type=float,default=0.998,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')

opt = parser.parse_args()
print(opt)


# RANDOM RECTANGLES
# x_one = np.random.randint(1, 128)
# x_two = np.random.randint(1, 128)
# x1 = min(x_one, x_two)
# x2 = max(x_one, x_two)

# y_one = np.random.randint(1, 128)
# y_two = np.random.randint(1, 128)
# y1 = min(y_one, y_two)
# y2 = max(y_one, y_two)


# HARD CODED MASKS
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
# mask[int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = 1


# CIRCLE MASK
# mask = np.zeros((128, 128))
# center_x = 64
# center_y = 64
# r = 20
# for i in range(128):
#     for j in range(128):
#         if (i - center_x)**2 + (j - center_y)**2 < r**2:
#             mask[i][j] = 1          


# CREATE RANDOM MASK
shape_mask = (128, 128) # Size of the masks. Use powers of 2.
num_masks = 1
# num_masks = opt.niter # Num of different masks
persistance = .4 # Controls the smoothness of the stains' boundaries. Should be float > 0. In practice, < 1
threshold = .8 # More or less controls the area of the stains 

# Mask generation
output_size = (num_masks, shape_mask[0], shape_mask[1])
generator = torch.Generator()
generator.manual_seed(0)

octaves = 5 # Controls level of detail. Should be integer 1-9, depending on the mask shape
resolutions = [(2 ** i, 2 ** i) for i in range(1, octaves + 1)]
factors = [persistance ** i for i in range(octaves)]
fp = pyperlin.FractalPerlin2D(output_size, resolutions, factors, generator=generator)
noise = fp().numpy()

generation = noise[0]
generation = generation - np.min(generation)
generation = generation/np.max(generation)
generation_t = (generation > .8).astype(np.uint8)
mask = generation_t

# Making Directories for Results
try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

# Setting Seed 
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# Enables cuda benchmarking which finds optimal algorithms for faster runtime
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Finds data and preprocesses it 
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
elif opt.dataset == 'streetview':
    transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
assert dataset

# Loads Data
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0

netG = _netG(opt)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
print(netG)


netD = _netlocalD(opt)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
print(netD)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

# What is a Float Tensor?
input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

# CHANGED
# real_center = torch.FloatTensor(opt.batchSize, 3, x2-x1, y2-y1)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
    # real_center = real_center.cuda()


input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)


# real_center = Variable(real_center)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(resume_epoch,opt.niter):
    shape_mask = (128, 128) # Size of the masks. Use powers of 2.
    num_masks = opt.niter
    # num_masks = opt.niter # Num of different masks
    persistance = .4 # Controls the smoothness of the stains' boundaries. Should be float > 0. In practice, < 1
    threshold = .8 # More or less controls the area of the stains 

    # Mask generation
    output_size = (num_masks, shape_mask[0], shape_mask[1])
    generator = torch.Generator()
    generator.manual_seed(0)

    octaves = 5 # Controls level of detail. Should be integer 1-9, depending on the mask shape
    resolutions = [(2 ** i, 2 ** i) for i in range(1, octaves + 1)]
    factors = [persistance ** i for i in range(octaves)]
    fp = pyperlin.FractalPerlin2D(output_size, resolutions, factors, generator=generator)
    noise = fp().numpy()
        
    for i, data in enumerate(dataloader, 0):
        generation = noise[i]
        generation = generation - np.min(generation)
        generation = generation/np.max(generation)
        generation_t = (generation > .8).astype(np.uint8)
        mask = generation_t
            
        real_cpu, _ = data
        # real_center_cpu = real_cpu[:,:,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)]
        # real_center_cpu = real_cpu[:,:,x1:x2,y1:y2]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)

        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data[:,:, mask==1] = 1.0
        # real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

        # RECTANGLES 
        # input_cropped.data[:,0,x1:x2,y1:y2] = 1.0
        # input_cropped.data[:,1,x1:x2,y1:y2] = 1.0
        # input_cropped.data[:,2,x1:x2,y1:y2] = 1.0


    # input_cropped.data[:,0,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = 1.0
    # input_cropped.data[:,1,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = 1.0
    # input_cropped.data[:,2,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = 1.0

        # train with real
    netD.zero_grad()
        # print("Discriminator")
        # summary(netD, (3, 128, 128))
       
        # labeling every image in the batch 
    label.data.resize_(batch_size).fill_(real_label)

    output = netD(input_real)
    errD_real = criterion(output.squeeze(1), label)
    errD_real.backward()
    D_x = output.data.mean()

    # train with fake
    # noise.data.resize_(batch_size, nz, 1, 1)
    # noise.data.normal_(0, 1)
    fake = netG(input_cropped, mask)
    label.data.fill_(fake_label)
    output = netD(fake.detach())
    errD_fake = criterion(output.squeeze(1), label)
    errD_fake.backward()
    D_G_z1 = output.data.mean()
    errD = errD_real + errD_fake
    optimizerD.step()


    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.data.fill_(real_label)  # fake labels are real for generator cost
    output = netD(fake)
    errG_D = criterion(output.squeeze(1), label)
    # errG_D.backward(retain_variables=True)

    # errG_l2 = criterionMSE(fake,real_center)
    
    # CHANGED
    # Increases the weight of pixels around the border
    # wtl2Matrix = input_real.clone()
    # wtl2Matrix.data.fill_(wtl2*overlapL2Weight)
    # wtl2Matrix.data[:,:,int(opt.overlapPred):int((x2-x1)- opt.overlapPred),int(opt.overlapPred):int((y2-y1) - opt.overlapPred)] = wtl2

    errG_l2 = (fake-input_real).pow(2)
    errG_l2 = errG_l2 * torch.Tensor(mask).to("cuda")
    errG_l2 = errG_l2.mean()

    errG = (1-wtl2) * errG_D + wtl2 * errG_l2

    errG.backward()

    # Not used 
    # D_G_z2 = output.data.mean()
    optimizerG.step()

    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
          % (epoch, opt.niter, i, len(dataloader),
             errD.data, errG_D.data,errG_l2.data, D_x,D_G_z1, ))
    if i % 100 == 0:
        vutils.save_image(real_cpu,
                'result/train/real/real_samples_epoch_%03d.png' % (epoch))
        vutils.save_image(input_cropped.data,
                'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
        recon_image = input_cropped.clone()
        print(recon_image.data.shape)
        vutils.save_image(fake.data,
                'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))


    # do checkpointing
    torch.save({'epoch':epoch+1,
                'state_dict':netG.state_dict()},
                'model/netG_streetview.pth' )
    torch.save({'epoch':epoch+1,
                'state_dict':netD.state_dict()},
                'model/netlocalD.pth' )
