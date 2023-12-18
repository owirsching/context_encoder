import torch
from PIL import Image
from torch.autograd import Variable
import torchvision
import pyperlin
import numpy as np

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().add(1).div(2).mul(255).clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= Variable(mean)
    # batch /= Variable(std)
    batch = torch.div(batch,Variable(std))
    return batch


def generate_masks(num_masks, shape_mask):
    # num_masks = opt.niter # Num of different masks
    persistance = .4 # Controls the smoothness of the stains' boundaries. Should be float > 0. In practice, < 1
    threshold = .8 # More or less controls the area of the stains 

    # Mask generation
    output_size = (num_masks, shape_mask[0], shape_mask[1])
    gen = torch.Generator()

    octaves = 5 # Controls level of detail. Should be integer 1-9, depending on the mask shape
    resolutions = [(2 ** i, 2 ** i) for i in range(1, octaves + 1)]
    factors = [persistance ** i for i in range(octaves)]
    fp = pyperlin.FractalPerlin2D(output_size, resolutions, factors, generator=gen)
    noise = fp().numpy()

    masks = []
    for i, data in enumerate(noise): 
        generation = data
        generation = generation - np.min(generation)
        generation = generation/np.max(generation)
        generation_t = (generation > .8).astype(np.uint8)
        mask = generation_t
        masks.append(mask)
    
    return masks


# STYLE 
# The code below could be used to extract style layers from an images

# def extract_features(X, style_layers):
#     contents = []
#     styles = []
#     for i in range(len(net)):
#         X = net[i](X)
#         if i in style_layers:
#             styles.append(X)
#     return styles
