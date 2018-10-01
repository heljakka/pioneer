import torch
from torch.autograd import Variable

import config
import os
import random

args = config.get_config()

def switch_grad_updates_to_first_of(a,b):
    requires_grad(a, True)
    requires_grad(b, False)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def split_labels_out_of_latent(z):
    label = torch.unsqueeze(z[:, -args.n_label], dim=1)
    return z[:, :args.nz], label          

def make_dirs():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.checkpoint_dir = args.save_dir + '/checkpoint'
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if args.use_TB:
        if not os.path.exists(args.summary_dir):
            os.makedirs(args.summary_dir)

def populate_z(z, nz, noise, batch_size):
    '''
    Fills noise variable `z` with noise U(S^M) [from https://github.com/DmitryUlyanov/AGE ]
    '''
    z.data.resize_(batch_size, nz) #, 1, 1)
    z.data.normal_(0, 1)
    if noise == 'sphere':
        normalize_(z.data)

def normalize_(x, dim=1):
    '''
    Projects points to a sphere inplace.
    '''
    zn = x.norm(2, dim=dim)
    zn = zn.unsqueeze(1)    
    x = x.div_(zn)
    x.expand_as(x)

def normalize(x, dim=1):
    '''
    Projects points to a sphere.
    '''
    zn = x.norm(2, dim=dim)
    zn = zn.unsqueeze(1)
    return x.div(zn).expand_as(x)        

def mismatch(x, y, dist):
    '''
    Computes distance between corresponding points points in `x` and `y`
    using distance `dist`.
    '''
    if dist == 'L2':
        return (x - y).pow(2).mean(dim=1).mean()
    elif dist == 'L1':
        return (x - y).abs().mean(dim=1).mean()
    elif dist == 'cos':
        x_n = normalize(x)
        y_n = normalize(y)

        ret = 2 - (x_n).mul(y_n)
        return ret.mean(dim=1).mean()
    else:
        assert dist == 'none', '?'

def var(x, dim=0):
    '''
    Calculates variance. [from https://github.com/DmitryUlyanov/AGE ]
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
