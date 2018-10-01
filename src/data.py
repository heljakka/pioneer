import os
import random
import h5py
import numpy as np
import scipy

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader


import config

args = config.get_config()

class CelebAHQ():
    def __init__(self, path):
        resolution = ['data2x2', 'data4x4', 'data8x8', 'data16x16', 'data32x32', 'data64x64', \
                        'data128x128', 'data256x256', 'data512x512', 'data1024x1024']
        self._base_key = 'data'
        print("Try H5 data path {}".format(path))
        self.dataset = h5py.File(path, 'r')
        self._len = {k:len(self.dataset[k]) for k in resolution}
        assert all([resol in self.dataset.keys() for resol in resolution])

    def __call__(self, batch_size, phase, alpha):
        size = 4*(2**phase)
        key = self._base_key + '{}x{}'.format(size, size)
        idx = np.random.randint(self._len[key], size=batch_size)
        hi_res_batch_x = np.array([self.dataset[key][i]/127.5-1.0 for i in idx], dtype=np.float32)
        ret_batch = []
        if alpha < 1.0 and phase > 0:
            lr_key = self._base_key + '{}x{}'.format(size//2, size//2)
            low_res_batch_x = np.array([self.dataset[lr_key][i]/127.5-1.0 for i in idx], dtype=np.float32).repeat(2, axis=2).repeat(2, axis=3)
            ret_batch = hi_res_batch_x * alpha + low_res_batch_x * (1.0 - alpha)
        else:
            ret_batch = hi_res_batch_x

        if args.sample_mirroring:
            for i in range(len(ret_batch)):
                if random.randint(0,1) == 0:
                    ret_batch[i] = np.flip(ret_batch[i], axis=2)

        return ret_batch

def get_loader(datasetName, path):
    if datasetName == 'celeba':
        loader = Utils.celeba_loader(path)
    elif datasetName == 'lsun':
        loader = Utils.lsun_loader(path)
    elif datasetName == 'cifar10':
        loader = Utils.cifar10_loader(path)
    elif datasetName == 'celebaHQ':
        loader = CelebAHQ(path) # Expects the path that has the .h5 file

    return loader

class Utils:
    @staticmethod
    def lsun_loader(path):
        def loader(transform, batch_size):
            data = datasets.LSUNClass(
                path, transform=transform,
                target_transform=lambda x: 0)
            data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                    num_workers=4, pin_memory=(args.gpu_count>1))

            return data_loader

        return loader

    @staticmethod
    def celeba_loader(path):
        def loader(transform, batch_size):
            data = datasets.ImageFolder(path, transform=transform)
            data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                    num_workers=4, drop_last=True, pin_memory=(args.gpu_count>1))

            return data_loader

        return loader

    @staticmethod
    def cifar10_loader(path):
        def loader(transform, batch_size):
            data = datasets.CIFAR10(root=path, download=True,
                           transform=transform)
            data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                    num_workers=2, pin_memory=(args.gpu_count>1))
            return data_loader

        return loader

    maybeRandomHorizontalFlip = transforms.RandomHorizontalFlip() if args.sample_mirroring else transforms.Lambda(lambda x: x)

    @staticmethod
    def sample_data(dataloader, batch_size, image_size=4):
        if (args.data == 'celebaHQ'):
            while True: #This is an infinite iterator
                batch = dataloader(batch_size, int(np.log2(image_size / 4)), 0.0)
                yield torch.from_numpy(batch), None #no label
                #yield torch.from_numpy(img), None #no label
            return # will never be reached

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.data == 'celeba': #center crop first to 128 (i.e. leave out the edge parts), then resize to this LOD size
            transform_with_resize = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.Resize(image_size),
                Utils.maybeRandomHorizontalFlip,
                #transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif args.data == 'lsun': # resize to the desired size, then center crop to this LOD size
            transform_with_resize = transforms.Compose([
                transforms.Resize(image_size),
                Utils.maybeRandomHorizontalFlip,
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif args.data == 'cifar10': # No center crop, no horizontal flipping
            transform_with_resize = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            assert(False)

        if not args.resize_training_data:
            print("WARNING! MAKE SURE YOU RUN ON PRE-RESIZED DATA! DATA WILL NOT BE RESIZED.")

        loader = dataloader(transform_with_resize if args.resize_training_data else transform, batch_size=batch_size)

        for img, label in loader:
            yield img, label

        #TODO Triple-check that image_size can be given this way instead of always calculated from session.step
    @staticmethod
    def sample_data2(dataloader, batch_size, image_size, session):
        if (args.data == 'celebaHQ'):
            while True: #This is an infinite iterator           
                batch = dataloader(batch_size, session.phase, session.alpha)
                yield torch.from_numpy(batch), None #no label
            return # will never be reached

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if args.data == 'celeba':
            transform_with_resize_norm = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.Resize(image_size),
                Utils.maybeRandomHorizontalFlip,
                #transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            transform_with_resize = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.Resize(image_size),
                Utils.maybeRandomHorizontalFlip,
                transforms.ToTensor(),
            ])
        elif args.data == 'lsun':
            transform_with_resize_norm = transforms.Compose([
                transforms.Resize(image_size),
                Utils.maybeRandomHorizontalFlip,
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            transform_with_resize = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                Utils.maybeRandomHorizontalFlip,
                transforms.ToTensor(),
            ])
        elif args.data == 'cifar10': # No center crop, no horizontal flipping
            transform_with_resize_norm = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            transform_with_resize = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        else:
            assert(False)

        # Note that random flip is not re-applied when downscaling
        transform_with_resize_previous = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(image_size/2)),
            transforms.ToTensor(),
        ])

        transform_with_normalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))        
        ])
        
        fade = session.phase > 0 and session.alpha < 1.0

        if not args.resize_training_data:
            print("WARNING! MAKE SURE YOU RUN ON PRE-RESIZED DATA! DATA WILL NOT BE RESIZED.")

        if fade:
            loader = dataloader(transform_with_resize if args.resize_training_data else transform, batch_size=batch_size)
        else:
            loader = dataloader(transform_with_resize_norm if args.resize_training_data else transform, batch_size=batch_size)
        
        for img, label in loader:           
            if not fade:
                yield img, label
            else:
                low_resol_batch_x = np.array([transform_with_resize_previous(img[i]).numpy() for i in range(img.size(0))], dtype=np.float32).repeat(2, axis=2).repeat(2, axis=3)

                # For testing:
                #alpha_delta = torch.from_numpy(np.linspace(0, 1, img.size(0),dtype=np.float32))
                #alpha_delta = alpha_delta.view((32,1,1,1))

                mixed_img = img * session.alpha + ((1.0-session.alpha)*torch.from_numpy(low_resol_batch_x))
                mixed_img = np.array([transform_with_normalize(mixed_img[i]).numpy() for i in range(img.size(0))], dtype=np.float32)

                # For testing:
                # yield img, torch.from_numpy(low_resol_batch_x), torch.from_numpy(mixed_img), label

                yield torch.from_numpy(mixed_img), label

def dump_training_set(loader, dump_trainingset_N, dump_trainingset_dir, session):
    batch_size = 8
    total = 0

    phase = min(args.max_phase, session.phase)

    reso = 4 * (2 ** phase)

    if not os.path.exists(dump_trainingset_dir):
        os.makedirs(dump_trainingset_dir)    

    print("Dumping training data with {}x{} and alpha {}".format(reso, reso, session.alpha))
    dataset = Utils.sample_data2(loader, batch_size, reso, session)

    for i in range(int(dump_trainingset_N / batch_size) + 1):
        curr_batch_size = min(batch_size, dump_trainingset_N - total)
        batch, _ = next(dataset)        
        for j in range(curr_batch_size):
            save_path = '{}/orig_{}.png'.format(dump_trainingset_dir, total)
            total += 1
            grid = utils.save_image(batch[j] / 2 + 0.5, save_path, padding=0)
        if total % 500 < batch_size:
            print(total)
