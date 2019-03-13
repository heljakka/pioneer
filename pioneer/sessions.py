import os

import torch
from torch import nn, optim
import copy

from pioneer import config
from pioneer import utils
from pioneer.model import Generator, Discriminator, SpectralNormConv2d

args   = config.get_config()

def loadSNU(path):
    if os.path.exists(path):
        map_location='cuda:0' if (not args.disable_cuda and torch.cuda.is_available()) else 'cpu'
        us_list = torch.load(path, map_location=map_location)
        #import ipdb; ipdb.set_trace()
        for layer_i, layer in enumerate(SpectralNormConv2d.spectral_norm_layers):
            setattr(layer, 'weight_u', us_list[layer_i])
    else:
        print("Warning! No SNU found.")

def saveSNU(path):
    us = []
    for layer in SpectralNormConv2d.spectral_norm_layers:
        us += [getattr(layer, 'weight_u')]
    torch.save(us, path)

class Session:
    def __init__(self):
        # Note: 4 requirements for sampling from pre-existing models:
        # 1) Ensure you save and load both multi-gpu versions (DataParallel) or both not.
        # 2) Ensure you set the same phase value as the pre-existing model and that your local and global alpha=1.0 are set
        # 3) Sample from the g_running, not from the latest generator
        # 4) You may need to warm up the g_running by running evaluate.reconstruction_dryrun() first

        self.alpha = -1
        self.sample_i = min(args.start_iteration, 0)
        self.phase = args.start_phase

        self.generator = nn.DataParallel( Generator(args.nz+1, args.n_label).to(device=args.device) )
        self.g_running = nn.DataParallel( Generator(args.nz+1, args.n_label).to(device=args.device) )
        self.encoder   = nn.DataParallel( Discriminator(nz = args.nz+1, n_label = args.n_label, binary_predictor = args.train_mode == config.MODE_GAN).to(device=args.device) )

        print("Using ", torch.cuda.device_count(), " GPUs!")

        self.reset_opt()

        print('Session created.')

    def reset_opt(self):
        self.optimizerG = optim.Adam(self.generator.parameters(), args.lr, betas=(0.0, 0.99))
        self.optimizerD = optim.Adam(self.encoder.parameters(), args.lr, betas=(0.0, 0.99)) # includes all the encoder parameters...

    def save_all(self, path):
        torch.save({'G_state_dict': self.generator.state_dict(),
                    'D_state_dict': self.encoder.state_dict(),
                    'G_running_state_dict': self.g_running.state_dict(),
                    'optimizerD': self.optimizerD.state_dict(),
                    'optimizerG': self.optimizerG.state_dict(),
                    'iteration': self.sample_i,
                    'phase': self.phase,
                    'alpha': self.alpha},
                path)

    def load(self, path):
        map_location='cuda:0' if (not args.disable_cuda and torch.cuda.is_available()) else 'cpu'

        checkpoint = torch.load(path, map_location=map_location)
        self.sample_i = int(checkpoint['iteration'])
        
        self.generator.load_state_dict(checkpoint['G_state_dict'])
        self.g_running.load_state_dict(checkpoint['G_running_state_dict'])
        self.encoder.load_state_dict(checkpoint['D_state_dict'])

        if args.reset_optimizers <= 0:
            self.optimizerD.load_state_dict(checkpoint['optimizerD'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG'])
            print("Reloaded old optimizers")
        else:
            print("Despite loading the state, we reset the optimizers.")

        self.alpha = checkpoint['alpha']
        self.phase = int(checkpoint['phase'])
        if args.start_phase > 0: #If the start phase has been manually set, try to actually use it (e.g. when have trained 64x64 for extra rounds and then turning the model over to 128x128)
            self.phase = min(args.start_phase, self.phase)
            print("Use start phase: {}".format(self.phase))
        if self.phase > args.max_phase:
            print('Warning! Loaded model claimed phase {} but max_phase={}'.format(self.phase, args.max_phase))
            self.phase = args.max_phase

    def create(self):
        if args.start_iteration <= 0:
            args.start_iteration = 1
            if args.no_progression:
                self.sample_i = args.start_iteration = int( (args.max_phase + 0.5) * args.images_per_stage ) # Start after the fade-in stage of the last iteration
                args.force_alpha = 1.0
                print("Progressive growth disabled. Setting start step = {} and alpha = {}".format(args.start_iteration, args.force_alpha))
        else:
            reload_from = '{}/checkpoint/{}_state'.format(args.save_dir, str(args.start_iteration).zfill(6)) #e.g. '604000' #'600000' #latest'   
            reload_from_SNU = '{}/checkpoint/{}_SNU'.format(args.save_dir, str(args.start_iteration).zfill(6))
            print(reload_from)
            if os.path.exists(reload_from):
                self.load(reload_from)
                print("Loaded {}".format(reload_from))
                print("Iteration asked {} and got {}".format(args.start_iteration, self.sample_i))               
                if args.load_SNU:
                    loadSNU(reload_from_SNU)

                if args.testonly:
                   self.generator = copy.deepcopy(self.g_running)
            else:
                assert(not args.testonly)
                self.sample_i = args.start_iteration
                print('Start from iteration {}'.format(self.sample_i))

        self.g_running.train(False)

        if args.force_alpha >= 0.0:
            self.alpha = args.force_alpha

        utils.accumulate(self.g_running, self.generator, 0)
