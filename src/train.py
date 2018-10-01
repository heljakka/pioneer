import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torchvision import utils

from model import Generator, Discriminator

from datetime import datetime
import random
import copy

import os

import config
import utils
import data
import evaluate

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from torch.nn import functional as F

args   = config.get_config()
writer = None

def batch_size(reso):
    if args.gpu_count == 1:
        save_memory = False
        if not save_memory:
            batch_table = {4:128, 8:128, 16:128, 32:64, 64:32, 128:16, 256:8, 512:4, 1024:1}
        else:
            batch_table = {4:128, 8:128, 16:128, 32:32, 64:16, 128:4, 256:2, 512:2, 1024:1}
    elif args.gpu_count == 2:
        batch_table = {4:256, 8:256, 16:256, 32:128, 64:64, 128:32, 256:16, 512:8, 1024:2}
    elif args.gpu_count == 4:
        batch_table = {4:512, 8:256, 16:128, 32:64, 64:32, 128:32, 256:32, 512:16, 1024:4}
    elif args.gpu_count == 8:
        batch_table = {4:512, 8:512, 16:512, 32:256, 64:256, 128:128, 256:64, 512:32, 1024:8}
    else:
        assert(False)
    
    return batch_table[reso]

def batch_size_by_phase(phase):
    return batch_size(4 * 2 ** phase)

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

        self.generator = nn.DataParallel( Generator(args.nz+1, args.n_label).cuda() )
        self.g_running = nn.DataParallel( Generator(args.nz+1, args.n_label).cuda() )
        self.encoder   = nn.DataParallel( Discriminator(nz = args.nz+1, n_label = args.n_label, binary_predictor = args.train_mode == config.MODE_GAN).cuda() )

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
        checkpoint = torch.load(path)
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
            print(reload_from)
            if os.path.exists(reload_from):
                self.load(reload_from)
                print("Loaded {}".format(reload_from))
                print("Iteration asked {} and got {}".format(args.start_iteration, self.sample_i))               

                if args.testonly:
                    self.generator = copy.deepcopy(self.g_running)
            else:
                assert(not args.testonly)
                self.sample_i = args.start_iteration
                print('Start from iteration {}'.format(self.sample_i))

        self.g_running.train(False)

        if args.force_alpha >= 0.0:
            self.alpha = args.force_alpha

        accumulate(self.g_running, self.generator, 0)

def setup():
    utils.make_dirs()
    if not args.testonly:
        config.log_args(args)

    if args.use_TB:
        from dateutil import tz
        from tensorboardX import SummaryWriter

        dt = datetime.now(tz.gettz('Europe/Helsinki')).strftime(r"%y%m%d_%H%M")
        global writer
        writer = SummaryWriter("{}/{}/{}".format(args.summary_dir, args.save_dir, dt))

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)   

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def get_grad_penalty(discriminator, real_image, fake_image, step, alpha):
    """ Used in WGAN-GP version only. """
    eps = torch.rand(batch_size_by_phase(step), 1, 1, 1).cuda()

    if eps.size(0) != real_image.size(0) or eps.size(0) != fake_image.size(0):
        # If end-of-batch situation, we restrict other vectors to matcht the number of training images available.
        eps = eps[:real_image.size(0)]
        fake_image = fake_image[:real_image.size(0)]

    x_hat = eps * real_image.data + (1 - eps) * fake_image.data
    x_hat = Variable(x_hat, requires_grad=True)

    if args.train_mode == config.MODE_GAN: # Regular GAN mode
        hat_predict, _ = discriminator(x_hat, step, alpha, args.use_ALQ)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
    else:
        hat_z = discriminator(x_hat, step, alpha, args.use_ALQ)
        # KL_fake: \Delta( e(g(Z)) , Z ) -> max_e
        KL_maximizer = KLN01Loss(direction=args.KL, minimize=False)
        KL_fake = KL_maximizer(hat_z) * args.fake_D_KL_scale
        grad_x_hat = grad(
            outputs=KL_fake.sum(), inputs=x_hat, create_graph=True)[0]

    # Push the gradients of the interpolated samples towards 1
    grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                        .norm(2, dim=1) - 1)**2).mean()
    grad_penalty = 10 * grad_penalty
    return grad_penalty

def D_prediction_of_G_output(generator, encoder, step, alpha):
    # To use labels, enable here and elsewhere:
    #label = Variable(torch.ones(batch_size_by_phase(step), args.n_label)).cuda()
    #               label = Variable(
    #                    torch.multinomial(
    #                        torch.ones(args.n_label), args.batch_size, replacement=True)).cuda()

    myz = Variable(torch.randn(batch_size_by_phase(step), args.nz)).cuda(async=(args.gpu_count>1))
    myz = utils.normalize(myz)
    myz, label = utils.split_labels_out_of_latent(myz)

    fake_image = generator(myz, label, step, alpha)
    fake_predict, _ = encoder(fake_image, step, alpha, args.use_ALQ)

    loss = fake_predict.mean()
    return loss, fake_image

class KLN01Loss(torch.nn.Module): #Adapted from https://github.com/DmitryUlyanov/AGE

    def __init__(self, direction, minimize):
        super(KLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'

        self.direction = direction

    def forward(self, samples):

        assert samples.nelement() == samples.size(1) * samples.size(0), '?'

        samples = samples.view(samples.size(0), -1)

        self.samples_var = utils.var(samples)
        self.samples_mean = samples.mean(0)

        samples_mean = self.samples_mean
        samples_var = self.samples_var

        if self.direction == 'pq':
            t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
            t2 = samples_var.log()

            KL = (t1 + t2 - 0.5).mean()
        else:
            # In the AGE implementation, there is samples_var^2 instead of samples_var^1
            t1 = (samples_var + samples_mean.pow(2)) / 2
            # In the AGE implementation, this did not have the 0.5 scaling factor:
            t2 = -0.5*samples_var.log()

            KL = (t1 + t2 - 0.5).mean()

        if not self.minimize:
            KL *= -1

        return KL

def train(generator, encoder, g_running, train_data_loader, test_data_loader, session, total_steps, train_mode):
    pbar = tqdm(initial=session.sample_i, total = total_steps)
    
    benchmarking = False
  
    match_x = args.match_x
    generatedImagePool = None

    refresh_dataset   = True
    refresh_imagePool = True

    # After the Loading stage, we cycle through successive Fade-in and Stabilization stages

    batch_count = 0

    reset_optimizers_on_phase_start = False

    # TODO Unhack this (only affects the episode count statistics anyway):
    if args.data != 'celebaHQ':
        epoch_len = len(train_data_loader(1,4).dataset)
    else:
        epoch_len = train_data_loader._len['data4x4']

    if args.step_offset != 0:
        if args.step_offset == -1:
            args.step_offset = session.sample_i
        print("Step offset is {}".format(args.step_offset))
        session.phase += args.phase_offset
        session.alpha = 0.0

    while session.sample_i < total_steps:
        #######################  Phase Maintenance ####################### 

        steps_in_previous_phases = max(session.phase * args.images_per_stage, args.step_offset)

        sample_i_current_stage = session.sample_i - steps_in_previous_phases

        # If we can move to the next phase
        if sample_i_current_stage >= args.images_per_stage:
            if session.phase < args.max_phase: # If any phases left
                iteration_levels = int(sample_i_current_stage / args.images_per_stage)
                session.phase += iteration_levels
                sample_i_current_stage -= iteration_levels * args.images_per_stage
                match_x = args.match_x # Reset to non-matching phase
                print("iteration B alpha={} phase {} will be reduced to 1 and [max]".format(sample_i_current_stage, session.phase))

                refresh_dataset = True
                refresh_imagePool = True # Reset the pool to avoid images of 2 different resolutions in the pool

                if reset_optimizers_on_phase_start:
                    utils.requires_grad(generator)
                    utils.requires_grad(encoder)
                    generator.zero_grad()
                    encoder.zero_grad()
                    session.reset_opt()
                    print("Optimizers have been reset.")                

        reso = 4 * 2 ** session.phase

        # If we can switch from fade-training to stable-training
        if sample_i_current_stage >= args.images_per_stage/2:
            if session.alpha < 1.0:
                refresh_dataset = True # refresh dataset generator since no longer have to fade
            match_x = args.match_x * args.matching_phase_x
        else:
            match_x = args.match_x

        session.alpha = min(1, sample_i_current_stage * 2.0 / args.images_per_stage) # For 100k, it was 0.00002 = 2.0 / args.images_per_stage

        if refresh_dataset:
            train_dataset = data.Utils.sample_data2(train_data_loader, batch_size(reso), reso, session)
            refresh_dataset = False
            print("Refreshed dataset. Alpha={} and iteration={}".format(session.alpha, sample_i_current_stage))
        if refresh_imagePool:
            imagePoolSize = 200 if reso < 256 else 100
            generatedImagePool = utils.ImagePool(imagePoolSize) #Reset the pool to avoid images of 2 different resolutions in the pool
            refresh_imagePool = False
            print('Image pool created with size {} because reso is {}'.format(imagePoolSize, reso))

        ####################### Training init ####################### 

        z = Variable( torch.FloatTensor(batch_size(reso), args.nz, 1, 1) ).cuda(async=(args.gpu_count>1))
        KL_minimizer = KLN01Loss(direction=args.KL, minimize=True)
        KL_maximizer = KLN01Loss(direction=args.KL, minimize=False)
        
        stats = {}

        one = torch.FloatTensor([1]).cuda(async=(args.gpu_count>1))

        try:
            real_image, _ = next(train_dataset)              
        except (OSError, StopIteration):
            train_dataset = data.Utils.sample_data2(train_data_loader, batch_size(reso), reso, session)
            real_image, _ = next(train_dataset)

        ####################### DISCRIMINATOR / ENCODER ###########################

        utils.switch_grad_updates_to_first_of(encoder, generator)
        encoder.zero_grad()

        x = Variable(real_image).cuda(async=(args.gpu_count>1))
        kls = ""
        if train_mode == config.MODE_GAN:
            
            # Discriminator for real samples
            real_predict, _ = encoder(x, session.phase, session.alpha, args.use_ALQ)
            real_predict = real_predict.mean() \
                - 0.001 * (real_predict ** 2).mean()
            real_predict.backward(-one) # Towards 1

            # (1) Generator => D. Identical to (2) see below
            
            fake_predict, fake_image = D_prediction_of_G_output(generator, encoder, session.phase, session.alpha)
            fake_predict.backward(one)

            # Grad penalty

            grad_penalty = get_grad_penalty(encoder, x, fake_image, session.phase, session.alpha)
            grad_penalty.backward()

        elif train_mode == config.MODE_CYCLIC:
            e_losses = []

            # e(X)
            
            real_z = encoder(x, session.phase, session.alpha, args.use_ALQ)
            if args.use_real_x_KL:
                # KL_real: - \Delta( e(X) , Z ) -> max_e
                KL_real = KL_minimizer(real_z) * args.real_x_KL_scale
                e_losses.append(KL_real)

                stats['real_mean'] = KL_minimizer.samples_mean.data.mean()
                stats['real_var'] = KL_minimizer.samples_var.data.mean()
                stats['KL_real'] = KL_real.data[0]
                kls = "{0:.3f}".format(stats['KL_real'])

            # The final entries are the label. Normal case, just 1. Extract it/them, and make it [b x 1]:

            real_z, label = utils.split_labels_out_of_latent(real_z)
            recon_x = generator(real_z, label, session.phase, session.alpha)
            if args.use_loss_x_reco:
                # match_x: E_x||g(e(x)) - x|| -> min_e
                err = utils.mismatch(recon_x, x, args.match_x_metric) * match_x
                e_losses.append(err)
                stats['x_reconstruction_error'] = err.data[0]

            args.use_wpgan_grad_penalty = False
            grad_penalty = 0.0
            
            if args.use_loss_fake_D_KL:
                # TODO: The following codeblock is essentially the same as the KL_minimizer part on G side. Unify
                utils.populate_z(z, args.nz+args.n_label, args.noise, batch_size(reso))
                z, label = utils.split_labels_out_of_latent(z)
                fake = generator(z, label, session.phase, session.alpha).detach()

                if session.alpha >= 1.0:
                    fake = generatedImagePool.query(fake.data)

                # e(g(Z))
                egz = encoder(fake, session.phase, session.alpha, args.use_ALQ)

                # KL_fake: \Delta( e(g(Z)) , Z ) -> max_e
                KL_fake = KL_maximizer(egz) * args.fake_D_KL_scale
                e_losses.append(KL_fake)

                stats['fake_mean'] = KL_maximizer.samples_mean.data.mean()
                stats['fake_var'] = KL_maximizer.samples_var.data.mean()
                stats['KL_fake'] = -KL_fake.data[0]
                kls = "{0}/{1:.3f}".format(kls, stats['KL_fake'])

                if args.use_wpgan_grad_penalty:
                    grad_penalty = get_grad_penalty(encoder, x, fake, session.phase, session.alpha)
            
            # Update e
            if len(e_losses) > 0:
                e_loss = sum(e_losses)                
                stats['E_loss'] = np.float32(e_loss.data)
                e_loss.backward()

                if args.use_wpgan_grad_penalty:
                    grad_penalty.backward()
                    stats['Grad_penalty'] = grad_penalty.data

                #book-keeping
                disc_loss_val = e_loss.data[0]

        session.optimizerD.step()

        torch.cuda.empty_cache()

        ######################## GENERATOR / DECODER #############################
        
        if (batch_count + 1) % args.n_critic == 0:
            utils.switch_grad_updates_to_first_of(generator, encoder)

            for _ in range(args.n_generator):
                generator.zero_grad()
                g_losses = []
                
                if train_mode == config.MODE_GAN:
                    fake_predict, _ = D_prediction_of_G_output(generator, encoder, session.phase, session.alpha)
                    loss = -fake_predict
                    g_losses.append(loss)
                    
                elif train_mode == config.MODE_CYCLIC: #TODO We push the z variable around here like idiots
                    def KL_of_encoded_G_output(generator, z):
                        utils.populate_z(z, args.nz+args.n_label, args.noise, batch_size(reso))
                        z, label = utils.split_labels_out_of_latent(z)                        
                        fake = generator(z, label, session.phase, session.alpha)
                        
                        egz = encoder(fake, session.phase, session.alpha, args.use_ALQ)
                        # KL_fake: \Delta( e(g(Z)) , Z ) -> min_g
                        return egz, label, KL_minimizer(egz) * args.fake_G_KL_scale, z

                    egz, label, kl, z = KL_of_encoded_G_output(generator, z)

                    if args.use_loss_KL_z:
                        g_losses.append(kl) # G minimizes this KL
                        stats['KL(Phi(G))'] = kl.data[0]
                        kls = "{0}/{1:.3f}".format(kls, stats['KL(Phi(G))'])

                    if args.use_loss_z_reco:
                        z = torch.cat((z, label), 1)
                        z_diff = utils.mismatch(egz, z, args.match_z_metric) * args.match_z # G tries to make the original z and encoded z match
                        g_losses.append(z_diff)                

                if len(g_losses) > 0:
                    loss = sum(g_losses)
                    stats['G_loss'] = np.float32(loss.data)
                    loss.backward()

                    # Book-keeping only:
                    gen_loss_val = loss.data[0]
                
                session.optimizerG.step()

                torch.cuda.empty_cache()

                if train_mode == config.MODE_CYCLIC:
                    if args.use_loss_z_reco:
                        stats['z_reconstruction_error'] = z_diff.data[0]

            accumulate(g_running, generator)

            del z, x, one, real_image, real_z, KL_real, label, recon_x, fake, egz, KL_fake, kl, z_diff

            if train_mode == config.MODE_CYCLIC:
                if args.use_TB:
                    for key,val in stats.items():                    
                        writer.add_scalar(key, val, session.sample_i)
                elif batch_count % 100 == 0:
                    print(stats)

            if args.use_TB:
                writer.add_scalar('LOD', session.phase + session.alpha, session.sample_i)

        ########################  Statistics ######################## 

        b = batch_size_by_phase(session.phase)
        zr, xr = (stats['z_reconstruction_error'], stats['x_reconstruction_error']) if train_mode == config.MODE_CYCLIC else (0.0, 0.0)
        e = (session.sample_i / float(epoch_len))
        pbar.set_description(
            ('{0}; it: {1}; phase: {2}; b: {3:.1f}; Alpha: {4:.3f}; Reso: {5}; E: {6:.2f}; KL(real/fake/fakeG): {7}; z-reco: {8:.2f}; x-reco {9:.3f}; real_var {10:.4f}').format(batch_count+1, session.sample_i+1, session.phase, b, session.alpha, reso, e, kls, zr, xr, stats['real_var'])
            )
            #(f'{i + 1}; it: {iteration+1}; b: {b:.1f}; G: {gen_loss_val:.5f}; D: {disc_loss_val:.5f};'
            # f' Grad: {grad_loss_val:.5f}; Alpha: {alpha:.3f}; Reso: {reso}; S-mean: {real_mean:.3f}; KL(real/fake/fakeG): {kls}; z-reco: {zr:.2f}'))

        pbar.update(batch_size(reso))
        session.sample_i += batch_size(reso) # if not benchmarking else 100
        batch_count += 1

        ########################  Saving ######################## 

        if batch_count % args.checkpoint_cycle == 0:
            for postfix in {'latest', str(session.sample_i).zfill(6)}:
                session.save_all('{}/{}_state'.format(args.checkpoint_dir, postfix))

            print("Checkpointed to {}".format(session.sample_i))

        ########################  Tests ######################## 

        try:
            evaluate.tests_run(g_running, encoder, test_data_loader, session, writer,
            reconstruction       = (batch_count % 800 == 0),
            interpolation        = (batch_count % 800 == 0),
            collated_sampling    = (batch_count % 800 == 0),
            individual_sampling  = (batch_count % (args.images_per_stage/batch_size(reso)/4) == 0)
            )
        except (OSError, StopIteration):
            print("Skipped periodic tests due to an exception.")

    pbar.close()

def main():
    setup()
    session = Session()
    session.create()

    print('PyTorch {}'.format(torch.__version__))

    if args.train_path:
        train_data_loader = data.get_loader(args.data, args.train_path)
    else:
        train_data_loader = None
    
    if args.test_path:
        test_data_loader = data.get_loader(args.data, args.test_path)
    elif args.aux_inpath:
        test_data_loader = data.get_loader(args.data, args.aux_inpath)
    else:
        test_data_loader = None

    # 4 modes: Train (with data/train), test (with data/test), aux-test (with custom aux_inpath), dump-training-set
    
    if args.run_mode == config.RUN_TRAIN:
        train(session.generator, session.encoder, session.g_running, train_data_loader, test_data_loader,
            session  = session,
            total_steps = args.total_kimg * 1000,
            train_mode = args.train_mode)
    elif args.run_mode == config.RUN_TEST:
        if args.reconstructions_N > 0 or args.interpolate_N > 0:
            evaluate.Utils.reconstruction_dryrun(session.generator, session.encoder, test_data_loader, session=session)
        evaluate.tests_run(session.generator, session.encoder, test_data_loader, session=session, writer=writer)
    elif args.run_mode == config.RUN_DUMP:
        session.phase = args.start_phase
        data.dump_training_set(train_data_loader, args.dump_trainingset_N, args.dump_trainingset_dir, session)

if __name__ == '__main__':
    main()
