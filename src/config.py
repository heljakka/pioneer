import argparse
import torch

MODE_GAN    = 0
MODE_CYCLIC = 1

RUN_TRAIN = 0
RUN_TEST  = 1
RUN_DUMP  = 2

def parse_args():
    parser = argparse.ArgumentParser(description='PIONEER')
    parser.add_argument('--train_path', type=str, help='training dataset root directory or H5 file')
    parser.add_argument('--test_path', type=str, default=None, help='testing dataset root directory or H5 file')
    parser.add_argument('--aux_inpath', type=str, default=None, help='Input path of specified dataset to reconstruct or interpolate for')
    parser.add_argument('--aux_outpath', type=str, default=None, help='Output path of specified dataset to reconstruct or interpolate for')
    parser.add_argument('--testonly', action='store_true', help='Run in test mode. Quit after the tests.')
    
    parser.add_argument('--phase_offset', type=int, default=0, help='Use the reloaded model but start fresh from next phase (when manually moving to the next )')
    parser.add_argument('--step_offset', type=int, default=0, help='Use the reloaded model but ignore the given number of steps (use -1 for all steps)')

    parser.add_argument('-d', '--data', default='celeba', type=str,
                        choices=['celeba', 'lsun', 'cifar10', 'celebaHQ'],
                        help=('Specify dataset. '
                            'Currently celeba, lsun and cifar10 are supported'))
    parser.add_argument('--KL', default='qp', help='The KL divergence direction [pq|qp]')
    parser.add_argument('--match_x_metric', default='L1', help='none|L1|L2|cos')
    parser.add_argument('--match_z_metric', default='cos', help='none|L1|L2|cos')
    parser.add_argument('--noise', default='sphere', help='normal|sphere')
    parser.add_argument('--summary_dir', default='log/pine/runs', help='Tensorflow summaries directory')
    parser.add_argument('--save_dir', default='tests',
                    help='folder to output images')
    #parser.add_argument('--batch_size', type=int,
    #                default=16, help='batch size')
    parser.add_argument('--no_TB', action='store_true', help='Do not create Tensorboard logs')
    parser.add_argument('--start_iteration', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use_ALQ', type=int, default=0, help='Reserved for future use')
    parser.add_argument('--images_per_stage', type=int, default=-1)
    
    parser.add_argument('--matching_phase_x', type=float, default=1.5)
    parser.add_argument('--force_alpha', type=float, default=-1.0)
    parser.add_argument('--start_phase', type=int, default=0)    
    parser.add_argument('--max_phase', type=int, default=-1, help='The highest progressive growth phase that we train for, e.g. phase 4 means 64x64. If not given, we use dataset-based defaults.')
    parser.add_argument('--reset_optimizers', type=int, default=0, help='Even if reloading from a checkpoint, reset the optimizer states.')
    parser.add_argument('--checkpoint_cycle', type=int, default=10000)
    parser.add_argument('--total_kimg', type=int, default=-1, help='The total number of samples to train for. 1 kimg = 1000 samples. Default values should be fine.')
    parser.add_argument('--manual_seed', type=int, default=123)
    parser.add_argument('--no_progression', action='store_true', help='No progressive growth. Set the network to the target size from the beginning. Note that the step count starts from non-zero to make the results comparable.')


    # Special modes:
    parser.add_argument('--dump_trainingset_N', type=int, default=0)
    parser.add_argument('--dump_trainingset_dir', type=str, default='.')
    parser.add_argument('--interpolate_N', type=int, default=0, help='Carry out the given number of interpolation runs (between 4 input images)')
    parser.add_argument('--reconstructions_N', type=int, default=0, help='The number of reconstructions to run')
    parser.add_argument('--sample_N', type=int, default=128, help='The number of random samples to run')

    return parser.parse_args()

args = None

def log_args(args):
    f = open('{}/config.txt'.format(args.save_dir), 'w')
    for key,val in vars(args).items():
        f.write("{}={}\n".format(key,val))
    f.close()

def init():
    global args

    args.run_mode = RUN_TRAIN
    if args.dump_trainingset_N > 0:
        args.run_mode = RUN_DUMP
    elif args.testonly:
        args.run_mode = RUN_TEST

    assert(args.run_mode != RUN_DUMP  or args.dump_trainingset_dir)
    assert(args.run_mode != RUN_TRAIN or (args.train_path and args.test_path))
    # Test path or aux test path is needed if we run tests other than just random-sampling
    assert(args.run_mode != RUN_TEST  or args.test_path or args.aux_inpath or (args.interpolate_N <=0 and args.reconstructions_N <=0 and args.sample_N > 0))

    assert(args.step_offset != 0 or args.phase_offset == 0)

    args.n_label = 1
    args.nz = 512 - args.n_label
    args.n_critic = 1
    args.n_generator = 2
    args.nc = 3

    args.use_loss_x_reco    = True
    args.use_real_x_KL      = True

    args.use_loss_fake_D_KL = True

    args.use_loss_z_reco    = True
    args.use_loss_KL_z      = True

    args.match_x = 1
    args.match_z = 100
    args.fake_D_KL_scale = 0.1
    args.fake_G_KL_scale = args.fake_D_KL_scale
    args.real_x_KL_scale = 0.1

    args.use_TB = not args.no_TB

    args.sample_mirroring = True
    if args.testonly:
        args.sample_mirroring = False
        print("In test mode, sample mirroring is disabled automatically.")

    args.resize_training_data = True # If you already have the right size, skip resize ops here, e.g. using /data/celeba_3k/train/resize_128

    args.train_mode = MODE_CYCLIC

    # TODO: Separate the celebaHQ-specific configuration: images_per_stage + img buffer size

    if args.images_per_stage == -1:
        args.images_per_stage = 2400e3 if args.data != 'celebaHQ' else 4800e3

    if args.max_phase == -1:
        if args.data == 'celebaHQ':
            args.max_phase = 6
        elif args.data != 'cifar10':
            args.max_phase = 5
        else:
            args.max_phase = 3

    # Due to changing batch sizes in different stages, we control the length of training by the total number of samples
    if args.total_kimg < 0:
        args.total_kimg = int(args.images_per_stage * (args.max_phase+2)/1000) #All stages once, the last stage trained twice.

    args.h5 = (args.data == 'celebaHQ')

    args.gpu_count = torch.cuda.device_count() # Set to 1 manually if don't want multi-GPU support

    print(args)

    print("Total training samples {}k. Max phase for dataset {} is {}. Once the maximum phase is trained the full round, we continue training that phase.".format(args.total_kimg, args.data, args.max_phase))
   

def get_config():
    global args
    if args == None:
        args = parse_args()
        init()
    return args
