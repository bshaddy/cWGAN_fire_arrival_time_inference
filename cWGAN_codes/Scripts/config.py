# Conditional GAN Deraining Script
# Created by: Deep Ray, University of Southern California
# Date: 27 December 2021

import argparse, textwrap

formatter = lambda prog: argparse.HelpFormatter(prog,max_help_position=50)

def cla():
    parser = argparse.ArgumentParser(description='list of arguments',formatter_class=formatter)

    # Data parameters
    parser.add_argument('--train_file', type=str, default='', help=textwrap.dedent('''Data file containing training data pairs'''))
    parser.add_argument('--n_train'   , type=int, default=4000, help=textwrap.dedent('''Number of training samples to use. Cannot be more than that available.'''))
    
    
    # Network parameters
    parser.add_argument('--g_type'    , type=str, default='resskip', choices=['resskip','denseskip'], help=textwrap.dedent('''Type of generator to use'''))
    parser.add_argument('--d_type'    , type=str, default='res', choices=['res','dense'], help=textwrap.dedent('''Type of critic to use'''))
    parser.add_argument('--g_out'     , type=str, default='x', choices=['x','dx','bdx'], help=textwrap.dedent('''x: U-Net learns de-rained image with sigmoid output function, dx: U-Net learns fluctuations about rainy-image with identity output function, bdx: dx with sigmoid output function'''))
    parser.add_argument('--gp_coef'   , type=float, default=10.0, help=textwrap.dedent('''Gradient penalty parameter'''))
    parser.add_argument('--n_critic'  , type=int, default=4, help=textwrap.dedent('''Number of critic updates per generator update'''))
    parser.add_argument('--n_epoch'   , type=int, default=1000, help=textwrap.dedent('''Maximum number of epochs'''))
    parser.add_argument('--z_dim'     , type=int, default=None, help=textwrap.dedent('''Dimension of the latent variable. If not prescribed, then a Pix2Pix type algorithm is trained.'''))
    parser.add_argument('--batch_size', type=int, default=16, help=textwrap.dedent('''Batch size while training'''))
    parser.add_argument('--reg_param' , type=float, default=1e-7, help=textwrap.dedent('''Regularization parameter'''))
    parser.add_argument('--act_param' , type=float, default=0.1, help=textwrap.dedent('''Activation parameter'''))
    parser.add_argument('--seed_no'   , type=int, default=1008, help=textwrap.dedent('''Set the random seed'''))
    parser.add_argument('--mismatch_param', type=float, default=0.0, help=textwrap.dedent('''Parameter to penalize deviation of ensemble samples from ground-truth'''))

    # Output parameters
    parser.add_argument('--save_freq'    , type=int, default=100, help=textwrap.dedent('''Number of epochs after which a snapshot and plots are saved'''))
    parser.add_argument('--sdir_suffix'  , type=str, default='', help=textwrap.dedent('''Suffix to directory where trained network/results are saved'''))
    parser.add_argument('--z_n_MC'       , type=int, default=10, help=textwrap.dedent('''Number of (z) samples used to generate emperical statistics. Used only if z_dim > 0'''))
    
    # Testing parameters
    parser.add_argument('--GANdir'       , type=str, default=None, help=textwrap.dedent('''Load checkpoint from user specified GAN directory. Else path will be infered from hyperparameters.'''))
    parser.add_argument('--test_file'    , type=str, default='', help=textwrap.dedent('''Paired data file for testing'''))
    parser.add_argument('--n_test'       , type=int, default=None, help=textwrap.dedent('''Number of test pair-samples to use. Cannot be more than that available.'''))
    parser.add_argument('--test_image'   , type=str, default='', help=textwrap.dedent('''Rainy image for testing'''))
    parser.add_argument('--psize'        , type=int, default=256, help=textwrap.dedent('''Patch size to used when loading test images'''))
    parser.add_argument('--results_dir'  , type=str, default='Test_results', help=textwrap.dedent('''Directory where test results are saved'''))
    parser.add_argument('--ckpt_id'      , type=int, default=-1, help=textwrap.dedent('''The checkpoint index to load when testing or restarting training'''))

    assert parser.parse_args().z_dim == None or parser.parse_args().z_dim > 0

    return parser.parse_args()


