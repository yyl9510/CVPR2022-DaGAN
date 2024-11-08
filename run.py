import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

# from frames_dataset import FramesDataset
import pdb
# from modules.generator import OcclusionAwareGenerator
from torch.utils.tensorboard import SummaryWriter 
import modules.generator as gen_module
from modules.discriminator import MultiScaleDiscriminator
# from modules.keypoint_detector import KPDetector
import modules.keypoint_detector as KPD
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
from train import train
# from reconstruction import reconstruction
from animate import animate
import json
import hashlib

from data.dataset import VDataset
from nnscaler.parallel import ComputeConfig, ReuseType, build_optimizer, parallelize
from data.autodist import autodist, pas_policy, PASData


def init_seeds(cuda_deterministic=True):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn
    seed = 0 + dist.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        cudnn.deterministic = True
        cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

def main(rank, world_size):
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    # parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
    #                     help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--use_depth",action='store_true',help='depth mode')
    parser.add_argument("--rgbd",action='store_true',help='rgbd mode')
    parser.add_argument("--kp_prior",action='store_true',help='use kp_prior in final objective function')

    # alter model
    parser.add_argument("--generator",required=True,help='the type of genertor')
    parser.add_argument("--kp_detector",default='KPDetector',type=str,help='the type of KPDetector')
    parser.add_argument("--GFM",default='GeneratorFullModel',help='the type of GeneratorFullModel')
    
    parser.add_argument("--batchsize",type=int, default=-1,help='user defined batchsize')
    parser.add_argument("--kp_num",type=int, default=-1,help='user defined keypoint number')
    parser.add_argument("--kp_distance",type=int, default=10,help='the weight of kp_distance loss')
    parser.add_argument("--depth_constraint",type=int, default=0,help='the weight of depth_constraint loss')

    parser.add_argument("--name",type=str,help='user defined model saved name')

    parser.set_defaults(verbose=False)
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += opt.name


    print("Training...")

    device=torch.device("cuda",rank)
    torch.cuda.set_device(device)
    config['train_params']['loss_weights']['depth_constraint'] = opt.depth_constraint
    config['train_params']['loss_weights']['kp_distance'] = opt.kp_distance
    if opt.kp_prior:
        config['train_params']['loss_weights']['kp_distance'] = 0
        config['train_params']['loss_weights']['kp_prior'] = 10
    if opt.batchsize != -1:
        config['train_params']['batch_size'] = opt.batchsize
    if opt.kp_num != -1:
        config['model_params']['common_params']['num_kp'] = opt.kp_num

    # create generator
    generator = getattr(gen_module, opt.generator)(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)
    if opt.verbose:
        print(generator)

    # create discriminator
    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    discriminator.to(device)
    if opt.verbose:
        print(discriminator)

    # # create kp_detector
    # if opt.use_depth:
    #     config['model_params']['common_params']['num_channels'] = 1
    # if opt.rgbd:
    #     config['model_params']['common_params']['num_channels'] = 4
        
    kp_detector = getattr(KPD, opt.kp_detector)(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])
    kp_detector.to(device)
    if opt.verbose:
        print(kp_detector)

    if config['backend_params']['backend_name'] == "cube":
        # backend configs
        plan_ngpus = config["backend_params"]["plan_ngpus"]
        runtime_ngpus = config["backend_params"]["runtime_ngpus"]
        batchsize = config["train_params"]["batch_size"]  # per gpu batch size
        batchsize *= plan_ngpus
        frame_shape = config["dataset_params"]["frame_shape"]
        # create dummy input tensors for cube graph tracing
        kp_detector_dummy_input = {"x": torch.randn(batchsize, 3, frame_shape[0], frame_shape[1])}

        generator_dummy_input = {
            'source_image': torch.randn(batchsize, 3, 256, 256),
            'kp_driving': {
                'value': torch.randn(batchsize, 15, 2),
                'jacobian': torch.randn(batchsize, 15, 2, 2)
            },
            'kp_source': {
                'value': torch.randn(batchsize, 15, 2),
                'jacobian': torch.randn(batchsize, 15, 2, 2)
            },
            'source_depth': torch.randn(batchsize, 1, 256, 256),
            'driving_depth': None   # torch.randn(batchsize, 1, 256, 256)
        }

        pyramide_generated = {
            'prediction_1': torch.randn(batchsize, 3, 256, 256),
            'prediction_0.5': torch.randn(batchsize, 3, 128, 128),
            'prediction_0.25': torch.randn(batchsize, 3, 64, 64),
            'prediction_0.125': torch.randn(batchsize, 3, 32, 32)
        }

        detached_kp_driving = {
            'value': torch.randn(batchsize, 15, 2),
            'jacobian': torch.randn(batchsize, 15, 2, 2)
        }

        compute_config = ComputeConfig(
            plan_ngpus, runtime_ngpus, use_zero=False, user_config={"batch_size": batchsize}
        )
        # parallelize models
        from nnscaler.graph.function.wrapnn import convert_to_wrapnn

        kp_detector = convert_to_wrapnn(kp_detector)
        kp_detector = parallelize(
            kp_detector,
            kp_detector_dummy_input,
            PASData, # partial(autodist, mem_cons=0.92)
            compute_config,
            reuse=ReuseType.MOO,
        )

        generator = convert_to_wrapnn(generator)
        generator = parallelize(
            generator,
            generator_dummy_input,
            PASData, # partial(autodist, mem_cons=0.92)
            compute_config,
            reuse=ReuseType.MOO,
        )

        discriminator = convert_to_wrapnn(discriminator)
        discriminator = parallelize(
            discriminator,
            {"x": pyramide_generated, "kp": detached_kp_driving},
            PASData, # partial(autodist, mem_cons=0.92)
            compute_config,
            reuse=ReuseType.MOO,
        )

    else:
        generator= torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
        generator = DDP(generator,device_ids=[rank],broadcast_buffers=False)

        discriminator= torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = DDP(discriminator,device_ids=[rank],broadcast_buffers=False)

        kp_detector= torch.nn.SyncBatchNorm.convert_sync_batchnorm(kp_detector)
        kp_detector = DDP(kp_detector,device_ids=[rank],broadcast_buffers=False)

    generator.to(device)
    discriminator.to(device)
    kp_detector.to(device)

    dataset = VDataset(is_train=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    combined_str = json.dumps(config, sort_keys=True) + json.dumps(vars(opt), sort_keys=True)
    hashstr = hashlib.sha256(combined_str.encode('utf-8')).hexdigest()
    if rank == 0:
        writer = SummaryWriter(os.path.join(log_dir, 'tensorboard-logs', hashstr))
    else:
        writer = None

    if opt.mode == 'train':
        train(config, generator, discriminator, kp_detector, opt.checkpoint, log_dir, dataset, rank, device, opt, writer)


if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    init_seeds()
    main(rank, world_size)
    dist.destroy_process_group()

    