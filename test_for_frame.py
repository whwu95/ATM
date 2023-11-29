import argparse
import datetime
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision

import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from dataset.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma

from datasets import build_dataset
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import  multiple_samples_collate, AverageMeter, accuracy, reduce_tensor
import utils

import clip
import yaml
from dotmap import DotMap


class VideoCLIP(nn.Module):
    def __init__(self, clip_arch, num_classes = 174, embed_dim = 512, tsm = 'tokent1d', T= 8, dropout=0.0, emb_dropout=0.0, 
                       pretrain=False, joint = False) :
        super(VideoCLIP, self).__init__()
            # get fp16 model and weight

        if clip_arch in ["EVA02-CLIP-L-14", "EVA02-CLIP-L-14-336"]:
            from eva_clip import create_model_and_transforms
            
            weight_path={
                "EVA02-CLIP-L-14": "./eva_clip/pretrain/EVA02_CLIP_L_psz14_s4B.pt",
                "EVA02-CLIP-L-14-336":"./eva_clip/pretrain/EVA02_CLIP_L_336_psz14_s6B.pt",
                }
            
            clip_model, _, preprocess=create_model_and_transforms(clip_arch, pretrained=weight_path[clip_arch], force_custom_clip=True,
                    tsm=tsm,
                    T=T,
                    dropout= 0.0,
                    emb_dropout= 0.0,
                    )
            clip_state_dict = clip_model.state_dict()
            
        else:
            clip_model, clip_state_dict = clip.load(
                clip_arch,
                device='cpu',jit=False,
                tsm=tsm,
                T=T,
                dropout= 0.0, #dropout,
                emb_dropout= 0.0, #emb_dropout,
                pretrain= pretrain,
                joint = joint) # Must set jit=False for training  ViT-B/32
        
        self.visual = clip_model.visual
        self.n_seg = T
        self.drop_out = nn.Dropout(p=0.0) #dropout
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, image):
        # image [B,C,T,H,W]
        b,t,c,h,w = image.size()
        #import pdb;pdb.set_trace()
        #image = image.transpose(1,2).reshape(-1,c,h,w)
        image = image.reshape(-1,c,h,w)
        image_emb = self.visual(image).view(b, self.n_seg, -1)
        image_emb = image_emb.mean(dim=1, keepdim=False)
        image_emb = self.drop_out(image_emb)
        logit = self.fc(image_emb)
        return logit
    
    @staticmethod
    def no_weight_decay():
        return {'pos_embed', 'temporal_embed'}

def get_args():
    parser = argparse.ArgumentParser('ATM evaluation and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=4)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default=' ', type=str, help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'SSV2', 'SSV1', 'UCF101', 'HMDB51','image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    parser.add_argument('--embed_dim', default=512, type=int, help='projector dim after clip_visual encoder')
    #---------------------------- for multi-crop multi-clip test --------------------------------
    parser.add_argument('--dense_sample', default=False, type=bool) 
    
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    #sampler_test = torch.utils.data.DistributedSampler(
    #        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    sampler_test = torch.utils.data.DistributedSampler(dataset_test)
    
    #sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    #if global_rank == 0 and args.log_dir is not None:
    #    os.makedirs(args.log_dir, exist_ok=True)
    #    log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    #else:
    #    log_writer = None

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=0, #args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    model = VideoCLIP(
        clip_arch = args.model, 
        num_classes = args.nb_classes, embed_dim = args.embed_dim,
        tsm = 'tokent1d', T=args.num_frames,
        dropout=args.drop_path, 
        emb_dropout=args.drop,
        pretrain= False, #args.finetune,
        joint = False,
    )
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    #print(model.state_dict().keys())
    
    checkpoint = torch.load(args.resume, map_location='cpu')
    if dist.get_rank() == 0:
        print('load model: epoch {}'.format(checkpoint['epoch']))
    msg = model_without_ddp.load_state_dict(update_dict(checkpoint['module']))
    del checkpoint
    print(msg)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
   
    prec1 = validate(
        data_loader_test, device, 
        model, args.num_frames, args.test_num_crop, args.test_num_segment)
    

    
def validate(val_loader, device, model, frames, test_num_crop, test_num_segment):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    proc_start_time = time.time()

    with torch.no_grad():
        for i, (image, class_id) in enumerate(val_loader): #[B,3*16*12,224,224]
            

            #import pdb;pdb.set_trace()
            batch_size = class_id.numel()
            
            num_crop = test_num_crop
            num_crop *= test_num_segment  # 4 clips for testing when using dense sample
            
            class_id = class_id.to(device)
            n_seg = frames
            image = image.view((-1, n_seg, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            
            image_input = image.to(device) #.view(-1, c, h, w)
            logits = model(image_input)  # bt n_class

            cnt_time = time.time() - proc_start_time

            logits = logits.view(batch_size, -1, logits.size(1)).softmax(dim=-1)
            logits = logits.mean(dim=1, keepdim=False)      # bs n_class

            prec = accuracy(logits, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))
    
            if i % 10 == 0 and dist.get_rank() == 0:
                runtime = float(cnt_time) / (i + 1) / (batch_size * dist.get_world_size())
                print(
                    ('Test: [{0}/{1}], average {runtime:.4f} sec/video \t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), runtime=runtime, top1=top1, top5=top5)))

    if dist.get_rank() == 0:
        print('-----Evaluation is finished------')
        print('Overall Prec@1 {:.03f}% Prec@5 {:.03f}%'.format(top1.avg, top5.avg))
    
    return top1.avg


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
