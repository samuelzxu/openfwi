# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import os
import sys
import time
import datetime
import json
import numpy as np

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from new_dataset import FineFWIDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose
import wandb

import utils
import network
from scheduler import WarmupMultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import transforms as T

step = 0

def train_one_epoch(model, criterion, optimizer, lr_scheduler, 
                    dataloader, device, epoch, print_freq, writer, grad_accum_steps):
    global step
    model.train()

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print(f"Gradient Accum. Steps: {grad_accum_steps}")

    optimizer.zero_grad()  # Zero gradients at the beginning
    accum_samples = 0
    
    for i, (data, label) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        start_time = time.time()
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss, loss_g1v, loss_g2v = criterion(output, label)
        
        # Scale the loss by the number of accumulation steps
        loss = loss / grad_accum_steps
        loss.backward()
        
        # Keep track of loss values for logging
        loss_val = loss.item() * grad_accum_steps  # Rescale for logging
        loss_g1v_val = loss_g1v.item()
        loss_g2v_val = loss_g2v.item()
        batch_size = data.shape[0]
        accum_samples += batch_size

        # Log metrics
        metric_logger.update(loss=loss_val, loss_g1v=loss_g1v_val, 
            loss_g2v=loss_g2v_val, lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(accum_samples / (time.time() - start_time))
        
        if writer:
            writer.add_scalar('loss', loss_val, step)
            writer.add_scalar('loss_g1v', loss_g1v_val, step)
            writer.add_scalar('loss_g2v', loss_g2v_val, step)
        
        # Log metrics to wandb
        if not args.distributed or (args.rank == 0 and args.local_rank == 0):
            wandb.log({
                'train/loss': loss_val,
                'train/loss_g1v': loss_g1v_val,
                'train/loss_g2v': loss_g2v_val,
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/samples_per_sec': accum_samples / (time.time() - start_time)
            }, step=step)
        
        # Only update weights after accumulating gradients for specified number of steps
        # or at the end of the epoch
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate after optimizer step
            lr_scheduler.step()
            
            step += 1
            accum_samples = 0  # Reset accumulated samples counter


def evaluate(model, criterion, dataloader, device, writer):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'

    data_types = ['Style_A','Style_B','CurveFault_A','CurveFault_B','FlatFault_A','FlatFault_B','CurveVel_A','CurveVel_B','FlatVel_A','FlatVel_B']
    g1v_unnorm_by_dt = {
        dt: [] for dt in data_types
    }
    # Add meter for un-normalized loss
    all_paths = []
    metric_logger.add_meter('unnorm_loss_g1v', utils.SmoothedValue(window_size=20, fmt='{value:.8f}'))
    with torch.no_grad():
        for data, label, paths in metric_logger.log_every(dataloader, 20, header):
            all_paths.extend(paths)
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(data)
            loss, loss_g1v, loss_g2v = criterion(output, label)
            
            # Compute un-normalized L1 loss
            # Get normalization parameters from dataset configuration
            with open('dataset_config.json') as f:
                ctx = json.load(f)[args.dataset]
            
            # Denormalize predictions and ground truth
            label_min, label_max = ctx['label_min'], ctx['label_max']

            # Shape [batch_size, 70, 70]
            output_unnorm = output * (label_max - label_min) / 2.0 + (label_max + label_min) / 2.0
            label_unnorm = label * (label_max - label_min) / 2.0 + (label_max + label_min) / 2.0
            
            # Compute un-normalized L1 loss
            unnorm_loss_g1v = nn.L1Loss()(output_unnorm, label_unnorm).item()
            unnorm_loss_g1v_expanded = nn.L1Loss(reduction='none')(output_unnorm, label_unnorm).mean(dim=(2,3))
            print(unnorm_loss_g1v_expanded.shape)

            for dt in data_types:
                indices = np.where([dt in p for p in paths])
                g1v_unnorm_by_dt[dt].append(unnorm_loss_g1v_expanded[indices])
            metric_logger.update(loss=loss.item(), 
                loss_g1v=loss_g1v.item(), 
                loss_g2v=loss_g2v.item(),
                unnorm_loss_g1v=unnorm_loss_g1v)
    
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
    print(' * Un-normalized G1V Loss {unnorm_loss_g1v.global_avg:.8f}\n'.format(unnorm_loss_g1v=metric_logger.unnorm_loss_g1v))
    
    if writer:
        writer.add_scalar('loss', metric_logger.loss.global_avg, step)
        writer.add_scalar('loss_g1v', metric_logger.loss_g1v.global_avg, step)
        writer.add_scalar('loss_g2v', metric_logger.loss_g2v.global_avg, step)
        writer.add_scalar('unnorm_loss_g1v', metric_logger.unnorm_loss_g1v.global_avg, step)
    
    # Log validation metrics to wandb
    if not args.distributed or (args.rank == 0 and args.local_rank == 0):
        wandb.log({
            'val/loss': metric_logger.loss.global_avg,
            'val/loss_g1v': metric_logger.loss_g1v.global_avg,
            'val/loss_g2v': metric_logger.loss_g2v.global_avg,
            'val/unnorm_loss_g1v': metric_logger.unnorm_loss_g1v.global_avg,
        }, step=step)
        
    return metric_logger.loss.global_avg


def main(args):
    global step

    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    utils.mkdir(args.output_path) # create folder to store checkpoints
    utils.init_distributed_mode(args) # distributed mode initialization

    # Initialize wandb only on the main process in distributed training
    if not args.distributed or (args.rank == 0 and args.local_rank == 0):
        wandb_mode = "disabled" if args.no_wandb else "online"
        wandb.init(
            project=args.wandb_project,
            name=f"{args.save_name}_{args.suffix}" if args.suffix else args.save_name,
            config=vars(args),
            mode=wandb_mode
        )

    # Set up tensorboard summary writer
    train_writer, val_writer = None, None
    if args.tensorboard:
        utils.mkdir(args.log_path) # create folder to store tensorboard logs
        if not args.distributed or (args.rank == 0) and (args.local_rank == 0):
            train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
            val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))
                                                                 

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

    # Create dataset and dataloader
    print('Loading data')
    print('Loading training data')
        
    # Normalize data and label to [-1, 1]
    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=args.k), T.log_transform(ctx['data_max'], k=args.k))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])

    dataset_train = FineFWIDataset(
        args.train_anno,
        sample_ratio=args.sample_temporal,
        transform_data=transform_data,
        transform_label=transform_label,
        expand_label_zero_dim=False,
        expand_data_zero_dim=False,
        squeeze=True
    )


    print('Loading validation data')
    dataset_valid = FineFWIDataset(
        args.val_anno,
        sample_ratio=args.sample_temporal,
        transform_data=transform_data,
        transform_label=transform_label,
        expand_label_zero_dim=False,
        expand_data_zero_dim=False,
        squeeze=True
    )

    dataset_valid.set_return_path(True)

    print('Creating data loaders')
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_valid)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        drop_last=True, collate_fn=default_collate)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        collate_fn=default_collate)

    print('Creating model')
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
    model = network.model_dict[args.model](upsample_mode=args.up_mode, 
        sample_spatial=args.sample_spatial, sample_temporal=args.sample_temporal).to(device)
    
    # Log model architecture with wandb
    if not args.distributed or (args.rank == 0 and args.local_rank == 0):
        wandb.watch(model, log="all", log_freq=args.print_freq)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Define loss function
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v

    # Scale lr according to effective batch size
    lr = args.lr * args.world_size
    # Account for gradient accumulation
    effective_batch_size = args.batch_size * args.grad_accum_steps
    if args.scale_lr:
        lr = lr * (effective_batch_size / args.batch_size)
        print(f"Scaling learning rate to {lr} for effective batch size of {effective_batch_size}")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Convert scheduler to be per iteration instead of per epoch
    # warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
    # Adjust for gradient accumulation - each optimizer step occurs after grad_accum_steps batches
    total_update_steps = (args.epochs * len(dataloader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
    # lr_scheduler = WarmupMultiStepLR(
    #     optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
    #     warmup_iters=warmup_iters, warmup_factor=1e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_update_steps, eta_min=args.lr/30)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(network.replace_legacy(checkpoint['model']))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        lr_scheduler.milestones=lr_milestones

    print('Start training')
    start_time = time.time()
    best_loss = 10
    chp=1 
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        loss = evaluate(model, criterion, dataloader_valid, device, val_writer)

        train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader_train,
                        device, epoch, args.print_freq, train_writer, args.grad_accum_steps)
        
        #loss = evaluate(model, criterion, dataloader_valid, device, val_writer)
        
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'args': args}
        # Save checkpoint per epoch
        if loss < best_loss:
            utils.save_on_master(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
            print('saving checkpoint at epoch: ', epoch)
            
            # Log best model with wandb
            if not args.distributed or (args.rank == 0 and args.local_rank == 0):
                wandb.log({"best_val_loss": loss}, step=step)
                if args.save_model_wandb:
                    checkpoint_path = os.path.join(args.output_path, 'checkpoint.pth')
                    wandb.save(checkpoint_path, base_path=args.output_path)
                
            chp = epoch
            best_loss = loss
        # Save checkpoint every epoch block
        print('current best loss: ', best_loss)
        print('current best epoch: ', chp)
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            model_path = os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1))
            utils.save_on_master(checkpoint, model_path)
            
            # Save model to wandb
            if not args.distributed or (args.rank == 0 and args.local_rank == 0):
                if args.save_model_wandb:
                    wandb.save(model_path, base_path=args.output_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # Finish wandb run
    if not args.distributed or (args.rank == 0 and args.local_rank == 0):
        wandb.finish()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-t', '--train-anno', default='train_ds.csv', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='val_ds.csv', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
    parser.add_argument('-l', '--log-path', default='Invnet_models', help='path to parent folder to save logs')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, help='inverse model name')
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    
    # Training related
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--grad-accum-steps', default=1, type=int, help='number of gradient accumulation steps')
    parser.add_argument('--scale-lr', action='store_true', help='scale learning rate by gradient accumulation steps')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4 , type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')   
    parser.add_argument('-eb', '--epoch_block', type=int, default=40, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=3, help='number of saved block')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')

    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
    
    # Distributed training related
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Tensorboard related
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')
    
    # Wandb related
    parser.add_argument('--wandb-project', default='OpenFWI', help='wandb project name')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--save-model-wandb', action='store_true', help='Save model checkpoints to wandb')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    
    args.epochs = args.epoch_block * args.num_block

    if args.resume:
        args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
