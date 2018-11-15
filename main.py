import os
import sys
import json
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from arguments import get_args
from utils import Logger

from model import generate_model
from dataset import get_training_set, get_validation_set, get_test_set
from train import train_epoch
from validation import val_epoch
import test

from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose


if __name__ == '__main__':
    
    # get arguments:
    args = get_args()
    
    # datetime:
    now = datetime.now()
    args.datetime = '{:%Y%m%d_%H%M}'.format(now)
    
    # setting up paths
    if args.root_path != '':
        args.video_path = os.path.join(args.root_path, args.video_path)
        args.annotation_path = os.path.join(args.root_path, args.annotation_path)
        args.result_path = os.path.join(args.root_path, args.result_path)
        if args.resume_path:
            args.resume_path = os.path.join(args.root_path, args.resume_path)
        if args.pretrain_path:
            args.pretrain_path = os.path.join(args.root_path, args.pretrain_path)
    
    # save args to file
    with open(os.path.join(args.result_path, 'args.json'), 'w') as args_file:
        json.dump(vars(args), args_file)
    
    # scale for multi-scale cropping
    args.scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
        
    # model arch
    args.arch = '{}-{}'.format(args.model, args.model_depth)
    
    # mean and std (used to train the model: activitynet or kinetics)
    args.mean = get_mean(args.norm_value, dataset=args.mean_dataset)
    args.std = get_std(args.norm_value)
    
    if args.no_mean_norm and not args.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not args.std_norm:
        norm_method = Normalize(args.mean, [1, 1, 1])
    else:
        norm_method = Normalize(args.mean, args.std)
    
    # create model
    args.device_ids = [0,1]  # gpu ids (manually set)
    torch.manual_seed(args.manual_seed)
    model, parameters = generate_model(args)
    # print(model)
    
    # loss function
    criterion = nn.CrossEntropyLoss()
    if not args.no_cuda:
        criterion = criterion.cuda()

    
    ###################################################################
    # Training prep
    ###################################################################
    if not args.no_train:
        
        # Data augmentation
        assert args.train_crop in ['random', 'corner', 'center']
        if args.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(args.scales, args.sample_size)
        elif args.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(args.scales, args.sample_size)
        elif args.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                args.scales, args.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(args.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(args.sample_duration)
        target_transform = ClassLabel()
        
        # Datraining data loader
        training_data = get_training_set(args, spatial_transform,
                                         temporal_transform, target_transform)
        #FIXME: num_worker fails
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True)
        
        # logger
        train_logger = Logger(
            os.path.join(args.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(args.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if args.nesterov:
            dampening = 0
        else:
            dampening = args.dampening
        optimizer = optim.SGD(
            parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            dampening=dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.lr_patience)
    
    
    ###################################################################
    # Validation prep
    ###################################################################
    if not args.no_val:
        
        # Data augmentation
        spatial_transform = Compose([
            Scale(args.sample_size),
            CenterCrop(args.sample_size),
            ToTensor(args.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(args.sample_duration)
        target_transform = ClassLabel()
        
        # dataloader
        validation_data = get_validation_set(
            args, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True)
        
        # logger
        val_logger = Logger(
            os.path.join(args.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    
    ###################################################################
    # Loading from checkpoint
    ###################################################################
    if args.resume_path:
        print('loading checkpoint {}'.format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        assert args.arch == checkpoint['arch']

        args.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        if not args.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    
    ###################################################################
    # Run training/testing
    ###################################################################
    print('run')
    for i in range(args.begin_epoch, args.n_epochs + 1):
        if not args.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, args,
                        train_logger, train_batch_logger)
        if not args.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, args,
                                        val_logger)

        if not args.no_train and not args.no_val:
            scheduler.step(validation_loss)

    if args.test:
        
        # Data augmentation
        spatial_transform = Compose([
            Scale(int(args.sample_size / args.scale_in_test)),
            CornerCrop(args.sample_size, args.crop_position_in_test),
            ToTensor(args.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(args.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(args, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True)
        test.test(test_loader, model, args, test_data.class_names)
