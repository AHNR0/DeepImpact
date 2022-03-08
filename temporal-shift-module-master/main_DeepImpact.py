# Based on the code "TSM: Temporal Shift Module for Efficient Video Understanding"


import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, accuracy_sens_prec, AverageMeter_confusion
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.modality)
    print(args.train_list)
    
    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    
    # args.store_name += '_layers3_4_dataset015_FullData_5gametest'
    # args.store_name += '_fullmodel_dataset015_FullData'
    args.store_name += '_fullmodel_dataset015_kalman_FullData_5gametest'
    # args.store_name += '_fullmodel_dataset015_Fold4'
    print('storing name: ' + args.store_name)

    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)
    
    
    # print(model)
    # params = model.state_dict()
    # print(params.keys())
    
    # print(model.base_model.conv1.weight)
    

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    # mdoel = torch.nn.parallel.DistributedDataParallel(model,device_ids=args.gpus).cuda()   ## what I found
    
    ##########################################################################################################################################
    ##########################################################################################################################################
    print('******** NEW MODEL')
    print(model)


    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    # ## parameters to be updated
    # model.module.base_model.layer3[0].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer3[0].bn1.weight.requires_grad=True
    # model.module.base_model.layer3[0].bn1.bias.requires_grad=True
    # model.module.base_model.layer3[0].conv2.weight.requires_grad=True
    # model.module.base_model.layer3[0].bn2.weight.requires_grad=True
    # model.module.base_model.layer3[0].bn2.bias.requires_grad=True
    # model.module.base_model.layer3[0].conv3.weight.requires_grad=True
    # model.module.base_model.layer3[0].bn3.weight.requires_grad=True
    # model.module.base_model.layer3[0].bn3.bias.requires_grad=True
    # model.module.base_model.layer3[0].downsample[0].weight.requires_grad=True
    # model.module.base_model.layer3[0].downsample[1].weight.requires_grad=True
    # model.module.base_model.layer3[0].downsample[1].bias.requires_grad=True
    # model.module.base_model.layer3[1].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer3[1].bn1.weight.requires_grad=True
    # model.module.base_model.layer3[1].bn1.bias.requires_grad=True
    # model.module.base_model.layer3[1].conv2.weight.requires_grad=True
    # model.module.base_model.layer3[1].bn2.weight.requires_grad=True
    # model.module.base_model.layer3[1].bn2.bias.requires_grad=True
    # model.module.base_model.layer3[1].conv3.weight.requires_grad=True
    # model.module.base_model.layer3[1].bn3.weight.requires_grad=True
    # model.module.base_model.layer3[1].bn3.bias.requires_grad=True
    # model.module.base_model.layer3[2].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer3[2].bn1.weight.requires_grad=True
    # model.module.base_model.layer3[2].bn1.bias.requires_grad=True 
    # model.module.base_model.layer3[2].conv2.weight.requires_grad=True
    # model.module.base_model.layer3[2].bn2.weight.requires_grad=True
    # model.module.base_model.layer3[2].bn2.bias.requires_grad=True
    # model.module.base_model.layer3[2].conv3.weight.requires_grad=True
    # model.module.base_model.layer3[2].bn3.weight.requires_grad=True
    # model.module.base_model.layer3[2].bn3.bias.requires_grad=True
    # model.module.base_model.layer3[3].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer3[3].bn1.weight.requires_grad=True
    # model.module.base_model.layer3[3].bn1.bias.requires_grad=True 
    # model.module.base_model.layer3[3].conv2.weight.requires_grad=True
    # model.module.base_model.layer3[3].bn2.weight.requires_grad=True
    # model.module.base_model.layer3[3].bn2.bias.requires_grad=True
    # model.module.base_model.layer3[3].conv3.weight.requires_grad=True
    # model.module.base_model.layer3[3].bn3.weight.requires_grad=True
    # model.module.base_model.layer3[3].bn3.bias.requires_grad=True
    # model.module.base_model.layer3[4].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer3[4].bn1.weight.requires_grad=True
    # model.module.base_model.layer3[4].bn1.bias.requires_grad=True 
    # model.module.base_model.layer3[4].conv2.weight.requires_grad=True
    # model.module.base_model.layer3[4].bn2.weight.requires_grad=True
    # model.module.base_model.layer3[4].bn2.bias.requires_grad=True
    # model.module.base_model.layer3[4].conv3.weight.requires_grad=True
    # model.module.base_model.layer3[4].bn3.weight.requires_grad=True
    # model.module.base_model.layer3[4].bn3.bias.requires_grad=True
    # model.module.base_model.layer3[5].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer3[5].bn1.weight.requires_grad=True
    # model.module.base_model.layer3[5].bn1.bias.requires_grad=True 
    # model.module.base_model.layer3[5].conv2.weight.requires_grad=True
    # model.module.base_model.layer3[5].bn2.weight.requires_grad=True
    # model.module.base_model.layer3[5].bn2.bias.requires_grad=True
    # model.module.base_model.layer3[5].conv3.weight.requires_grad=True
    # model.module.base_model.layer3[5].bn3.weight.requires_grad=True
    # model.module.base_model.layer3[5].bn3.bias.requires_grad=True
    # model.module.base_model.layer4[0].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer4[0].bn1.weight.requires_grad=True
    # model.module.base_model.layer4[0].bn1.bias.requires_grad=True
    # model.module.base_model.layer4[0].conv2.weight.requires_grad=True
    # model.module.base_model.layer4[0].bn2.weight.requires_grad=True
    # model.module.base_model.layer4[0].bn2.bias.requires_grad=True
    # model.module.base_model.layer4[0].conv3.weight.requires_grad=True
    # model.module.base_model.layer4[0].bn3.weight.requires_grad=True
    # model.module.base_model.layer4[0].bn3.bias.requires_grad=True
    # model.module.base_model.layer4[0].downsample[0].weight.requires_grad=True
    # model.module.base_model.layer4[0].downsample[1].weight.requires_grad=True
    # model.module.base_model.layer4[0].downsample[1].bias.requires_grad=True
    # model.module.base_model.layer4[1].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer4[1].bn1.weight.requires_grad=True
    # model.module.base_model.layer4[1].bn1.bias.requires_grad=True
    # model.module.base_model.layer4[1].conv2.weight.requires_grad=True
    # model.module.base_model.layer4[1].bn2.weight.requires_grad=True
    # model.module.base_model.layer4[1].bn2.bias.requires_grad=True
    # model.module.base_model.layer4[1].conv3.weight.requires_grad=True
    # model.module.base_model.layer4[1].bn3.weight.requires_grad=True
    # model.module.base_model.layer4[1].bn3.bias.requires_grad=True
    # model.module.base_model.layer4[2].conv1.net.weight.requires_grad=True
    # model.module.base_model.layer4[2].bn1.weight.requires_grad=True
    # model.module.base_model.layer4[2].bn1.bias.requires_grad=True 
    # model.module.base_model.layer4[2].conv2.weight.requires_grad=True
    # model.module.base_model.layer4[2].bn2.weight.requires_grad=True
    # model.module.base_model.layer4[2].bn2.bias.requires_grad=True
    # model.module.base_model.layer4[2].conv3.weight.requires_grad=True
    # model.module.base_model.layer4[2].bn3.weight.requires_grad=True
    # model.module.base_model.layer4[2].bn3.bias.requires_grad=True
    # model.module.new_fc.weight.requires_grad= True
    # model.module.new_fc.bias.requires_grad= True


    print('Parameters that will be updated:')
    for name, param in model.named_parameters():
        if param.requires_grad:print(name)
    

    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     print(name)
    # print(policies)
    ##########################################################################################################################################
    ##########################################################################################################################################
    
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(policies,
    #                             args.lr,
    #                             weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                             args.lr)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
    #                new_length=data_length,
    #                modality=args.modality,
    #                image_tmpl=prefix,
    #                random_shift=False,
    #                transform=torchvision.transforms.Compose([
    #                    GroupScale(int(scale_size)),
    #                    GroupCenterCrop(crop_size),
    #                    Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
    #                    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
    #                    normalize,
    #                ]), dense_sample=args.dense_sample),
    #     batch_size=1, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        # criterion = torch.nn.CrossEntropyLoss().cuda()
        
        W_L=torch.tensor([1.0,10.0])
        criterion = torch.nn.CrossEntropyLoss(weight=W_L).cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))


    #########################################################################################################################################
    #########################################################################################################################################
    
    # print(model.module.base_model.conv1.weight)
    # print(model.module.base_model.layer1[1].conv2.weight)
    # print(model.module.base_model.layer3[0].conv1.net.weight)
    # print(model.module.base_model.layer4[2].conv3.weight)
    # print(model.module.new_fc.weight)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
        
        
        # print(model.module.base_model.conv1.weight)
        # print(model.module.base_model.layer1[1].conv2.weight)
        # print(model.module.base_model.layer3[0].conv1.net.weight)
        # print(model.module.base_model.layer4[2].conv3.weight)
        # print(model.module.new_fc.weight)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
    
    


def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    sensitivity=AverageMeter()
    precision=AverageMeter()
    ConfMatrix = AverageMeter_confusion()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()
    # model.eval()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # # my compute output
        # # model.eval()
        # out_temp1=model.module.base_model.conv1(input_var)

        # # model.train()
        # output=model.module.new_fc(out_temp1)

        ## compute output_default
        output = model(input_var)
        
        #compute loss
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
        sens, prec, accu, cm1 = accuracy_sens_prec(output.data, target)

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        sensitivity.update(sens.item(),input.size(0))
        precision.update(prec.item(),input.size(0))
        ConfMatrix.update(cm1.cpu().detach().numpy())

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            Sensi = ConfMatrix.sum[1,1]/(ConfMatrix.sum[1,1]+ConfMatrix.sum[1,0])*100
            Preci = ConfMatrix.sum[1,1]/(ConfMatrix.sum[1,1]+ConfMatrix.sum[0,1])*100
            Accu = (ConfMatrix.sum[0,0]+ConfMatrix.sum[1,1])/(ConfMatrix.sum[0,0]+ConfMatrix.sum[0,1]+ConfMatrix.sum[1,0]+ConfMatrix.sum[1,1])*100
            
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    #   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {accu:.3f} ({Accu:.3f})\t'
                      'Sensitivity {sensitivity.val:.3f} ({Sensi:.3f})'
                      'precision {precision.val:.3f} ({Preci:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, Accu=Accu, top1=top1, sensitivity=sensitivity, precision=precision, lr=optimizer.param_groups[-1]['lr'] * 0.1, Sensi=Sensi, Preci = Preci,accu=accu))  # TODO
            print(output)
            print(cm1)
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    sensitivity=AverageMeter()
    precision=AverageMeter()
    ConfMatrix=AverageMeter_confusion()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # print(input.shape)
            target = target.cuda()
            # print(target)
            # compute output
            output = model(input)
            # print(output)
            # print(output.data)
            loss = criterion(output, target)
            # print(loss)
            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
            prec1, prec5 = accuracy(output.data, target, topk=(1, 1))
            sens, prec, accu, confMat = accuracy_sens_prec(output.data, target)
            
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            sensitivity.update(sens.item(),input.size(0))
            precision.update(prec.item(),input.size(0))
            ConfMatrix.update(confMat.cpu().detach().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Sensitivity {sensitivity.val:.3f} ({sensitivity.avg:.3f})'
                          'precision {precision.val:.3f} ({precision.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, sensitivity=sensitivity, precision=precision))
                print(output)
                print(confMat)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()
    
    Sensi = ConfMatrix.sum[1,1]/(ConfMatrix.sum[1,1]+ConfMatrix.sum[1,0])*100
    Preci = ConfMatrix.sum[1,1]/(ConfMatrix.sum[1,1]+ConfMatrix.sum[0,1])*100
    Accu = (ConfMatrix.sum[0,0]+ConfMatrix.sum[1,1])/(ConfMatrix.sum[0,0]+ConfMatrix.sum[0,1]+ConfMatrix.sum[1,0]+ConfMatrix.sum[1,1])*100

    output = ('Testing Results: Prec@1 {Accu:.3f} Loss {loss.avg:.5f} Sensitivity {Sensi:.3f} Precision {Preci:.3f}'
              .format(top1=top1, loss=losses, Sensi=Sensi, Preci=Preci,Accu=Accu))
    print(output)
    print(ConfMatrix.sum)
    # print(ConfMatrix.sum[1,1]/(ConfMatrix.sum[1,1]+ConfMatrix.sum[1,0])*100)
    # print(ConfMatrix.sum[1,1]/(ConfMatrix.sum[1,1]+ConfMatrix.sum[0,1])*100)
    if log is not None:
        log.write(output + '\n')
        log.flush()

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
