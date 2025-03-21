import argparse
import os
import random
import time
import warnings
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from engine_for_finetuning import train_one_epoch, evaluate
import timm
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import PIL
import utils

def main_worker(args):
    
    print('参数:', args)

    # mean,std from imagenet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #mean,std from OS
    normalize = transforms.Normalize(mean = [0.50351185, 0.30116007, 0.20442231], std = [0.2821921, 0.22173707, 0.17406568])

    train_transforms = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
            # transforms.RandomResizedCrop((image_size, image_size) ),  #RandomRotation scale=(0.9, 1.0)
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    if not args.evaluate:
        # train
        train_dataset = datasets.ImageFolder(args.train_dir,train_transforms)
        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

        train_set_name = os.path.basename(args.train_dir)
        model_store_dir = os.path.join(args.output_dir, args.arch, args.arch + '_' + train_set_name + '_' + args.tag)
        os.makedirs(model_store_dir, exist_ok=True)
        args.model_store_dir = model_store_dir
        print('model store path:', model_store_dir)

    val_dataset = datasets.ImageFolder(args.val_dir, val_transforms)
    
    print ('classes:', val_dataset.classes)
    # Get number of labels
    n_classes = len(val_dataset.classes)
    print('batch_size:{}, num_workers:{}'.format(args.batch_size, args.workers))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate or args.fine_tuning:
        model = timm.create_model(args.arch, pretrained=False, num_classes=n_classes)
    else:
        model = timm.create_model(args.arch, pretrained=args.pretrained, num_classes=n_classes)
    
    if args.fine_tuning:
        utils.load_state_dict_from_pretrained(model, args)
    
    model_without_ddp = model
    if args.gpu is not None:
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        model.to(device)
        if args.parallel:
            model = torch.nn.DataParallel(model)
            model_without_ddp = model.module

    print('label smoothing :{}, class_weights:{}'.format(args.smoothing, args.class_weight))
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing, weight=torch.Tensor(args.class_weight)).cuda()
    
    print('train lr:{}, momentum:{}, weight_decay:{}'.format(args.lr, args.momentum, args.weight_decay))
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    cudnn.benchmark = True
    begin_time = time.time()
    if args.evaluate:
        res = evaluate(val_loader, model, criterion, args)
        print('预测耗时：{}s'.format(int(time.time() - begin_time)))
        return

    #auto resume
    utils.auto_load_model(args, model, model_without_ddp, optimizer)

    # times = []
    epoch_list, train_loss_list, test_loss_list, aucs_list, step_losses = [], [], [], [], []
    max_acc = 0.
    min_loss = 1000.0
    
    for epoch in range(args.start_epoch, args.epochs):
        # time1 = time.time() 
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        losses, step_loss, acc = train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)

        print('Epoch:{}, loss:{:.3f}, train Acc: {:.3f}'.format(epoch, losses.avg, acc))
        train_loss_list.append(losses.avg)
        epoch_list.append(epoch+1)

        # evaluate on validation set
        acc1,test_losses,aucs = evaluate(val_loader, model, criterion, args)
        test_loss_list.append(test_losses.avg)
        aucs_list.append(aucs)
        step_losses.extend(step_loss)
        # remember best acc@1 and save checkpoint
        
        if args.parallel:
            model_ = model.module
        else:
            model_ = model
            
        save_obj = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'model': model_.state_dict(),
            'max_acc': max_acc,
            'optimizer' : optimizer.state_dict(),
            'model_args': vars(args),
        }
        
        #保留最近的模型训练文件
        torch.save(save_obj, os.path.join(model_store_dir, 'checkpoint.pth'))
        #保存最大acc的模型
        if acc1 > max_acc:
            max_acc = acc1
            torch.save(save_obj, os.path.join(model_store_dir, 'checkpoint-max-acc.pth'))
        
        #保存最小loss的模型
        if test_losses.avg < min_loss:
            min_loss = test_losses.avg
            torch.save(save_obj, os.path.join(model_store_dir, 'checkpoint-min-loss.pth'))
            
        # time2 = time.time()
        # print("第",epoch,"epoch的时间为",time2-time1)
        # times.append(time2-time1)
    print('Finished Training best_acc: ', max_acc)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 20))
    lr_decay = 0.1 ** (epoch // 20)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay
        # param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--train_dir', metavar='DIR', default='',help='path to dataset')
    parser.add_argument('--val_dir', metavar='DIR', default='', required=True, help='path to dataset')
    parser.add_argument('--output_dir', metavar='DIR', default='', help='path to model file.')
    parser.add_argument('--tag', metavar='TAG', default='', help='path to dataset')
    parser.add_argument('--fine_tuning',default='', help='pretrained model path.')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_large_patch16_224',
                        help='model architecture.')

    def parse_weights(weights):
        return [float(w) for w in weights.split(',')]

    parser.add_argument('--class_weight', type=parse_weights, default='1.0,1.0,1.0',
                        help='Comma-separated list of weights (e.g., 1.0,1.0,1.0)')
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.0)')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--auto_resume', action='store_true', default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate moinrtdel on validation set')
    parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--model_key', default='model|module', type=str)

    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--image_size', default=224, type=int,
                        help='image size')
    parser.add_argument('--advprop', default=False, action='store_true',
                        help='use advprop or not')
    parser.add_argument('--gpu', default='0,1,2,3', help='gpu ids.')
    
    args = parser.parse_args()
    
    utils.set_gpu(args.gpu)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    args.parallel = False
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        #使用多个gpu，采用并行的模式
        if len(args.gpu.split(',')) > 1:
            args.parallel = True
    
    main_worker(args)

