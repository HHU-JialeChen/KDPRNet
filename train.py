from __future__ import print_function
from __future__ import division

import os
import sys
import datetime
import time

import argparse
import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import tqdm
from models.net import Model

from data_manager import DataManager
from utils.losses import CrossEntropyLoss
from utils.optimizers import init_optimizer

from utils.iotools import save_checkpoint
from utils import AverageMeter
from utils import Logger
from utils.torchtools import one_hot, adjust_learning_rate

tsne=False

class WeightedCrossEntropyLoss(nn.Module):  
    def __init__(self):  
        super(WeightedCrossEntropyLoss, self).__init__()  
  
    def forward(self, logits, targets, weights):  
        loss = F.cross_entropy(logits, targets)   
        #if weights is not None:  
        loss = loss * weights  
        return loss.mean()

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    print("Currently using GPU {}".format(args.gpu_devices))
    cudnn.benchmark = False  #
    cudnn.deterministic = True

    args.save_dir = os.path.join(args.save_dir, args.dataset, args.model, str(args.nKnovel)+"_"+str(args.nExemplars)+"_"+"model_"+'global_' + str(args.weight_global) )

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    Dataset = DataManager(args, use_gpu)
    trainloader, testloader = Dataset.return_dataloaders()
    print('Initializing image data manager')


    model = Model(num_classes=args.num_classes, backbone=args.model,save_dir=args.save_dir,is_tsne=tsne,num_sample=args.nKnovel*args.nExemplars+75) ###num_classer=100

    if use_gpu:
        model = model.cuda()
    print("save checkpoint to '{}'".format(args.save_dir))


    criterion1 = torch.nn.CrossEntropyLoss().cuda()
    criterion2 = torch.nn.CrossEntropyLoss().cuda()
    criterion3 = WeightedCrossEntropyLoss().cuda()#torch.nn.CrossEntropyLoss().cuda()
    #optimizer = init_optimizer(args.optim, model.parameters(), args.lr, args.weight_decay)
    optimizer = init_optimizer(args.optim, [  
    {'params': [p for name, p in model.named_parameters() if 'base' in name], 'lr': 0.1},  
    {'params': [p for name, p in model.named_parameters() if 'base' not in name], 'lr': 0.1}  
], args.lr, args.weight_decay)


    if args.model == 'C':
        args.max_epoch = 160
        args.LUT_lr = [(40, 0.03), (70, 0.02), (100, 0.01), (120, 0.005), (140, 0.0025), (160, 0.001)]
    if args.model == 'R':
        args.max_epoch = 160
        args.LUT_lr = [(45, 0.02), (70, 0.01), (85, 0.006), (100,0.002), (120,0.001), (140,0.0005), (160,0.0001)]
        
    if args.model == 'R18':
        args.max_epoch = 120
        args.LUT_lr = [(45, 0.1), (65, 0.08),(85,0.06),(100,0.04),(110,0.02),(120,0.01)]

    start_time = time.time()
    train_time = 0
    best_acc = -np.inf
    best_epoch = 0
    
    print("LR:{}".format(', '.join([f'({x[0]}, {x[1]})' for x in args.LUT_lr])))

    for epoch in range(args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)

        start_train_time = time.time()
        train(args, epoch, model, criterion1, criterion2,criterion3, optimizer, trainloader, learning_rate, use_gpu, args.nKnovel*args.nExemplars)
        train_time += round(time.time() - start_train_time)

        if epoch == 0 or epoch % 5 == 0 or (epoch >= (args.LUT_lr[0][0] - 1) and epoch % 2 == 0):
            acc = ttest(model, testloader, use_gpu, args.nKnovel*args.nExemplars,epoch)
            is_best = acc > best_acc
            
            if is_best:
                best_acc = acc
                best_epoch = epoch + 1
            
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    print("==========\nArgs:{}\n==========".format(args))


def train(args, epoch, model, criterion1, criterion2,criterion3, optimizer, trainloader, learning_rate, use_gpu,num_train):
    losses = AverageMeter()
    losses_glo = AverageMeter()
    losses_glo_sim = AverageMeter()
    losses_xcos = AverageMeter()
    losses_distance = AverageMeter()
    losses_sup = AverageMeter()
    losses_kl = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids_1, pids_2,s_wordvec) in enumerate(tqdm(trainloader)):

        data_time.update(time.time() - end)
        
        images_train, labels_train = images_train.cuda(), labels_train.cuda()
        images_test, labels_test = images_test.cuda(), labels_test.cuda()
        pids_1 = pids_1.cuda()
        pids_2 = pids_2.cuda()
        s_wordvec = s_wordvec.cuda()


        q_batch_size,num_test_examples = images_test.size(0),images_test.size(1)
        labels_train_1hot = one_hot(labels_train).cuda()
        labels_test_1hot = one_hot(labels_test).cuda()

        s1, s2, output1 , output2, sup_loss,distance_loss = model(images_train, images_test, labels_train_1hot, labels_test_1hot,pids_1,epoch,s_wordvec)#,marginLoss

        loss_xcos1 = criterion1(s1, labels_test.view(-1))
        loss_xcos2 = criterion1(s2, labels_test.view(-1))
        loss_xcos = loss_xcos1 +loss_xcos2
        
        loss_xcos_output1 = criterion1(output1.view(-1,5), torch.cat([labels_train,labels_test],dim=1).view(-1))
        loss_xcos_output2 = criterion1(output2.view(-1,5), torch.cat([labels_train,labels_test],dim=1).view(-1))
        loss_xcos_output = loss_xcos_output1 +  loss_xcos_output2
        
        loss = loss_xcos+loss_xcos_output+ sup_loss + 0.1*distance_loss 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids_2.size(0))

        losses_xcos.update(loss_xcos.item(), pids_2.size(0))
        losses_distance.update(distance_loss.item(), pids_2.size(0))
        losses_sup.update(sup_loss.item(), pids_2.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print('[{0}] '
          'Epoch{1} '
          'lr: {2} '
          'Loss:{loss.avg:.4f} '
          'Loss_xcos:{loss_xcos.avg:.4f} '
          'Loss_sup:{loss_sup.avg:.4f} '
          'Loss_distance:{loss_distance.avg:.4f} ' 
          .format(tm,
           epoch+1, learning_rate, loss=losses,loss_xcos=losses_xcos,loss_sup=losses_sup,loss_distance=losses_distance))        


def ttest(model, testloader, use_gpu, num_train,epoch):
    accs = AverageMeter()
    accs1 = AverageMeter()
    accs2 = AverageMeter()
    accs3 = AverageMeter()
    accs4 = AverageMeter()
    accs124 = AverageMeter()
    accs14 = AverageMeter()
    accs24 = AverageMeter()
    
    test_accuracies = []
    test_accuracies1 = []
    test_accuracies2 = []
    test_accuracies3 = []
    test_accuracies4 = []
    test_accuracies124 = []
    test_accuracies14 = []
    test_accuracies24 = []
    model.eval()

    with torch.no_grad():
        for batch_idx , (images_train, labels_train, images_test, labels_test,s_wordvec) in enumerate(tqdm(testloader)):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()


            q_batch_size,num_test_examples = images_test.size(0), images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()
            s_wordvec = s_wordvec.cuda()

            s1,s2, output1, output2= model(images_train, images_test, labels_train_1hot, labels_test_1hot,None,epoch,s_wordvec)#
            s3 = output1[:,num_train:,:].reshape(-1, 5)#[:,num_train:,:]
            s4 = output2[:,num_train:,:].reshape(-1, 5)#[:,num_train:,:]
            s_he=s1*0.25 + s2*0.25+s3*0.25 + s4*0.25 

            s_he = s_he.view(q_batch_size * num_test_examples, -1)
            s1 = s1.view(q_batch_size * num_test_examples, -1)
            s2 = s2.view(q_batch_size * num_test_examples, -1)
            s3 = s3.view(q_batch_size * num_test_examples, -1)
            s4 = s4.view(q_batch_size * num_test_examples, -1)
            s124 = (s1*0.5+s2*0.5).view(q_batch_size * num_test_examples, -1)
            s14 = (s1*0.5+s3*0.5).view(q_batch_size * num_test_examples, -1)
            s24 = (s2*0.5+s4*0.5).view(q_batch_size * num_test_examples, -1)
            
            labels_test = labels_test.view(q_batch_size * num_test_examples).detach().cpu().float()


            _, preds = torch.max(s_he.detach().cpu(), 1)
            _, preds1 = torch.max(s1.detach().cpu(), 1)
            _, preds2 = torch.max(s2.detach().cpu(), 1)
            _, preds3 = torch.max(s3.detach().cpu(), 1)
            _, preds4 = torch.max(s4.detach().cpu(), 1)
            _, preds124 = torch.max(s124.detach().cpu(), 1)
            _, preds14 = torch.max(s14.detach().cpu(), 1)
            _, preds24 = torch.max(s24.detach().cpu(), 1)
            
            gt = (preds == labels_test).float()
            gt1 = (preds1 == labels_test).float()
            gt2 = (preds2 == labels_test).float()
            gt3 = (preds3 == labels_test).float()
            gt4 = (preds4 == labels_test).float()
            gt124 = (preds124 == labels_test).float()
            gt14 = (preds14 == labels_test).float()
            gt24 = (preds24 == labels_test).float() 
            
            acc = (torch.sum(gt).float()) / labels_test.size(0)
            acc1 = (torch.sum(gt1).float()) / labels_test.size(0)
            acc2 = (torch.sum(gt2).float()) / labels_test.size(0)
            acc3 = (torch.sum(gt3).float()) / labels_test.size(0)
            acc4 = (torch.sum(gt4).float()) / labels_test.size(0)
            acc124 = (torch.sum(gt124).float()) / labels_test.size(0)
            acc14 = (torch.sum(gt14).float()) / labels_test.size(0)
            acc24 = (torch.sum(gt24).float()) / labels_test.size(0)
            
            accs.update(acc.item(), labels_test.size(0))
            accs1.update(acc1.item(), labels_test.size(0))
            accs2.update(acc2.item(), labels_test.size(0))
            accs3.update(acc3.item(), labels_test.size(0))
            accs4.update(acc4.item(), labels_test.size(0))
            accs124.update(acc124.item(), labels_test.size(0))
            accs14.update(acc14.item(), labels_test.size(0))
            accs24.update(acc24.item(), labels_test.size(0))        
            
            gt = gt.view(q_batch_size, num_test_examples).numpy() #[b, n]
            gt1 = gt1.view(q_batch_size, num_test_examples).numpy() #[b, n]
            gt2 = gt2.view(q_batch_size, num_test_examples).numpy() #[b, n]
            gt3 = gt3.view(q_batch_size, num_test_examples).numpy() #[b, n]
            gt4 = gt4.view(q_batch_size, num_test_examples).numpy() #[b, n]
            gt124 = gt124.view(q_batch_size, num_test_examples).numpy() #[b, n]
            gt14 = gt14.view(q_batch_size, num_test_examples).numpy() #[b, n]
            gt24 = gt24.view(q_batch_size, num_test_examples).numpy() #[b, n]
            
            acc = np.sum(gt, 1) / num_test_examples
            acc1 = np.sum(gt1, 1) / num_test_examples
            acc2 = np.sum(gt2, 1) / num_test_examples
            acc3 = np.sum(gt3, 1) / num_test_examples
            acc4 = np.sum(gt4, 1) / num_test_examples
            acc124 = np.sum(gt124, 1) / num_test_examples
            acc14 = np.sum(gt14, 1) / num_test_examples
            acc24 = np.sum(gt24, 1) / num_test_examples
            
            acc = np.reshape(acc, (q_batch_size))
            acc1 = np.reshape(acc1, (q_batch_size))
            acc2 = np.reshape(acc2, (q_batch_size))
            acc3 = np.reshape(acc3, (q_batch_size))
            acc4 = np.reshape(acc4, (q_batch_size))
            acc124 = np.reshape(acc124, (q_batch_size))
            acc14 = np.reshape(acc14, (q_batch_size))
            acc24 = np.reshape(acc24, (q_batch_size))
            
            test_accuracies.append(acc)
            test_accuracies1.append(acc1)
            test_accuracies2.append(acc2)
            test_accuracies3.append(acc3)
            test_accuracies4.append(acc4)
            test_accuracies124.append(acc124)
            test_accuracies14.append(acc14)
            test_accuracies24.append(acc24)

    accuracy = accs.avg
    accuracy1 = accs1.avg
    accuracy2 = accs2.avg
    accuracy3 = accs3.avg
    accuracy4 = accs4.avg
    accuracy124 = accs124.avg
    accuracy14 = accs14.avg
    accuracy24 = accs24.avg
    
    test_accuracies = np.array(test_accuracies)
    test_accuracies1 = np.array(test_accuracies1)
    test_accuracies2 = np.array(test_accuracies2)
    test_accuracies3 = np.array(test_accuracies3)
    test_accuracies4 = np.array(test_accuracies4)
    test_accuracies124 = np.array(test_accuracies124)
    test_accuracies14 = np.array(test_accuracies14)
    test_accuracies24 = np.array(test_accuracies24)
    
    test_accuracies = np.reshape(test_accuracies, -1)
    test_accuracies1 = np.reshape(test_accuracies1, -1)
    test_accuracies2 = np.reshape(test_accuracies2, -1)
    test_accuracies3 = np.reshape(test_accuracies3, -1)
    test_accuracies4 = np.reshape(test_accuracies4, -1)
    test_accuracies124 = np.reshape(test_accuracies124, -1)
    test_accuracies14 = np.reshape(test_accuracies14, -1)
    test_accuracies24 = np.reshape(test_accuracies24, -1)
    
    stds = np.std(test_accuracies, 0)

    
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)  


    
    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
   
    print('[{}] Accuracyhe: {:.2%}, stdhe: {:.2%},Accuracy1: {:.2%}, Accuracy2: {:.2%}, Accuracy3: {:.2%},Accuracy4: {:.2%},Accuracy124: {:.2%},Accuracy14: {:.2%},Accuracy24: {:.2%}'.format(tm,accuracy, ci95,accuracy1,  accuracy2,accuracy3,  accuracy4, accuracy124, accuracy14, accuracy24))

    return accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--dataset', type=str, default='CUB-200-2011')
    parser.add_argument('--model', default='C', type=str,
                        help='C for ConvNet, R for ResNet')
    parser.add_argument('--workers', default=16, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--train-batch', default=2, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=2, type=int,
                        help="test batch size")
    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--num_classes', type=int, default=120)
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save-dir', type=str, default='./result/')
    parser.add_argument('--resume', type=str, default="", metavar='PATH')
    parser.add_argument('--gpu-devices', default='3', type=str)
    # ************************************************************
    # Loss settings
    # ************************************************************
    parser.add_argument('--weight_global', type=float, default=1.)
    parser.add_argument('--weight_local', type=float, default=0.1)
    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--nExemplars', type=int, default=5,
                        help='number of training examples per novel category.')
    parser.add_argument('--train_nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--train_epoch_size', type=int, default=1200,
                        help='number of batches per epoch when training')
    parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=600,
                        help='number of batches per epoch')

    parser.add_argument('--phase', default='val', type=str,
                        help='use test or val dataset to early stop')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    main(args)
