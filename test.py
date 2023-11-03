from cProfile import label
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import torchvision
import network
from model.ANFL import *
from dataset import *
from utils import *
from conf import get_config,set_logger,set_outdir
import argparse
import random
from model.ema import WeightEMA
import itertools
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
log_path=os.path.join(".","log",'trainAddAU')
write1=SummaryWriter(logdir=log_path)
import random
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def hamming_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("两个向量长度不一致")
    return sum(bit1 != bit2 for bit1, bit2 in zip(v1, v2))

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(conf.dataset_path, train=1, fold = 1, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        trainset2 = BP4D(conf.dataset_path, train=2, fold = 2, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader2 = DataLoader(trainset2, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)


        valset = BP4D(conf.dataset_path, train=0, fold=1, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=1, fold = 1, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
        trainset2 = DISFA(conf.dataset_path, train=2, fold = 2, transform=image_train(crop_size=conf.crop_size), crop_size=conf.crop_size, stage = 1)
        train_loader2 = DataLoader(trainset2, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)


        valset = DISFA(conf.dataset_path, train=0, fold=1, transform=image_test(crop_size=conf.crop_size), stage = 1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    return train_loader, train_loader2,val_loader, len(trainset),len(trainset2), len(valset)



# Val
def val(net,val_loader):
    net.eval() 
    statistics_list = None
    for batch_idx, (inputs, targets,p,t) in enumerate(val_loader):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            pred1= net(inputs)
           #####
            #pred1=torch.sigmoid(pred1)
           
            if batch_idx == 0:
                all_output = pred1.data.cpu().float()
                all_label= targets.data.cpu().float()
            else:
                all_output = torch.cat((all_output, pred1.data.cpu().float()), 0)
                all_label = torch.cat((all_label, targets.data.cpu().float()), 0)
    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_label.data.numpy()

    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    missing_label=999
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)
           ####
    #         update_list = statistics(pred1, targets.detach(), 0.5)
    #         statistics_list = update_statistics_list(statistics_list, update_list)
    # mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    # mean_acc, acc_list = calc_acc(statistics_list)
    
    return f1score_arr.mean(), f1score_arr, acc_arr.mean(), acc_arr
     # print(pred1)
           
def main(conf,arg1,arg2,arg3,resume,choice,backbone):
    #1 2为损失系数  3为消融实验设置   1为只有源域  2为只有目标域  3为都没有（一致性损失）
    start_epoch = 0
    # data
    train_loader,train_loader2,val_loader,train_data_num,train_data_num2,val_data_num = get_dataloader(conf)
    list='list'
    train_weight1 = torch.from_numpy(np.loadtxt('./data/'+conf.dataset+'/'+list+'/'+conf.dataset+'_weight_fold1.txt'))
    train_weight2 = torch.from_numpy(np.loadtxt('./data/'+conf.dataset+'/'+list+'/'+conf.dataset+'_weight_fold2.txt'))

    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))
    logging.info("Fold: [{} | {}  train_data_num: {} ]".format(conf.fold, conf.N_fold, train_data_num))
    logging.info("Fold: [{} | {}  trsin_data_num2: {} ]".format(conf.fold, conf.N_fold, train_data_num2))
#'./napshots/resnet50base.pth'
    teacher=MEFARG(num_classes=conf.num_classes, backbone=backbone, neighbor_num=conf.neighbor_num, metric=choice)
    student=MEFARG(num_classes=conf.num_classes, backbone=backbone, neighbor_num=conf.neighbor_num, metric=choice)
    # resume
    if resume != '':
        logging.info("Resume form | {} ]".format(resume))
        teacher = load_state_dict(teacher, resume)
        student = load_state_dict(student, resume)
    if torch.cuda.is_available():
        teacher=teacher.cuda()
        student=student.cuda()
        train_weight1 = train_weight1.cuda()
        train_weight2 = train_weight2.cuda()
    student_detection_params = []
    params = []
    lr=conf.learning_rate
    for key, value in dict(student.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr, \
                            'weight_decay': 0.0005}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': 0.0005}]
            student_detection_params += [value]

    teacher_detection_params = []
    for key, value in dict(teacher.named_parameters()).items():
        if value.requires_grad:
            teacher_detection_params += [value]
            value.requires_grad = False
    studentOptimizer = optim.AdamW(student_detection_params,  betas=(0.9, 0.999), lr=conf.learning_rate, weight_decay=conf.weight_decay)
    teacher_optimizer = WeightEMA(teacher_detection_params, student_detection_params, alpha=0.99)
    criterion1 = WeightedAsymmetricLoss(weight=train_weight1)
    criterion2 = WeightedAsymmetricLoss(weight=train_weight2)
    Lc=nn.MSELoss()
    print('the init learning rate is ', conf.learning_rate)

    #train and val
    bestepoch=0
    bestf1=0
    Teabestepoch=0
    Teabestf1=0
    TarAcc=[1,1,1,1,1,1]       #BP4d评测6个   disfa评测5个   pain测评2个                           
    for epoch in range(start_epoch, conf.epochs):
        lr = studentOptimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        f1score_arrmean, f1score_arr, acc_arrmean, acc_arrS = val(student, val_loader)
        # print("##########val student########")
        # print(acc_arrmean)
        # print(acc_arrS)
        # print("##########val student########")
        Teaf1score_arrmean, Teaf1score_arr, acc_arrmean, acc_arrT = val(teacher, val_loader)             
        if f1score_arrmean>bestf1:
            bestf1=f1score_arrmean
            bestepoch=epoch  
            checkpoint = {
            'epoch': epoch,
            'state_dict': student.state_dict(),
            'optimizer': studentOptimizer.state_dict(),
        }
            
        if Teaf1score_arrmean>Teabestf1:
            Teabestf1=Teaf1score_arrmean
            Teabestepoch=epoch    
             
        print('epoch:'+str(epoch))
        print('f1score_arrmean:'+str(f1score_arrmean))
        print('Tf1score_arrmean:'+str(Teaf1score_arrmean))
        print(f1score_arr)
        print("Sbestf1:"+str(bestf1)+" Sbestepoch:"+str(bestepoch))
        print("Tbestf1:"+str(Teabestf1)+" Tbestepoch:"+str(Teabestepoch))
     
       
        



if __name__=="__main__":
  
    conf = get_config()
   
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    resume='./checkpoints/stu_dots0.4268199344397079__4.pth'
    #resume='./checkpoints/Noise/Gass/stu_dots0.35698353600834015__4.pth'
    #resume='./checkpoints/BP4D_emotioNet/rnn/resnet50_dots0.6223674592997951.pth'
    choice='dots'
    backbone='resnet50'
    
    arg1=[0.01]#5
    arg2=[1.8]#7
    ########Noise  Gass
    for step1 in arg1:#只含源域一致性
        for step2 in arg2:
            main(conf,step1,step2,4,resume,choice,backbone)

    
