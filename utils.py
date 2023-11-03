from math import cos, pi
from re import A
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from sklearn import preprocessing
import math
import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def statistics(pred, y, thresh):
    batch_size = pred.size(0)
    class_nb = pred.size(1)
    pred = pred >= thresh
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                elif y[i][j]==999:
                     jjj=1
                else:
                    
                    assert False
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                elif y[i][j]==999:
                     jjj=1
                else:
                    assert False
            else:
                assert False
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list


def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def calc_acc(statistics_list):
    acc_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']

        acc = (TP+TN)/(TP+TN+FP+FN)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)

    return mean_acc_score, acc_list


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    assert len(old_list) == len(new_list)

    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']

    return old_list


def BP4D_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5])}
    return infostr

def DISFA_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    return infostr


def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):
   
    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class image_train(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


class image_test(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            #transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


class image_test_Linear(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def load_state_dict(model,path):
    checkpoints = torch.load(path,map_location=torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,strict=False)
    return model
class TriLoss(nn.Module):
    def __init__(self, disable_torch_grad=True):
        super(TriLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        

    def forward(self, x, y,z):

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        x_l2=preprocessing.normalize(x,norm='l2')#Fa
        y_l2=preprocessing.normalize(y,norm='l2')#Fn
        z_l2=preprocessing.normalize(z,norm='l2')#Fp
        fea1=y_l2-x_l2
        fea2=x_l2-z_l2
        
        fea1=fea1.flatten()
        fea2=fea2.flatten()
        l1=math.sqrt(sum(e ** 2 for e in fea1))
        l2=math.sqrt(sum(e ** 2 for e in fea2))
        loss=0.5-(l1 - l2)
        return max(0,loss)

class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):

        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1,-1)

        loss = loss.mean(dim=-1)
        return -loss.mean()

def readfile():
    file1=open('./data/BP4D/graphList/emotioNet_train_path.txt','r')
    file2=open('./data/BP4D/graphList/BP4D_train_img_path_fold1.txt','r')
    s1=file1.readline().replace('\n','').replace('./','')
    s2=file2.readline().replace('\n','').replace('./','')
    all1=[]
    all2=[]
    while len(s1)!=0:
        all1.append('/data2/lkl/graph/'+s1)
        s1=file1.readline().replace('\n','').replace('./','')
    while len(s2)!=0:
        all2.append('/data2/lkl/graph/'+s2)
        s2=file2.readline().replace('\n','').replace('./','')
    file3=open('./new/newbp4d.txt','w')
    file4=open('./new/newemo.txt','w')
    for p in all1:
        file4.write(p+'\n')
    for p in all2:
        file3.write(p+'\n')
    file1.close()
    file2.close()
    file3.close()
    file4.close()
def cal_To_label(exp):#统计生成伪标签
    #img  expression  Au1 Au2 Au3 Au4 AU5 AU6
    express=0
    Au1=0
    Au2=0
    Au3=0
    Au4=0
    Au5=0
    Au6=0
    f=open('./new/AllLabelBp4d.txt','r')
    s=f.readline().replace('\n','')
    all=[]
    while len(s)!=0:
        all.append(s)
        s=f.readline().replace('\n','')
    
    for cal in all:
        label=cal.split(' ')
        
        #if int(label[1])==exp:
        if True:
            express=express+1
            if int(label[2])==1:
                Au1=Au1+1
            if int(label[3])==1:
                Au2=Au2+1
            if int(label[4])==1:
                Au3=Au3+1
            if int(label[5])==1:
                Au4=Au4+1
            if int(label[6])==1:
                Au5=Au5+1
            if int(label[7])==1:
                Au6=Au6+1
    result=[]
    result.append(str(float(Au1)/float(express)))
    result.append(str(float(Au2)/float(express)))
    result.append(str(float(Au3)/float(express)))
    result.append(str(float(Au4)/float(express)))
    result.append(str(float(Au5)/float(express)))
    result.append(str(float(Au6)/float(express)))
    print(result)
def getlabel():#多个txt合并
    dic={}
    f1=open('./data/BP4D/graphList/BP4D_train_img_path_fold1.txt','r')#路径
    f2=open('./data/BP4D/graphList/BP4D_train_label_fold1.txt','r')#AU
    s1=f1.readline().replace('\n','')
    s2=f2.readline().replace('\n','')
    while len(s1)!=0:
        dic[s1.split('BP4D/')[1]]=s2
        s1=f1.readline().replace('\n','')
        s2=f2.readline().replace('\n','')
    
    f3=open('./new/bp4dexpress.txt','r')
    s3=f3.readline().replace('\n','')
    all=[]
    while len(s3)!=0:
        all.append(s3)
        s3=f3.readline().replace('\n','')
    
    f4=open('./new/AllLabelBp4d.txt','w')
    for r in all:
        express=r.split(' ')
        if  express[0].split('BP4D/')[1] in dic:
            f4.write(r+" "+dic[express[0].split('BP4D/')[1]]+'\n')
    f4.close()
def deleteNone():
    f=open('./new/AllLabelemo.txt','r')
    all=[]
    s=f.readline().replace('\n','')
    while len(s)!=0:
        arr=s.split(' ')
        if len(arr[1])>3:
            pass
        else:
            all.append(s)
        s=f.readline().replace('\n','')
    f2=open('./new/AllLabelemo2.txt','w')
    for p in all:
        f2.write(p+'\n')
    f2.close()
def getemopresudoLabel():
    exp1=[0, 0, 0, 0, 0 ,0]
    exp2=[1, 1 ,1, 1 ,0, 1]
    exp3=[0, 0 ,1, 1 ,0 ,1]
    exp4=[1, 1 ,0, 1 ,1 ,0]
    exp5=[1, 1 ,0 ,1, 1 ,0]
    exp6=[1, 0, 0 ,0 ,0, 1]
    exp7=[1, 1 ,0 ,0, 1 ,0]
    f=open('./new/AllLabelemo.txt','r')
    s=f.readline()
    label=[]
    path=[]
    while len(s)!=0:
        arr=s.split(' ')
        path.append(arr[0])
        if int(arr[1])==0:
            label.append(exp1)
        if int(arr[1])==1:
            label.append(exp2)
        if int(arr[1])==2:
            label.append(exp3)
        if int(arr[1])==3:
            label.append(exp4)
        if int(arr[1])==4:
            label.append(exp5)
        if int(arr[1])==5:
            label.append(exp6)
        if int(arr[1])==6:
            label.append(exp7)
        s=f.readline()
    np.savetxt('./new/emolabel.txt',label,fmt="%d")
    f2=open('./new/emopath.txt','w')
    for p in path:
        p=p.split('EmotioNet/')[1]
        p='./data/imgs/EmotioNet/'+p
        f2.write(p+'\n')
    f2.close()
def label_smoothing2(inputs, epsilon=0.1):
    #k=inputs.shape[-1]
    k=2
    return ((1-epsilon)*inputs)+(epsilon/k)
def yuzhiLabel(inputs,num1,num2,num3,num4,num5,num6,zero,one):
    Wei_AUoccur_pred = torch.tensor(np.zeros(inputs.shape)).cuda()
    i,j=inputs.shape
    for index1 in range(i):
        if inputs[index1][0]<num1:
            Wei_AUoccur_pred[index1][0]=zero
        else:
            Wei_AUoccur_pred[index1][0]=one
        ##############
        if inputs[index1][1]<num2:
            Wei_AUoccur_pred[index1][1]=zero
        else:
            Wei_AUoccur_pred[index1][1]=one
        #############
        if inputs[index1][2]<num3:
            Wei_AUoccur_pred[index1][2]=zero
        else:
            Wei_AUoccur_pred[index1][2]=one
            ##########
        if inputs[index1][3]<num4:
            Wei_AUoccur_pred[index1][3]=zero
        else:
            Wei_AUoccur_pred[index1][3]=one
            ############
        if inputs[index1][4]<num5:
            Wei_AUoccur_pred[index1][4]=zero
        else:
            Wei_AUoccur_pred[index1][4]=one
            ########
        if inputs[index1][5]<num6:
            Wei_AUoccur_pred[index1][5]=zero
        else:
            Wei_AUoccur_pred[index1][5]=one
    return Wei_AUoccur_pred
def label_smoothing(inputs, epsilon):#输入    准确率
    k=2
    testnum=6  #pain  2    disfa  5    bp4d    6
    epsilon2=[1,1,1,1,1,1]
    for i in range(testnum):
        epsilon2[i]=1-epsilon[i]   #平滑度
    newSmoothlabel=[1,1,1,1,1,1]#返回正标签
    newSmoothlabel[0]=((1-epsilon2[0])*1)+(epsilon2[0]/k)   #平滑标签
    newSmoothlabel[1]=((1-epsilon2[1])*1)+(epsilon2[1]/k)
    newSmoothlabel[2]=((1-epsilon2[2])*1)+(epsilon2[2]/k)
    newSmoothlabel[3]=((1-epsilon2[3])*1)+(epsilon2[3]/k)
    newSmoothlabel[4]=((1-epsilon2[4])*1)+(epsilon2[4]/k)
    newSmoothlabel[5]=((1-epsilon2[5])*1)+(epsilon2[5]/k)
    Wei_AUoccur_pred = torch.tensor(np.zeros(inputs.shape)).cuda()
    i,j=inputs.shape
    for index1 in range(i):
        if inputs[index1][0]<0.5:
            Wei_AUoccur_pred[index1][0]=1-newSmoothlabel[0]
        else:
            Wei_AUoccur_pred[index1][0]=newSmoothlabel[0]
        ##############
        if inputs[index1][1]<0.5:
            Wei_AUoccur_pred[index1][1]=1-newSmoothlabel[1]
        else:
            Wei_AUoccur_pred[index1][1]=newSmoothlabel[1]
        #############
        if inputs[index1][2]<0.5:
           Wei_AUoccur_pred[index1][2]=1-newSmoothlabel[2]
        else:
           Wei_AUoccur_pred[index1][2]=newSmoothlabel[2]
            #########
        if inputs[index1][3]<0.5:
           Wei_AUoccur_pred[index1][3]=1-newSmoothlabel[3]
        else:
           Wei_AUoccur_pred[index1][3]=newSmoothlabel[3]
            ###########
        if inputs[index1][4]<0.5:
           Wei_AUoccur_pred[index1][4]=1-newSmoothlabel[4]
        else:
           Wei_AUoccur_pred[index1][4]=newSmoothlabel[4]
        if inputs[index1][5]<0.5:
           Wei_AUoccur_pred[index1][5]=1-newSmoothlabel[5]
        else:
           Wei_AUoccur_pred[index1][5]=newSmoothlabel[5]
            ########
    return Wei_AUoccur_pred
#高斯分布概率
def prob(data,avg,sig):
    sqrt_2pi=np.power(2*np.pi,0.5)
    coef=1/(sqrt_2pi*sig)
    powercoef=-1/(2*np.power(sig,2))
    mypow=powercoef*(np.power((data-avg),2))
    return coef*(np.exp(mypow))
def Gass_label_smoothing(inputs,TarACc):
    k=2
    u=0.1
    testnum=6  #pain  2    disfa  5    bp4d     6
    epsilon2=[1,1,1,1,1,1]#2222222222222222222222222222222222222222222222222222222222222222222
    epsilon2=prob(np.array(TarACc),0,u)#每一类平滑度
    newSmoothlabel=[1,1,1,1,1,1]#返回正标签#2222222222222222222222222222222222222222222222222222222222222222222
    newSmoothlabel[0]=((1-epsilon2[0])*1)+(epsilon2[0]/k)#每一类正标签
    newSmoothlabel[1]=((1-epsilon2[1])*1)+(epsilon2[1]/k)
    newSmoothlabel[2]=((1-epsilon2[2])*1)+(epsilon2[2]/k)
    newSmoothlabel[3]=((1-epsilon2[3])*1)+(epsilon2[3]/k)
    newSmoothlabel[4]=((1-epsilon2[4])*1)+(epsilon2[4]/k)
    newSmoothlabel[5]=((1-epsilon2[5])*1)+(epsilon2[5]/k)
    Wei_AUoccur_pred = torch.tensor(np.zeros(inputs.shape)).cuda()
    i,j=inputs.shape
    for index1 in range(i):#计算batch size的标签
        if inputs[index1][0]<0.5:
            Wei_AUoccur_pred[index1][0]=1-newSmoothlabel[0]
        else:
            Wei_AUoccur_pred[index1][0]=newSmoothlabel[0]
        ##############
        if inputs[index1][1]<0.5:
            Wei_AUoccur_pred[index1][1]=1-newSmoothlabel[1]
        else:
            Wei_AUoccur_pred[index1][1]=newSmoothlabel[1]
        #############
        if inputs[index1][2]<0.5:
            Wei_AUoccur_pred[index1][2]=1-newSmoothlabel[2]
        else:
            Wei_AUoccur_pred[index1][2]=newSmoothlabel[2]
                #########
        if inputs[index1][3]<0.5:
            Wei_AUoccur_pred[index1][3]=1-newSmoothlabel[3]
        else:
            Wei_AUoccur_pred[index1][3]=newSmoothlabel[3]
                ###########
        if inputs[index1][4]<0.5:
            Wei_AUoccur_pred[index1][4]=1-newSmoothlabel[4]
        else:
            Wei_AUoccur_pred[index1][4]=newSmoothlabel[4]
        if inputs[index1][5]<0.5:
            Wei_AUoccur_pred[index1][5]=1-newSmoothlabel[5]
        else:
            Wei_AUoccur_pred[index1][5]=newSmoothlabel[5]
            ########
    
    return Wei_AUoccur_pred


#EST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def normalization(data):
       _range = np.max(data) - np.min(data)
       return (data - np.min(data)) / _range

def sigmoid(data):
       s = 1 / (1 + np.exp(-data))
       return s

def softmax(x):
       e_x = np.exp(x - np.max(x))
       return e_x / e_x.sum()
def draw(y,path):
    plt.cla()
    matplotlib.rcParams.update({'font.size': 10})
    ax = plt.gca()
    border_width = 2
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['left'].set_linewidth(border_width)
    ax.spines['right'].set_linewidth(border_width)

    x = [tmp for tmp in range(1,29,4)]
    # y = np.array([0.4324539, 0.42874807, 0.4255728 , 0.42591235, 0.42857426,
    #         0.42726862 , 0.43146998])
    print(y)
    y = sigmoid(np.array(y))
   # print(y)
    print('normalization y: ',y)
    y_argmax = np.argmax(y)
    l1 = plt.plot(x,y,'-',label='EST',marker='o',mec='slateblue',c='slateblue',ms=12,linewidth=6)
    plt.plot(x[y_argmax],y[y_argmax],'ys',ms=14)
    plt.xlabel('snippts')
    plt.ylabel('attention weight')
    #plt.ylim(ymin=0.4275,ymax=0.4295)
    #plt.ylim(ymin=0.4255,ymax=0.4325)
    #plt.ylim(ymin=0.60545,ymax=0.60559999)
    #plt.ylim(ymin=0.3,ymax=0.7)
    plt.ylim(ymin=0.535,ymax=0.536)
    plt.xlim(xmin=-2,xmax=29)
    #plt.ylim(ymin=-0.1,ymax=1.2)
    #plt.xticks(x,['1','2','3','4','5','6','7'])
    plt.xticks(x,['1','2','3','4','5','6','7'])
    #plt.show()
    plt.savefig(path+'/mean2.jpg')
def draw2(y,y2,path):
    plt.cla()
    matplotlib.rcParams.update({'font.size': 20})
    plt.subplot(121)
    plt.figure(figsize=(22,8))#30   12
    ax = plt.subplot(121)
    border_width = 2
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['left'].set_linewidth(border_width)
    ax.spines['right'].set_linewidth(border_width)

    x = [tmp for tmp in range(1,29,4)]
    # y = np.array([0.4324539, 0.42874807, 0.4255728 , 0.42591235, 0.42857426,
    #         0.42726862 , 0.43146998])
    
    y = sigmoid(np.array(y))
    print(y)
    y=(y-0.53)*100
    print('normalization y: ',y)
    y_argmax = np.argmax(y)
    l1 = plt.plot(x,y,'-',label='EST',marker='o',mec='slateblue',c='slateblue',ms=12,linewidth=8)
    plt.plot(x[y_argmax],y[y_argmax],'ys',ms=14)
    plt.xlabel('snippts')
    plt.ylabel('attention weight')
    plt.ylim(ymin=0.53,ymax=0.6)
    plt.xlim(xmin=-2,xmax=29)
    plt.xticks(x,['1','2','3','4','5','6','7'])
  
    ax2 = plt.subplot(122)
    border_width = 2
    ax2.spines['top'].set_linewidth(border_width)
    ax2.spines['bottom'].set_linewidth(border_width)
    ax2.spines['left'].set_linewidth(border_width)
    ax2.spines['right'].set_linewidth(border_width)

    x2 = [tmp for tmp in range(1,29,4)]
    # y = np.array([0.4324539, 0.42874807, 0.4255728 , 0.42591235, 0.42857426,
    #         0.42726862 , 0.43146998])
    y2 = sigmoid(np.array(y2))
    #print(y2)
    y2=(y2-0.53)*100
    #y2=((y2-0.53)*100-0.56)*100
    #print('normalization y: ',y2)
    y_argmax2 = np.argmax(y2)
    l1 = plt.plot(x2,y2,'-',label='EST',marker='o',mec='slateblue',c='slateblue',ms=12,linewidth=8)
    plt.plot(x2[y_argmax2],y2[y_argmax2],'ys',ms=14)
    plt.xlabel('snippts')
    plt.ylabel('attention weight')
   # plt.ylim(ymin=0.535625,ymax=0.535675)
    plt.ylim(ymin=0.56,ymax=0.57)
    plt.xlim(xmin=-2,xmax=29)
    plt.xticks(x,['1','2','3','4','5','6','7'])
    print(ax2.get_figure())
    plt.savefig(path+'/mean2.jpg')
def read_npy2(path,path2):
    loadData0=np.load(path+'0.npy')
    loadData1=np.load(path+'1.npy')
    loadData2=np.load(path+'2.npy')
    data3=(loadData0+loadData1+loadData2)/3
    loadData3=np.load(path2+'0.npy')
    loadData4=np.load(path2+'1.npy')
    loadData5=np.load(path2+'2.npy')
    data4=(loadData3+loadData4+loadData5)/3
    draw2(data3,data4,path)
import seaborn as sns
def read_npy(path):
    loadData0=np.load(path+'0.npy')
    loadData1=np.load(path+'1.npy')
    loadData2=np.load(path+'2.npy')
    data3=(loadData0+loadData1+loadData2)/3
    draw2(data3,path)
   # draw_weight(data3,path)
    np.save(path+'mean.npy',data3)
def draw_weight(y,path):
    print(y.shape)
    print(y[:, np.newaxis].shape)
    y=y[:, np.newaxis]
    y=y.reshape(1,7)
    ax=sns.heatmap(y)
    heatmap=ax.get_figure()
    heatmap.savefig(path+'/mean.jpg',dpi=300)
    plt.close()




if __name__=="__main__":
    
    #m038    F048
    read_npy2('/data1/wwb/EST/heat_map/bu3d_just_no_ssop_test/M038/Sad/trans/','/data1/wwb/EST/heat_map/bu3d_test/M038/Sad/trans/')
    #read_npy2('/data1/wwb/EST/heat_map/bu3d_just_no_ssop_test/F048/Fear/trans/','/data1/wwb/EST/heat_map/bu3d_test/F048/Fear/trans/')
    #read_npy('/data1/wwb/EST/heat_map/bu3d_test/M038/Sad/trans/')
    # read_npy('/data1/wwb/EST/heat_map/bu3d_just_no_ssop_test/M038/Sad/trans/')
    # read_npy('/data1/wwb/EST/heat_map/bu3d_test/M038/Sad/trans/')
    # read_npy('/data1/wwb/EST/heat_map/bu3d_just_no_ssop_test/F054/Fear/trans/')
    # initpath1='/data1/wwb/EST/heat_map/bu3d_just_no_ssop_test/'
    # initpath2='/data1/wwb/EST/heat_map/bu3d_test/'
    # path1=[]
    # path1.append(initpath1)
    # path1.append(initpath2)
    # path2=['F048','F049','F050','F051','F052','F053','F054','F055','F056','F057','F058',
    #        'M035','M036','M037','M038','M039','M040','M041','M042','M043']
    # path3=['Angry','Disgust','Fear','Happy','Sad','Surprise']
    # allpath=[]
    # for p1 in path1:
    #     for p2 in path2:
    #         for p3 in path3:
    #             path=p1+p2+'/'+p3+'/'+'trans'+'/'
    #             allpath.append(path)
   
    # for i in allpath:
    #     read_npy(i)
            

    #read_npy('/data1/wwb/EST/heat_map/bu3d_just_no_ssop_test/F052/Happy/trans/')
    # label=torch.tensor([        [0.49,0.8,0.2,0.1,0.9,0.7],
    #                             [0.3,0.92,0.1,0.1,0.1,0.1],
    #                             [0.46,0.88,0.2,0.2,0.2,0.2]
                                
                                
                                
                                
    #                             ])
    
    # print(yuzhiLabel(label,0.4,0.8,0.15,0.3,0.8,0.6))
   