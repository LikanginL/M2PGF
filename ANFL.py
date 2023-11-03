from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101,resnet152,resnet34
from .graph import normalize_digraph
from .basic_block import *

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)

        return output
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()
        self.linear=nn.Linear(num_classes*512,num_classes,bias=False)
        self.batch_norm = nn.BatchNorm1d(num_classes*512)
       # self.rnn = nn.RNN(512, 1024)
        self.fc = nn.Linear(1024, 512)
        #self.lstm=BiLSTM(input_dim=512,hidden_dim=512,output_dim=512)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 8, 512,0.1),
             4)
        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)#[b,6,49,512]
        
        f_v = f_u.mean(dim=-2)#[b,6,512]
#########################################################################   RNN
# 假设输入数据为 x，形状为 (seq_len, batch_size, input_size)   6  b  512
        # f_v=f_v.transpose(0,1)#6  b  512
        # h0 = torch.zeros(1, f_v.size(1), 1024).cuda()
        # output, hn = self.rnn(f_v, h0)
        # f_v = self.fc(output)
        # f_v=f_v.transpose(0,1)#b  6  512
        # b, n, c = f_v.shape
        # sc = self.sc
        # sc = self.relu(sc)
        # sc = F.normalize(sc, p=2, dim=-1)
        # cl = F.normalize(f_v, p=2, dim=-1)
        # cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        # cl=torch.sigmoid(cl)
        # fea=1
        #############################################transformer
        # f_v = self.transformer_encoder(f_v)  # 使用Transformer进行特征提取和关系建模
        # b, n, c = f_v.shape
        # sc = self.sc
        # sc = self.relu(sc)
        # sc = F.normalize(sc, p=2, dim=-1)
        # cl = F.normalize(f_v, p=2, dim=-1)
        # cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        # cl=torch.sigmoid(cl)
        # fea=1
       #########################################Bilstm
        # f_v=f_v.transpose(0,1)#6  b  512
        # f_v=self.lstm(f_v)#6  b  512
        # f_v=f_v.transpose(0,1)#b  6  512
        # fea=f_v
        # b, n, c = f_v.shape
        # sc = self.sc
        # sc = self.relu(sc)
        # sc = F.normalize(sc, p=2, dim=-1)
        # cl = F.normalize(f_v, p=2, dim=-1)
        # cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        # cl=torch.sigmoid(cl)
        ###########################FGG
        fea=self.gnn(f_v)#Tsne 特征
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        ############################linear AU
        # b, n, c = f_v.shape
        # f_v=f_v.view(b,-1)#[b,numclass*512]
        # f_v=self.batch_norm(f_v)
        # cl=self.linear(f_v)
        # cl=torch.sigmoid(cl)
        ############
        return cl,fea
class AU_Detect2(nn.Module):
    def __init__(self, au_num):
        super(AU_Detect2, self).__init__()
        self.in_channels = 512
        self.num_classes = 6
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        b, n, c = x.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(x, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl
class MEFARG(nn.Module):
    def __init__(self, num_classes=6, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(MEFARG, self).__init__()
        
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18(pretrained=True)
            elif backbone == 'resnet101':
                self.backbone = resnet101(pretrained=True)
            elif backbone == 'resnet34':
                self.backbone = resnet34(pretrained=True)
            elif backbone == 'resnet152':
                self.backbone = resnet152(pretrained=True)
            else:
                self.backbone = resnet50(pretrained=True)
            self.in_channels = self.backbone.fc.weight.shape[1]#2048
            self.out_channels = self.in_channels // 4#512
            
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        
        self.head = Head(self.out_channels, num_classes, neighbor_num, metric)
    
    def forward(self, x):
        #[b,3,224,224]
        x = self.backbone(x) #提取的特征
        
        #[b,49,2048]
        x = self.global_linear(x)
        #[b,49,512]
        
        cl,fea = self.head(x)#提取特征
       
        return cl

class MEFARG2(nn.Module):
    def __init__(self, num_classes=6, neighbor_num=4, metric='dots'):
        super(MEFARG2, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.num_classes=num_classes
        self.in_channels = self.backbone.fc.weight.shape[1]#2048
        self.out_channels = self.in_channels // 4#512
        self.backbone.fc = None
        self.gnn = GNN(self.out_channels, num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.out_channels)))
        self.relu = nn.ReLU()
        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        
        self.head = Head(self.out_channels, num_classes, neighbor_num, metric)
        
    def forward(self, f_u):
        f_v = f_u.mean(dim=-2)#[b,6,512]
        # FGG
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl


class Au_Linear(nn.Module):
    def __init__(self,input_fc=512,out_fc=6):
        super(Au_Linear,self).__init__()
        self.batch_norm=nn.BatchNorm1d(input_fc)
        self.linear=nn.Linear(input_fc,out_fc,bias=False)
    def forward(self,in_put):
        #64 6 512
        in_put=in_put.view(-1,512)
       
        out=self.batch_norm(in_put)
        out=self.linear(out)
        return out 



#[b,6,49,512]
class AUFea(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AUFea, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x): #[b,49,512]
        # AFG
        f_u = []
        for i, layer in enumerate(self.class_linears):
            f_u.append(layer(x).unsqueeze(1))
        
        f_u = torch.cat(f_u, dim=1)#[b,6,49,512]
        
        return f_u
class AU_Detect(nn.Module):
    def __init__(self, au_num):
        super(AU_Detect, self).__init__()

        self.aus_feat = nn.ModuleList([nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        ) for i in range(6)])

        self.aus_fc = nn.ModuleList([
            nn.Linear(64, 1)
            for i in range(6)])

    def forward(self, x):
        start = True
        for i in range(len(self.aus_fc)):
            au_feat = self.aus_feat[i](x)

            au_feat_interm = F.avg_pool2d(au_feat, au_feat.size()[2:])
            au_feat_interm = au_feat_interm.view(au_feat_interm.size(0), -1)
            au_output = self.aus_fc[i](au_feat_interm)

            if start:
                aus_output = au_output
                aus_feat = au_feat_interm
                start = False
            else:
                aus_output = torch.cat((aus_output, au_output), 1)
                aus_feat = torch.cat((aus_feat, au_feat_interm), 1)

        return  aus_output
class E(nn.Module):
    def __init__(self, num_classes=6, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(E, self).__init__()
        
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50(pretrained=True)
            self.in_channels = self.backbone.fc.weight.shape[1]#2048
            self.out_channels = self.in_channels // 4#512
            
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        
        self.AU = AUFea(self.out_channels, num_classes)

    def forward(self, x):
        #[b,3,224,224] 
        x = self.backbone(x)
        #[b,49,2048]
        x = self.global_linear(x)
        
        #[b,49,512]
        cl = self.AU(x)#[b,6,49,512]
       
        return cl