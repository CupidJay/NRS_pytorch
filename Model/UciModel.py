'''
Created on 2018.07.05

@author: caoyh
'''
import torch
import torch.nn as nn
import numpy as np
from DataLoader import perm_mask
from torch.nn.parameter import Parameter

__all__ = ['UciNRSNet', 'UciFCNet']

class BasicNRSLayer(nn.Module):
    def __init__(self, params):
        super(BasicNRSLayer, self).__init__()
        self.dd, self.dH, self.dW, self.nMul, self.nPer = params
        mask = perm_mask(self.dd, self.dH, self.dW, self.nMul)
        print(params, mask)
        self.register_buffer('mask', torch.from_numpy(mask))

        self.nrs = nn.Sequential(
            #DGConv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=self.dH,
            #          padding=0),
            nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=3,
                      padding=0, groups=self.dd * self.nMul // self.nPer),
            nn.BatchNorm2d(self.dd * self.nMul),
            nn.ReLU()
        )

        if self.dH >3:
            self.nrs2 = nn.Sequential(
                # DGConv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=self.dH,
                #          padding=0),
                nn.Conv2d(in_channels=self.dd * self.nMul, out_channels=self.dd * self.nMul, kernel_size=3,
                          padding=0, groups=self.dd * self.nMul // self.nPer),
                nn.BatchNorm2d(self.dd * self.nMul),
                nn.ReLU()
            )
        else:
            self.nrs2 = nn.Sequential()


    def forward(self, x):
        #older version
        #x = torch.stack([xi[self.mask] for xi in torch.unbind(x, dim=0)], dim=0)

        #faster version
        now_ind = self.mask.unsqueeze(0).repeat([x.size(0), 1])
        x = x.repeat([1, self.nMul])
        x = torch.gather(x, 1, now_ind)


        # print(self.dd, self.nMul, self.dH, self.dW)
        x = x.view(x.size(0), self.dd * self.nMul, self.dH, self.dW)
        x = self.nrs(x)
        x = self.nrs2(x)
        res = x.view(x.size(0),-1)

        return res

    def get_out_features(self):
        return self.dd*self.nMul

#the valinna NFL, NFL module+dense classification head
class UciNRSNet(nn.Module): # a simple CNN with only 1 active depthwise conv. layer and 2 FC layers. BN and ReLU are both used
    def __init__(self, cfg, params=None):
        super(UciNRSNet,self).__init__()
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.dW = cfg.MODEL.DW
        self.nMul = cfg.MODEL.N_MUL
        self.nPer = cfg.MODEL.N_PER_GROUP
        self.sigmoid = cfg.MODEL.SIGMOID
        nFC = cfg.MODEL.FC.N_FC
        bFC = cfg.MODEL.FC.B_FC
        nClass = cfg.DATASETS.CLASS

        
        basic_layer = BasicNRSLayer

        if params is None:
            params = [self.dd, self.dH, self.dW, self.nMul, self.nPer]

        self.nrs_layer = basic_layer(params)

        out_dim = self.nrs_layer.get_out_features()

        if bFC==True:
            self.dense = nn.Sequential(
                nn.Linear(out_dim, nFC),
                nn.BatchNorm1d(nFC),
                nn.ReLU(),
                nn.Linear(nFC, nFC),
                nn.BatchNorm1d(nFC),
                nn.ReLU(),
                nn.Linear(nFC, nClass)
            )
        else:
            self.dense = nn.Sequential(
                nn.Linear(out_dim, nClass)
            )

    def forward(self,x):
        x = self.nrs_layer(x)
        x = x.view(x.size(0), -1)
        out = self.dense(x)

        if self.sigmoid:
            out = torch.sigmoid(out)

        return out

class UciFCNet(nn.Module): # a simple CNN with only 1 active depthwise conv. layer and 2 FC layers. BN and ReLU are both used
    def __init__(self, cfg):
        super(UciFCNet,self).__init__()
        self.dd = cfg.MODEL.DD
        self.dH = cfg.MODEL.DH
        self.nMul = cfg.MODEL.N_MUL
        nFC = cfg.MODEL.FC.N_FC
        bFC = cfg.MODEL.FC.B_FC
        nClass = cfg.DATASETS.CLASS
        nPerGroup = cfg.MODEL.N_PER_GROUP

        self.sigmoid = cfg.MODEL.SIGMOID

        self.dense = nn.Sequential(
            #nn.Dropout(p=0.8),
            nn.Linear(self.dd, nFC),
            nn.BatchNorm1d(nFC),
            nn.ReLU(),
            nn.Linear(nFC, nFC),
            nn.BatchNorm1d(nFC),
            nn.ReLU(),
            nn.Linear(nFC, nFC),
            nn.BatchNorm1d(nFC),
            nn.ReLU(),
            nn.Linear(nFC, nClass)
            )

    def forward(self,x):
        x = x.view(x.size(0), -1)
        out = self.dense(x)

        if self.sigmoid:
            out = torch.sigmoid(out)

        return out
