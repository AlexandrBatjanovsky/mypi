#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:12:51 2023

@author: alexandersn
"""

import numpy as np
import torch
from torch import nn
from typing import Union as U, Optional as O

def get_nlin(nlin: str):
    if nlin == 'relu':
        ret = nn.ReLU(inplace=True)
    elif nlin == 'elu':
        ret = nn.ELU(inplace=True)
    elif nlin == 'prelu':
        ret = nn.PReLU()
    elif nlin == 'lrelu':
        ret = nn.LeakyReLU(inplace=True)
    else:
        raise NotImplementedError
    return ret


class ConvBN(nn.Sequential):

    def __init__(self, inp: int, out: int, ks: int = 3, nlin: O[str] = 'relu', bias=False):
        super().__init__()
        pad = ks // 2
        body = [
            nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=ks, bias=bias, padding=pad),
            nn.BatchNorm2d(out),
        ]
        if nlin is not None:
            body.append(get_nlin(nlin))
        super().__init__(*body)


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation, ks=3, nlin: str = 'relu'):
        pad = (ks + (ks-1)*(dilation-1) - 1) // 2
        modules = [
            nn.Conv2d(in_channels, out_channels, ks, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            get_nlin(nlin)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPMultiConvBlock(nn.Module):

    def __init__(self, inp_dim: int, out_dim: int, ks=3, ks_prj=1, dilation=(2, ), num_conv=1, nlin: str = 'relu'):
        super().__init__()
        self.inp = inp_dim
        self.out = out_dim
        num_aspp = len(dilation)
        self.proj_inp = ConvBN(inp_dim, out_dim, ks=ks_prj, nlin=nlin)
        self.proj_out = nn.ModuleList([ConvBN(num_aspp * out_dim, out_dim, ks=ks_prj, nlin=nlin) for x in range(num_conv)])
        self.aspp = nn.ModuleList()
        for ci in range(num_conv):
            aspp = nn.ModuleList([ASPPConv(out_dim, out_dim, x, ks=ks, nlin=nlin) for x in dilation])
            self.aspp.append(aspp)

    def forward(self, x):
        x = self.proj_inp(x)
        for aspp_conv, prj in zip(self.aspp, self.proj_out):
            x = torch.cat([m(x) for m in aspp_conv], dim=1)
            x = prj(x)
        return x


class CSEBlock(nn.Module):

    def __init__(self, ch, re=16, nlin: str = 'relu'):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch // re, 1),
                                 get_nlin(nlin),
                                 nn.Conv2d(ch // re, ch, 1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x)


class SCSEBlock(nn.Module):
    def __init__(self, ch, re=16, nlin: str = 'relu'):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 get_nlin(nlin),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


def get_attention(ch: int, atype: str = 'cse', nlin: str = 'relu'):
    if atype == 'cse':
        return CSEBlock(ch=ch, nlin=nlin)
    elif atype == 'scse':
        return SCSEBlock(ch=ch, nlin=nlin)
    elif atype is None:
        return nn.Identity()
    else:
        raise NotImplementedError


class BottleneckASPPSEBlock(nn.Module):

    def __init__(self, inp_dim: int, emb_dim: int,
                 dilation: U[tuple, list] = (2, ), num_conv=1,
                 ks: int = 3, nlin: str = 'relu', attention_type=None):
        super().__init__()
        self.conv_prj_inp = ConvBN(inp_dim, emb_dim, ks=1, nlin=nlin)
        # self.conv_spatial = ConvBN(emb_dim, emb_dim, ks=ks, nlin=nlin)
        self.conv_spatial = ASPPMultiConvBlock(emb_dim, emb_dim, ks=ks, nlin=nlin,
                                               num_conv=num_conv, dilation=dilation)
        self.conv_prj_out = ConvBN(emb_dim, inp_dim, ks=1, nlin=None)
        self.attention = get_attention(inp_dim, attention_type, nlin='relu')
        self.out_activation = get_nlin(nlin)

    def forward(self, x):
        inp = x
        x = self.conv_prj_inp(x)
        x = self.conv_spatial(x)
        x = self.conv_prj_out(x)
        x = self.attention(x)
        x += inp
        x = self.out_activation(x)
        return x


class MultiBottleneckSEBlock(nn.Module):

    def __init__(self, inp_dim: int, emb_dim: int,
                 dilation: U[tuple, list] = (2, ), num_conv=1,
                 ks: int = 3, nlin: str = 'relu', attention=None,
                 num_blocks=1):
        super().__init__()
        inps = [inp_dim] * num_blocks
        embs = [emb_dim] * num_blocks
        # dims = [inp_dim] + [emb_dim] * num_blocks
        body = [BottleneckASPPSEBlock(num_inp, num_emb, dilation=dilation, num_conv=num_conv,
                                      ks=ks, nlin=nlin, attention_type=attention)
                for num_inp, num_emb in zip(inps, embs)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class ASPPResNetSE(nn.Module):

    def __init__(self, inp: int, out: int, nlin: str = 'elu', attention='cse', num_stages=4):
        super().__init__()
        self.num_out = out
        stages_params = [
            {'inp': inp, 'out': 64,  'emb': 32, 'dil_conv': (2, 4,  8),  'dil_res': (1, 2, 4), 'numc': 2, 'numr': 2},
            {'inp': 64,  'out': 96,  'emb': 64, 'dil_conv': (4, 8,  16), 'dil_res': (1, 2, 4), 'numc': 2, 'numr': 3},
            {'inp': 96,  'out': 96,  'emb': 64, 'dil_conv': (8, 16, 32), 'dil_res': (1, 2, 4), 'numc': 2, 'numr': 3},
            {'inp': 96,  'out': 128, 'emb': 64, 'dil_conv': (8, 16, 32), 'dil_res': (1, 2, 4), 'numc': 2, 'numr': 3},
            {'inp': 128, 'out': 160, 'emb': 96, 'dil_conv': (8, 16, 32, 48), 'dil_res': (1, 2, 4, 8), 'numc': 2, 'numr': 3},
            {'inp': 160, 'out': 160, 'emb': 96, 'dil_conv': (8, 16, 32, 48), 'dil_res': (1, 2, 4, 8), 'numc': 2,
             'numr': 3},
            {'inp': 160, 'out': 160, 'emb': 96, 'dil_conv': (8, 16, 32, 48), 'dil_res': (1, 2, 4, 8), 'numc': 2,
             'numr': 3},
        ]
        #
        body = [nn.BatchNorm2d(stages_params[0]['inp'])]
        for stage in range(num_stages):
            sp = stages_params[stage]
            stage_conv = ASPPMultiConvBlock(sp['inp'], sp['out'], dilation=sp['dil_conv'],
                                            ks_prj=3, nlin=nlin, num_conv=sp['numc'])
            stage_res = MultiBottleneckSEBlock(sp['out'], sp['emb'], dilation=sp['dil_res'],
                                               nlin=nlin, num_blocks=sp['numr'], attention=attention)
            body.extend([stage_conv, stage_res])
        self.body = nn.Sequential(*body)
        num_ = stages_params[num_stages - 1]['out']
        
        self.out_block = nn.Conv2d(num_, out, kernel_size=1)
        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        print(x.shape)
        x = self.body(x)
        x = self.out_block(x)
        if self.num_out == 1:
            x = torch.squeeze(x, dim=1)
        return x

def count_model_parameters(model:nn.Module):
    params_total = np.sum([p.numel() for p in model.parameters()])
    params_grad = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    return {
        'params_total': params_total,
        'params_grad': params_grad
    }