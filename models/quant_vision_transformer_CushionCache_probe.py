import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
from probe import probe
from token_probe import norm_probing_not_sorted
from token_select import token_select
import pandas as pd
import os
_logger = logging.getLogger(__name__)
__all__ = ['qt_deit_small_patch16_224']

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class Quantizer():
    def __init__(self, N_bits: int, type: str = "per_tensor", signed: bool = True, symmetric: bool = True):
        super().__init__()
        self.N_bits = N_bits
        self.signed = signed
        self.symmetric = symmetric
        self.q_type = type
        self.minimum_range = 1e-6
        
        if self.N_bits is None:
            return 

        if self.signed:
            self.Qn = - 2 ** (self.N_bits - 1)
            self.Qp = 2 ** (self.N_bits - 1) - 1
            
        else:
            self.Qn = 0
            self.Qp = 2 ** self.N_bits - 1

    def __call__(self, x):  
        return self.forward(x)

    def forward(self, x): 
        if self.N_bits is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 가능 여부 확인
            if self.q_type == 'per_tensor': 
                torch_one = torch.ones(1).to(device)
                return x,torch_one
            if self.q_type == 'per_token': 
                torch_one = torch.ones(x.shape[0])
                return x, torch_one.to(device)

        if self.symmetric:
            if self.q_type == 'per_tensor': 
                max_x = x.abs().max()
            elif self.q_type == 'per_token': #토큰 별 가장 큰 값
                max_x = x.abs().amax(dim=-1, keepdim=True)           
            elif self.q_type == 'per_channel': #채널별 가장 큰 값 
                max_x = x.abs().amax(dim=0, keepdim=True)
            max_x = max_x.clamp_(self.minimum_range)
            scale = max_x / self.Qp
            x = x / scale 
            x = round_pass(x)
            
        else: #Asymmetric
            if self.q_type == 'per_tensor': 
                min_x = x.min().detach()
                max_x = x.max().detach()
            elif self.q_type == 'per_token': 
                min_x = x.min(dim=-1, keepdim=True).detach()
                max_x = x.max(dim=-1, keepdim=True).detach()
            elif self.q_type == 'per_channel': 
                min_x = x.min(dim=0, keepdim=True).detach()
                max_x = x.max(dim=0, keepdim=True).detach()

            range_x = (max_x - min_x).detach().clamp_(min=self.minimum_range)
            scale = range_x / (self.Qp - self.Qn)
            zero_point = torch.round((min_x / scale) - self.Qn)
            x = (x / scale) + zero_point
            x = round_pass(x.clamp_(self.Qn, self.Qp))

        return x, scale


class QuantAct(nn.Module):
    def __init__(self, 
                 N_bits: int, 
                 type: str , 
                 signed: bool = True, 
                 symmetric: bool = True):
        super(QuantAct, self).__init__()
        self.quantizer = Quantizer(N_bits=N_bits, type = type, signed=signed, symmetric=symmetric)

    def forward(self, x):
        q_x, s_qx = self.quantizer(x)
        return q_x, s_qx

class Quantized_Linear(nn.Linear):
    def __init__(self, weight_quantize_module: Quantizer, act_quantize_module: Quantizer, weight_grad_quantize_module: Quantizer, act_grad_quantize_module: Quantizer,
                 in_features, out_features, abits, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        self.weight_grad_quantize_module = weight_grad_quantize_module
        self.act_grad_quantize_module = act_grad_quantize_module
        self.prefix_qmodule = Quantizer(abits, 'per_token')

    def forward(self, input, block_num, epoch, iteration, device_id, prefix_token_num = 0, layer_info = None):
        return _quantize_global.apply(block_num, epoch, iteration, device_id, prefix_token_num, layer_info, input, self.weight, self.bias, self.weight_quantize_module,
                                      self.act_quantize_module, self.weight_grad_quantize_module, self.act_grad_quantize_module, self.prefix_qmodule)
    

class Quantized_Linear(nn.Linear):
    def __init__(self, weight_quantize_module: Quantizer, act_quantize_module: Quantizer, weight_grad_quantize_module: Quantizer, act_grad_quantize_module: Quantizer,
                 in_features, out_features, abits, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        self.act_quantize_module = act_quantize_module
        self.weight_grad_quantize_module = weight_grad_quantize_module
        self.act_grad_quantize_module = act_grad_quantize_module
        self.prefix_qmodule = Quantizer(abits, 'per_token')

    def forward(self, input, block_num, epoch, iteration, device_id, prefix_token_num = 0, layer_info = None):
        return _quantize_global.apply(block_num, epoch, iteration, device_id, prefix_token_num, layer_info, input, self.weight, self.bias, self.weight_quantize_module,
                                      self.act_quantize_module, self.weight_grad_quantize_module, self.act_grad_quantize_module, self.prefix_qmodule)
    
class _quantize_global(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block_num, epoch, iteration, device_id, prefix_token_num, layer_info, x, w, bias=None,
                w_qmodule=None, a_qmodule=None, w_g_qmodule=None, a_g_qmodule=None, prefix_qmodule=None):
        #save for backward
        ctx.block_num = block_num
        ctx.iteration = iteration
        ctx.layer_info = layer_info
        ctx.a_g_qmodule = a_g_qmodule
        ctx.w_g_qmodule = w_g_qmodule 
        ctx.has_bias = bias is not None
        ctx.epoch = epoch
        ctx.device_id=device_id
        B, S, C = x.shape[0], x.shape[1], x.shape[2]
        ctx.reshape_3D_size = B,S,C
        x_2d = x.view(-1, C)
        
        if all(x is None for x in (w_qmodule.N_bits, a_qmodule.N_bits, w_g_qmodule.N_bits, a_g_qmodule.N_bits)):
            ctx.fullprecision = True
            output = torch.matmul(x_2d, w.t())
            ctx.fullprecision = True
            ctx.weight = w.detach()
            ctx.activation = x_2d.detach()
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)

            return output.view(*ctx.reshape_3D_size[:-1], -1)

        else: 
            ctx.fullprecision = False

        if prefix_token_num == 0:
            input_quant, s_input_quant = a_qmodule(x_2d)
            weight_quant, s_weight_quant = w_qmodule(w)
            ctx.weight = (weight_quant, s_weight_quant)
            if w_g_qmodule.N_bits is not None:
                ctx.activation = input_quant, s_input_quant
            else: 
                ctx.activation = x_2d
            output = torch.matmul(input_quant, weight_quant.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            s_o = s_input_quant * s_weight_quant

            if a_qmodule.q_type == 'per_tensor':
                s_o=s_o.expand(output.shape[0], output.shape[1]) 

            elif a_qmodule.q_type == 'per_token':

                if a_qmodule.N_bits is not None:
                    s_o = s_o.expand(-1, output.shape[1])
                else: 
                    s_o = s_o.expand(-1, output.shape[1], -1)


            output = output * s_o
            return output.view(B, S, -1)

        else: #prefix_token exists
            # 3D 상태에서 prefix token 따오기 (B, S, C)
            prefix_token = x[:, :(prefix_token_num + 1)] #[32, 9, 768]
            patch_x = x[:, (prefix_token_num + 1):] #[32, 196, 768]

            #2D
            prefix_token = prefix_token.reshape(-1, C) #[288, 768]
            patch_x = patch_x.reshape(-1, C) #[6272, 768]

            q_prefix_token, s_prefix_token = prefix_qmodule(prefix_token) #prefix_token quantization : per-token 
            q_patch_x, s_patch_x = a_qmodule(patch_x)# patch_x quantization :기존

            q_prefix_token = q_prefix_token.reshape(B,-1, C)
            q_patch_x = q_patch_x.reshape(B,-1,C)

            input_quant = torch.cat((q_prefix_token, q_patch_x), dim=1)

            if a_qmodule.q_type == 'per_token':
                s_prefix_token = s_prefix_token.reshape(B, -1) #[32, 9]
                s_patch_x = s_patch_x.reshape(B, -1) #[32, 196]
                s_input_quant = torch.cat((s_prefix_token, s_patch_x),dim=1)
            elif a_qmodule.q_type == 'per_tensor':
                s_prefix_token = s_prefix_token.reshape(B, -1) #[32, 9]
                s_patch_x = s_patch_x.expand(B, S-prefix_token_num-1) #[32, 196]
                s_input_quant = torch.cat((s_prefix_token, s_patch_x), dim=1) #[32, 205]

            if w_g_qmodule.N_bits is not None:
                ctx.activation = input_quant, s_input_quant
            else: 
                ctx.activation = x_2d

            weight_quant, s_weight_quant = w_qmodule(w) #s_weight size: [[]]
            ctx.weight = (weight_quant, s_weight_quant)

        s_o = s_weight_quant * s_input_quant

        output = torch.matmul(input_quant, weight_quant.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        output = output.view(B, S, -1) #[32, 205, 768])
        s_o = s_o.unsqueeze(-1).expand(-1, -1, output.shape[2]) 
        output = output * s_o

        return output

    @staticmethod
    def backward(ctx, g_3D):
        if ctx.device_id == 0 and ctx.iteration is not None:
            if ctx.iteration == 0 and ctx.layer_info is not None:
                probe(g_3D, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_before', epoch=ctx.epoch, iteration=ctx.iteration)

        g_2D = g_3D.reshape(-1, g_3D.size(-1)) #reshape to 2D
        grad_X = grad_W = grad_bias = None
        B,S,C = ctx.reshape_3D_size
        
        if ctx.fullprecision:
            w = ctx.weight
            x = ctx.activation
            grad_W = torch.matmul(g_2D.t(), x)
            grad_X = torch.matmul(g_2D, w)
            grad_X = grad_X.view(B,S,-1)
            if ctx.has_bias:
                grad_bias = g_2D.sum(dim=0)
            else:
                grad_bias = None
        else:
            q_w, s_w = ctx.weight
            a_g_qmodule = ctx.a_g_qmodule
            w_g_qmodule = ctx.w_g_qmodule
            a_g_2D_quant, a_s_g_2D_quant = a_g_qmodule(g_2D)
            w_g_2D_quant, w_s_g_2D_quant = w_g_qmodule(g_2D)

            if w_g_qmodule.N_bits is not None:
                q_x,s_x = ctx.activation
            else:
                x= ctx.activation

            grad_X = torch.matmul(a_g_2D_quant, q_w)
            s_grad_X = a_s_g_2D_quant * s_w
            grad_X = grad_X * s_grad_X

            if w_g_qmodule.N_bits is None: #not quant
                grad_W = torch.matmul(g_2D.t(), x)
                grad_X = grad_X.view(B,S,-1)
            else: #quant 
                grad_W = torch.matmul(w_g_2D_quant.t(), q_x) #([768, 3072])
                s_grad_W = w_s_g_2D_quant * s_x
                print(grad_W.size())
                print(s_grad_W.size())
                exit()
                grad_W = grad_W * s_grad_W
                grad_X = grad_X.view(B,S,-1)

            if ctx.has_bias:
                grad_bias = g_2D.sum(dim=0)
            else:
                grad_bias = None

        if ctx.device_id == 0 and ctx.iteration is not None:
            if ctx.iteration == 0 and ctx.layer_info is not None:
                probe(grad_X, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
                probe(grad_W, block_num=ctx.block_num, layer=ctx.layer_info + 'W_grad_after', epoch=ctx.epoch, iteration=ctx.iteration)
 
        return None, None, None, None, None, None, grad_X, grad_W, grad_bias, None, None, None, None, None

class Mlp(nn.Module):
    def __init__(
            self,
            block_num,
            a_quant_type, ag_quant_type,
            abits, 
            wbits,
            w_gbits, 
            a_gbits,
            in_features,
            hidden_features=None,
            act_layer=False):
        super().__init__()
        self.block_num = block_num
        out_features = in_features
        self.fc1 = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, a_quant_type), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_tensor'),
                                act_grad_quantize_module=Quantizer(a_gbits, ag_quant_type),
                                in_features=in_features, 
                                out_features=hidden_features, 
                                abits = abits,
                                bias=True
                                )
        self.act = act_layer()
        self.fc2 = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, a_quant_type), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_tensor'),
                                act_grad_quantize_module=Quantizer(a_gbits, ag_quant_type),
                                in_features=hidden_features, 
                                out_features=out_features, 
                                abits = abits, 
                                bias=True
                                )

    def forward(self, x, epoch, iteration, device_id, prefix_token_num):
        x = self.fc1(x, self.block_num, epoch, iteration, device_id, prefix_token_num, layer_info = 'During_MLP(fc1)')
        x = self.act(x)
        x = self.fc2(x, self.block_num, epoch, iteration, device_id, prefix_token_num, layer_info = 'During_MLP(fc2)')
        return x

class Attention(nn.Module):
    def __init__(
            self,
            block_num,
            a_quant_type, ag_quant_type,
            abits, 
            wbits, 
            w_gbits,
            a_gbits,
            dim,
            num_heads,
            qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_num = block_num

        # self.q_norm = nn.LayerNorm(self.head_dim) 
        # self.k_norm = nn.LayerNorm(self.head_dim) 

        self.qkv = Quantized_Linear(
                                    weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                    act_quantize_module=Quantizer(abits, a_quant_type), 
                                    weight_grad_quantize_module=Quantizer(w_gbits, 'per_tensor'),
                                    act_grad_quantize_module=Quantizer(a_gbits, ag_quant_type),
                                    in_features=dim, 
                                    out_features=dim * 3, 
                                    abits = abits,
                                    bias=qkv_bias
                                    )
        self.proj = Quantized_Linear(
                                weight_quantize_module=Quantizer(wbits, 'per_tensor'), 
                                act_quantize_module=Quantizer(abits, a_quant_type), 
                                weight_grad_quantize_module=Quantizer(w_gbits, 'per_tensor'),
                                act_grad_quantize_module=Quantizer(a_gbits, ag_quant_type),
                                in_features=dim, 
                                out_features=dim, 
                                abits = abits,
                                bias=True
        )
        # self.qact3 = QuantAct(abits, 'per_tensor')

    def forward(self, x, epoch, iteration, device_id, prefix_token_num):
        B, N, C = x.shape 
        x = self.qkv(x, self.block_num, epoch, iteration, device_id, prefix_token_num, layer_info = 'qkv') 
        qkv = x.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 
        # q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        # print(attn.size())#32, 12, 205, 205])

        # === Attention Map 시각화 및 저장 ===

        if self.block_num == 11:
            DIR = os.getenv("DIR", "default_model")
            VERSION = os.getenv("VERSION", "default_version")
            
            save_dir = f"{DIR}/prob/cifar100/{VERSION}/AttentionMap2D"
            os.makedirs(save_dir, exist_ok=True)

            attn_subset = attn[:5].detach().cpu()  # (5, num_heads, seq_len, seq_len)
            num_heads = attn.shape[1]
            seq_len = attn.shape[2]

            for i in range(attn_subset.shape[0]):  # i: data index
                for h in range(num_heads):         # h: head index
                    plt.figure(figsize=(6, 5))
                    plt.imshow(attn_subset[i, h], cmap='viridis', interpolation='nearest')
                    plt.colorbar()
                    plt.title(f"Block {self.block_num} | Data {i} | Head {h}")
                    plt.xlabel("Key Token Index")
                    plt.ylabel("Query Token Index")

                    save_path = f"{save_dir}/attn_block{self.block_num}_epoch{epoch}_iter{iteration}_device{device_id}_data{i}_head{h}.png"
                    plt.savefig(save_path)
                    plt.close()
        # =============================================


        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    
        # x, act_scaling_factor = self.qact2(x)

        x = self.proj(x, self.block_num, epoch, iteration, device_id, prefix_token_num, layer_info='Attention_proj') 

        return x

class Q_Block(nn.Module):
    def __init__(self, a_quant_type, ag_quant_type, abits, wbits, w_gbits, a_gbits, block_num, dim, num_heads, mlp_ratio=4., 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.block_num = block_num
        # self.qact1 = QuantAct(abits, 'per_tensor')
        self.attn = Attention(
            block_num,
            a_quant_type, ag_quant_type,
            abits,
            wbits,
            w_gbits,
            a_gbits,
            dim,
            num_heads=num_heads
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.qact2 = QuantAct(abits, 'per_tensor')
        self.mlp = Mlp(
            block_num,
            a_quant_type, ag_quant_type,
            abits, 
            wbits, 
            w_gbits, 
            a_gbits,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

        
    def forward(self, x, is_index_eval, epoch, iteration, device_id, prefix_token_num):
        residual_1 = x
        x = self.norm1(x)
        x = self.attn(x, epoch, iteration, device_id, prefix_token_num)
        x = residual_1 + x

        residual_2 = x 
        x = self.norm2(x)
        x = self.mlp(x, epoch, iteration, device_id, prefix_token_num) 

        x = residual_2 + x

        ###########
        # #x 텐서의 값 분포를 찍어서 현재 디렉토리에 저장 
        if self.block_num ==11 :  
            DIR = os.getenv("DIR", "default_model")  # 환경 변수가 없을 경우 기본값 설정
            VERSION = os.getenv("VERSION", "default_version")   
            # 저장할 디렉토리 생성
            os.makedirs(f"{DIR}/prob/cifar100/{VERSION}/3D_plot", exist_ok=True)
            os.makedirs(f"{DIR}/prob/cifar100/{VERSION}/Token_max", exist_ok=True)
            
            # 배치의 처음 5장만 추출하여 plot
            x_subset = x[:5].abs().cpu()  # (5, Sequence Length, Channel)
            seq_length = x.shape[1]
            channel = x.shape[2]
            
            # 1. 3D 플롯 저장
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            for i in range(5):
                data_index = i  # 데이터 인덱스 수동 지정
                
                # 1. 3D 플롯 저장
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                X = np.arange(seq_length)
                Y = np.arange(channel)
                X, Y = np.meshgrid(X, Y)
                Z = x_subset[i].T  # (Channel, Sequence Length)로 맞춰서 플롯
                ax.plot_surface(X, Y, Z, cmap='viridis')
                
                plt.title(f'3D Plot for Block {self.block_num}, Data {data_index}')
                plt.savefig(f'{DIR}/prob/cifar100/{VERSION}/3D_plot/block_f{self.block_num}_data_f{data_index}.png')
                plt.close()
                
                # 2. 토큰 별 max값 2D 히스토그램으로 plot 저장

                x_max,_ = x_subset[i].max(axis=1)  # (Sequence_length,)
                print(x_max.size())
                
                plt.figure(figsize=(8, 5))
                plt.bar(np.arange(len(x_max)), x_max.numpy(), color='b', alpha=0.7)
                plt.xlabel('Token Index')
                plt.ylabel('Max Value')
                plt.title(f'Block {self.block_num}, Data {data_index}')
                plt.savefig(f'{DIR}/prob/cifar100/{VERSION}/Token_max/block_f{self.block_num}_data_f{data_index}.png')
                plt.close()
            exit()
            #############################################################################

        if device_id == 'cuda:0':
            if iteration == 0: 
                norm_probing_not_sorted(x, block_num=self.block_num, layer='Hidden_State', epoch=epoch, iteration=iteration)
        return x

class CustomSequential(nn.Module):
    def __init__(self, *modules):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, is_index_eval, epoch, iteration, device_id, prefix_token_num):
        for module in self.modules_list:
            x = module(x, is_index_eval, epoch, iteration, device_id, prefix_token_num)
        return x
    

class lowbit_VisionTransformer(VisionTransformer):
    def __init__(self, a_quant_type, ag_quant_type, register_num, num_classes, abits, wbits, w_gbits, a_gbits,
        patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
        norm_layer, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
        super().__init__(patch_size=patch_size, 
                         embed_dim=embed_dim, 
                         depth=depth, 
                         num_heads=num_heads,
                         mlp_ratio=mlp_ratio, 
                         qkv_bias=qkv_bias,
                         norm_layer=norm_layer, 
                         **kwargs)
        num_patches = self.patch_embed.num_patches
        self.prefix_token_num=register_num

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + self.prefix_token_num, self.embed_dim))
        self.blocks = CustomSequential(*[
            Q_Block(a_quant_type, ag_quant_type, abits, wbits, w_gbits, a_gbits, block_num=i, dim=embed_dim,
                    num_heads=num_heads, mlp_ratio=mlp_ratio)
            for i in range(depth)])
        
        # self.head = Quantized_Linear(
        #                 weight_quantize_module=Quantizer(None, 'per_tensor'), 
        #                 act_quantize_module=Quantizer(None, 'per_tensor'), 
        #                 weight_grad_quantize_module=Quantizer(None, 'per_tensor'),
        #                 act_grad_quantize_module=Quantizer(None, 'per_tensor'),
        #                 in_features=embed_dim, 
        #                 out_features=num_classes, 
        #                 abits = abits,
        #                 bias=True
        #                 )
        self.head = nn.Linear(in_features=embed_dim, out_features = num_classes)


        ####################White Patch Prefix#########################
        main_dir=os.environ.get("DIR")
        prefix_token_np = np.load(f"{main_dir}/zz_prefix_patch_token_for_initialization/pretrained_model_raw_patch_token_768.npy")
        prefix_token_tensor = torch.tensor(prefix_token_np, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prefix_token_tensor = prefix_token_tensor.to(device)
        self.reg_token = nn.Parameter(prefix_token_tensor[:1, :self.prefix_token_num, :].clone())  # (1, num_reg, 384)

        ###################################################################
        #self.reg_token = nn.Parameter(torch.zeros(1, self.prefix_token_num, embed_dim))
        #self.reg_token = nn.Parameter(torch.rand(1, self.prefix_token_num, embed_dim))
        
    def forward_features(self, x, is_index_eval, epoch, iteration, device_id):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1) 

        prefix_mode = 'register'
        calibration = ''

        if prefix_mode in ['zero', 'one', 'random', 'background_patch', 'high-frequency']: 
            prefix_token = token_select(B, prefix_mode, calibration).to(x.device) #[256, 3, 224, 224]
            prefix_token = self.patch_embed(prefix_token) #256, 197, 384
            prefix_token = prefix_token[:, :self.prefix_token_num, :]
        
        else : 
            prefix_token = self.reg_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, prefix_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x, is_index_eval, epoch, iteration, device_id, self.prefix_token_num)

        if is_index_eval:
            return x
        
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, is_index_eval=False, epoch=None, iteration=None, device_id=None):
        x = self.forward_features(x, is_index_eval, epoch, iteration, device_id)

        if is_index_eval:
            return x
        x = self.head(x)

        # x = self.head(x, 100, epoch, iteration, device_id, layer_info='Head')

        if device_id == 0 and iteration is not None:
            if iteration == 0:
                probe(x, block_num=100, layer='Head_output', epoch=epoch, iteration=iteration)
        
    
        return x           
        
@register_model
def fullbits_vit_base_patch16_224(pretrained=False, a_quant_type=None, ag_quant_type=None, register_num=0, num_classes=0, **kwargs):
    model = lowbit_VisionTransformer(
        a_quant_type=a_quant_type,
        ag_quant_type=ag_quant_type,
        register_num=register_num, num_classes=num_classes,
        abits=None, wbits=None, w_gbits=None, a_gbits=None,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()
    
    return model


@register_model
def fourbits_vit_base_patch16_224(pretrained=False, a_quant_type=None, ag_quant_type=None, register_num=0, num_classes=0, **kwargs):
    model = lowbit_VisionTransformer(
        a_quant_type=a_quant_type,
        ag_quant_type=ag_quant_type,
        register_num=register_num, num_classes=num_classes,
        abits=4, wbits=4, w_gbits=None, a_gbits=4,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()
    
    return model

@register_model
def full_fourbits_vit_base_patch16_224(pretrained=False,  a_quant_type=None, ag_quant_type=None, register_num=0, num_classes=0, **kwargs):
    model = lowbit_VisionTransformer(
        a_quant_type=a_quant_type,
        ag_quant_type=ag_quant_type,
        register_num=register_num, num_classes=num_classes,
        abits=4, wbits=4, w_gbits=4, a_gbits=4,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()
    
    return model

@register_model
def full_eightbits_vit_base_patch16_224(pretrained=False,  a_quant_type=None, ag_quant_type=None, register_num=0, num_classes=0, **kwargs):
    model = lowbit_VisionTransformer(
        a_quant_type=a_quant_type,
        ag_quant_type=ag_quant_type,
        register_num=register_num, num_classes=num_classes,
        abits=8, wbits=8, w_gbits=8, a_gbits=8,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()
    
    return model

@register_model
def eightbits_vit_base_patch16_224(pretrained=False, a_quant_type=None, ag_quant_type=None, register_num=0, num_classes=0, **kwargs):
    model = lowbit_VisionTransformer(
        a_quant_type=a_quant_type,
        ag_quant_type=ag_quant_type,
        register_num=register_num, num_classes=num_classes,
        abits=8, wbits=8, w_gbits=None, a_gbits=8,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()
    
    return model

@register_model
def fourbits_vit_small_patch16_224(pretrained=False,  a_quant_type=None, ag_quant_type=None, register_num=0, num_classes=0, **kwargs):
    model = lowbit_VisionTransformer(
        abits=4, wbits=4, w_gbits=None, a_gbits=4,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()

    
    return model

@register_model
def eightbits_deit_small_patch16_224(pretrained=False, **kwargs):
    model = lowbit_VisionTransformer(
        abits=8, wbits=8, w_gbits=None, a_gbits=8,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()
    
    if finetune and os.path.exists(finetune):
        print(f"Loading checkpoint from {finetune}")
        state_dict = torch.load(finetune, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    return model

