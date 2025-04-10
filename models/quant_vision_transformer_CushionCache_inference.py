import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


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
            return x, 1

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
        ctx.reshape_3D_size = x.size() # x as 3D 
        ctx.has_bias = bias is not None
        ctx.epoch = epoch
        ctx.device_id=device_id

        if device_id == 0 and iteration is not None:
            if iteration == 0 and layer_info is not None:
                probe(w, block_num=block_num, layer=layer_info + 'weight', epoch=epoch, iteration=iteration)

        x_2d = x.view(-1, x.size(-1)).to(torch.float32)  # [batch_size * seq_len, feature_dim]
        w = w.to(torch.float32)

        ctx.fullprecision_x = x_2d.detach()

        #if full precision
        if all(x is None for x in (w_qmodule.N_bits, a_qmodule.N_bits, w_g_qmodule.N_bits, a_g_qmodule.N_bits)):
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
            if a_qmodule.q_type == 'per_tensor' and w_qmodule.q_type == 'per_tensor':
                ctx.weight = (weight_quant.detach(), s_weight_quant)
                s_o = s_weight_quant * s_input_quant
                ctx.activation = input_quant, s_input_quant
            else: #per_token
                ctx.weight = (weight_quant.detach(), s_weight_quant.detach())
                s_o = s_input_quant * s_weight_quant
                ctx.activation = input_quant, s_input_quant
            output = torch.matmul(input_quant, weight_quant.t())

            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            output = output *s_o
            return output.view(*ctx.reshape_3D_size[:-1], -1)

        prefix_token = x_2d[:(prefix_token_num + 1) * x.size(0)]
        patch_x = x_2d[(prefix_token_num + 1) * x.size(0):]

        q_prefix_token, s_prefix_token = prefix_qmodule(prefix_token) #prefix_token quantization : per-token 
        q_patch_x, s_patch_x = a_qmodule(patch_x)# patch_x quantization :기존

        if a_qmodule.q_type == 'per_tensor':
            input_quant = torch.cat((q_prefix_token, q_patch_x), dim=0)
            s_input_quant = torch.cat((s_prefix_token, s_patch_x.expand(q_patch_x.shape[0]).unsqueeze(-1)), dim=0)
        else: #per_token
            input_quant = torch.cat((q_prefix_token, q_patch_x), dim=0)
            s_input_quant = torch.cat((s_prefix_token, s_patch_x),dim=0)

        if w_g_qmodule.N_bits is not None:
            ctx.activation = input_quant, s_input_quant
        weight_quant, s_weight_quant = w_qmodule(w)
        if a_qmodule.q_type == 'per_tensor' and w_qmodule.q_type == 'per_tensor':
            ctx.weight = (weight_quant.detach(), s_weight_quant)
            s_o = s_weight_quant * s_input_quant
        else:
            ctx.weight = (weight_quant.detach(), s_weight_quant.detach())
            s_o = s_input_quant * s_weight_quant


        output = torch.matmul(input_quant, weight_quant.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        s_o = s_weight_quant * s_input_quant
        output = output * s_o

        return output.view(*ctx.reshape_3D_size[:-1], -1)


    @staticmethod
    def backward(ctx, g_3D):
        if ctx.device_id == 0 and ctx.iteration is not None:
            if ctx.iteration == 0 and ctx.layer_info is not None:
                probe(g_3D, block_num=ctx.block_num, layer=ctx.layer_info + 'X_grad_before', epoch=ctx.epoch, iteration=ctx.iteration)

        g_2D = g_3D.reshape(-1, g_3D.size(-1)) #reshape to 2D
        grad_X = grad_W = grad_bias = None
        if ctx.fullprecision:
            w = ctx.weight
            x = ctx.activation
            reshape_3D = ctx.reshape_3D_size
            grad_W = torch.matmul(g_2D.t(), x)
            grad_X = torch.matmul(g_2D, w)
            if ctx.layer_info != 'Head':
                grad_X = grad_X.view(reshape_3D[0],reshape_3D[1],-1)
            if ctx.has_bias:
                grad_bias = g_2D.sum(dim=0)
            else:
                grad_bias = None
        else:
            q_w, s_w = ctx.weight
            a_g_qmodule = ctx.a_g_qmodule
            w_g_qmodule = ctx.w_g_qmodule
            reshape_3D = ctx.reshape_3D_size
            a_g_2D_quant, a_s_g_2D_quant = a_g_qmodule(g_2D)
            w_g_2D_quant, w_s_g_2D_quant = w_g_qmodule(g_2D)

            if w_g_qmodule.N_bits is not None:
                q_x,s_x = ctx.activation
            else:
                fullprecision_x = ctx.fullprecision_x

            grad_X = torch.matmul(a_g_2D_quant, q_w)

            #Activation gradient quant / not quant
            if a_g_qmodule.q_type == 'per_tensor':
                s_grad_X = a_s_g_2D_quant * s_w
            else: #per_token
                s_grad_X = a_s_g_2D_quant * s_w

            grad_X = grad_X * s_grad_X


            #Weight gradient quant / not quant
            if ctx.layer_info == 'Head':
                grad_W = torch.matmul(g_2D.t(), fullprecision_x)
            elif w_g_qmodule.N_bits is None: #not quant
                grad_W = torch.matmul(g_2D.t(), fullprecision_x)
                grad_X = grad_X.view(reshape_3D[0],reshape_3D[1],-1)
            else: #quant 
                grad_W = torch.matmul(w_g_2D_quant.t(), q_x) #([768, 3072])
                s_grad_W = w_s_g_2D_quant * s_x
                grad_W = grad_W * s_grad_W
                grad_X = grad_X.view(reshape_3D[0],reshape_3D[1],-1)

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
        self.last_attn_map = None

    def forward(self, x, epoch, iteration, device_id, prefix_token_num):
        B, N, C = x.shape 
        x = self.qkv(x, self.block_num, epoch, iteration, device_id, prefix_token_num, layer_info = 'qkv') 
        qkv = x.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 
        # q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        self.last_attn_map = attn 

        #######################################################
        # save_dir = f"/home/shkim/SSF_org/SSF/inference_output/QKAttentionMap"
        # os.makedirs(save_dir, exist_ok=True)

        # with torch.no_grad():
        #     for b in range(min(1, B)):  # 첫 배치만 저장 (필요시 여러 개)
        #         for h in range(H):  # 모든 헤드에 대해 저장
        #             attn_map = attn[b, h].cpu().numpy()  # shape: (N, N)
        #             plt.figure(figsize=(6, 6))
        #             plt.imshow(attn_map, cmap='viridis')
        #             plt.title(f"Attention Map - Epoch {epoch}, Iter {iteration}, Head {h}")
        #             plt.colorbar()
        #             fname = os.path.join(save_dir, f"attn_ep{epoch}_it{iteration}_b{b}_h{h}.png")
        #             plt.savefig(fname)
        #             plt.close()
        #######################################################

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
        self.hidden_state = None

        
    def forward(self, x, is_index_eval, epoch, iteration, device_id, prefix_token_num):
        residual_1 = x
        x = self.norm1(x)
        x = self.attn(x, epoch, iteration, device_id, prefix_token_num)
        x = residual_1 + x

        residual_2 = x 
        x = self.norm2(x)
        x = self.mlp(x, epoch, iteration, device_id, prefix_token_num) 

        x = residual_2 + x
        self.hidden_state = x

        # x = torch.clamp(x, max=10)
        # x = torch.clamp(x, min=-10)

        # #x 텐서의 값 분포를 찍어서 현재 디렉토리에 저장 
        if self.block_num ==11 : 
            import torch
            import matplotlib.pyplot as plt    

            plt.figure(figsize=(6, 4))
       
            plt.hist(x.detach().cpu().numpy().flatten(), bins=50, alpha=0.75, color='b')
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.title(f"Tensor Value Distribution, {x.max()}")

            # 파일로 저장
            os.makedirs('./Hidden_state_output', exist_ok=True)
            plt.savefig("./Hidden_state_output/Attention.png")
            plt.close()

            # x = torch.clamp(x, max=1500)
            # x = torch.clamp(x, max=-1500)

            # plt.figure(figsize=(6, 4))
            # plt.hist(x.detach().cpu().numpy().flatten(), bins=50, alpha=0.75, color='b')
            # plt.xlabel("Value")
            # plt.ylabel("Frequency")
            # plt.title("Tensor Value Distribution")

            # # 파일로 저장
            # plt.savefig("/home/shkim/QT/deit/output/finetune/test/x_after_clamping.png")
            # plt.close()
            # exit()
        #################################################################
        # if self.block_num == 11: 
        # x = torch.where(x > 10, torch.tensor(0, dtype=x.dtype), x)
        # x = torch.where(x < -10, torch.tensor(0, dtype=x.dtype), x)
        # print(torch.max(x), torch.min(x))

        #################################################################
        
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

