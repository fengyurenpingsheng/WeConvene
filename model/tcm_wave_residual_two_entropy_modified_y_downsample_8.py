from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.layers import GDN


import torch.nn.functional as F
from torch import Tensor
#from torch_wavelets import DWT_2D, IDWT_2D

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath

import pywt
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable, gradcheck

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64




class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        #self.filters = self.filters.to(dtype=torch.float16)
        self.filters = self.filters.to(dtype=torch.float32)
    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        #self.w_ll = self.w_ll.to(dtype=torch.float16)
        #self.w_lh = self.w_lh.to(dtype=torch.float16)
        #self.w_hl = self.w_hl.to(dtype=torch.float16)
        #self.w_hh = self.w_hh.to(dtype=torch.float16)

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


class ResidualBlockWithStride_wave(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, wavelet='haar'):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn_low = GDN(out_ch)
        self.gdn_high = GDN(3*out_ch)
        self.dwt = DWT_2D(wave=wavelet)
        self.idwt = IDWT_2D(wave=wavelet)

        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.high_freq_conv = conv3x3(3*out_ch, 3*out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)

        # DWT
        dwt_output = self.dwt(out)

        # Separate low-frequency and high-frequency components
        low_freq = dwt_output[:, :out.size(1), :, :]
        high_freq = dwt_output[:, out.size(1):, :, :]

        # Process low-frequency and high-frequency components separately
        low_freq_processed = self.low_freq_conv(low_freq)
        low_freq_processed = self.gdn_low(low_freq_processed)
        high_freq_processed = self.high_freq_conv(high_freq)
        high_freq_processed = self.gdn_high(high_freq_processed)

        # Reassemble the processed components 
        dwt_processed = torch.cat([low_freq_processed, high_freq_processed], dim=1)

        # IDWT
        output = self.idwt(dwt_processed)


        if self.skip is not None:
            identity = self.skip(x)

        output += identity
        return output

class ResidualBlockUpsample_wave(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """
    
    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2, wavelet='haar'):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        #self.conv = conv3x3(out_ch, out_ch)
        self.igdn_low = GDN(out_ch, inverse=True)
        self.igdn_high = GDN(3*out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)
        self.dwt = DWT_2D(wave=wavelet)
        self.idwt = IDWT_2D(wave=wavelet)

        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.high_freq_conv = conv3x3(3*out_ch, 3*out_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        #out = self.conv(out)
        #out = self.igdn(out)
        # DWT 
        dwt_output = self.dwt(out)

        # Separate low-frequency and high-frequency components
        low_freq = dwt_output[:, :out.size(1), :, :]
        high_freq = dwt_output[:, out.size(1):, :, :]

        # Process low-frequency and high-frequency components separately
        low_freq_processed = self.low_freq_conv(low_freq)
        low_freq_processed = self.igdn_low(low_freq_processed)
        high_freq_processed = self.high_freq_conv(high_freq)
        high_freq_processed = self.igdn_high(high_freq_processed)

        # Reassemble the processed components 
        dwt_processed = torch.cat([low_freq_processed, high_freq_processed], dim=1)

        # IDWT
        output = self.idwt(dwt_processed)

        identity = self.upsample(x)
        output += identity
        return output


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        #self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        #self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.block_1 = ResidualBlock(input_dim, output_dim)
        self.block_2 = ResidualBlock(input_dim, input_dim)
        self.window_size = window_size


    def forward(self, x):
        #resize = False
        #if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            #padding_row = (self.window_size - x.size(-2)) // 2
            #padding_col = (self.window_size - x.size(-1)) // 2
            #x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        #trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(x)
        trans_x =  self.block_2(trans_x)
        #trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        #if resize:
            #x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x

class TCM_residual_wave_two_entropy_modified_y_downsample_8(CompressionModel):
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0, N=128,  M=320, num_slices=5, max_support_slices=5, wavelet='haar', **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.dwt = DWT_2D(wave=wavelet)
        self.idwt = IDWT_2D(wave=wavelet)
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0


        self.m_down1 = [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)] +\
                      [ResidualBlockWithStride_wave(N, N, stride=2)]
        begin += config[0]
        self.m_down2 =  [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)] + \
                      [ResidualBlockWithStride_wave(N, N, stride=2)]
        begin += config[1]
        self.m_down3 =  [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)]  + \
                      [conv3x3(N, M, stride=2)]

        begin += config[2]
        self.m_up1 =  [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)] +\
                      [ResidualBlockUpsample_wave(N, N, 2)]
        self.m_up2 =  [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)] +\
                      [ResidualBlockUpsample_wave(N, N, 2)]
        self.m_up3 =  [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)] +\
                      [subpel_conv3x3(N, 3, 2)]
        
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, N, 1)] + self.m_down1 + self.m_down2 + self.m_down3)
        

        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, N, 1)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down1 = [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)] +\
                      [conv3x3(N, 192, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride_wave(320*4, N, 2)] + \
            self.ha_down1
        )

        self.hs_up1 =  [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)] + \
                      [subpel_conv3x3(N, 320, 2)]

        self.h_mean_s = nn.Sequential(
            *[ResidualBlockUpsample_wave(192, N, 2)] + \
            self.hs_up1
        )

        self.hs_up2 =  [ResidualBlock(dim, dim)] + [ResidualBlock(dim, dim)] +[ResidualBlock(dim, dim)] +\
                      [subpel_conv3x3(N, 320, 2)]


        self.h_scale_s = nn.Sequential(
            *[ResidualBlockUpsample_wave(192, N, 2)] + \
            self.hs_up2
        )

        self.atten_mean_imag = nn.ModuleList(
            nn.Sequential(
                SWAtten((320+320 + (320*3//self.num_slices)*min(i, 5)), (320 + (320*3//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )

        self.atten_mean_real = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )

        self.atten_scale_imag = nn.ModuleList(
            nn.Sequential(
                SWAtten((320+320 + (320*3//self.num_slices)*min(i, 5)), (320 + (320*3//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        self.atten_scale_real = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )

        self.cc_mean_transforms_real = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.cc_mean_transforms_imag = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320*3//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320*3//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        #mu = self.cc_mean_transforms_imag[slice_index](mean_support)
        self.cc_scale_transforms_real = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )

        self.cc_scale_transforms_imag = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320*3//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320*3//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
            )

        self.lrp_transforms_real = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms_imag = nn.ModuleList(
            nn.Sequential(
                conv(320+ 320 + (320*3//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320*3//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional_real = GaussianConditional(None)
        self.gaussian_conditional_imag = GaussianConditional(None)
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated_real = self.gaussian_conditional_real.update_scale_table(scale_table, force=force)
        updated_imag = self.gaussian_conditional_imag.update_scale_table(scale_table, force=force)
        updated_real |= super().update(force=force)
        updated_imag |= super().update(force=force)
        return updated_real, updated_imag
    
    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        y_output = self.dwt(y)

        #low_frequency and high_frequency
        low_freq = y_output[:, :320, :, :]
        high_freq = y_output[:, 320:, :, :]

        y_input = torch.cat([low_freq,high_freq], dim=1)
        z = self.h_a(y_input)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_real_slices = low_freq.chunk(self.num_slices, 1)
        y_real_hat_slices = []
        y_real_likelihood = []
        mu_real_list = []
        scale_real_list = []

        y_imag_slices = high_freq.chunk(self.num_slices, 1)
        y_imag_hat_slices = []
        y_imag_likelihood = []
        mu_imag_list = []
        scale_imag_list = []

        for slice_index, y_slice in enumerate(y_real_slices):
            support_slices = (y_real_hat_slices if self.max_support_slices < 0 else y_real_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean_real[slice_index](mean_support)
            mu = self.cc_mean_transforms_real[slice_index](mean_support)
            mu = mu[:, :, :, :]
            mu_real_list.append(mu)
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale_real[slice_index](scale_support)
            scale = self.cc_scale_transforms_real[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_real_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional_real(y_slice, scale, mu)
            y_real_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu
            # if self.training:
            #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
            # else:
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms_real[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_real_hat_slices.append(y_hat_slice)

        y_real_hat = torch.cat(y_real_hat_slices, dim=1)
        means_real = torch.cat(mu_real_list, dim=1)
        scales_real = torch.cat(scale_real_list, dim=1)
        y_real_likelihoods = torch.cat(y_real_likelihood, dim=1)

        for slice_index, y_slice in enumerate(y_imag_slices):
            support_slices = (y_imag_hat_slices if self.max_support_slices < 0 else y_imag_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means]+ [y_real_hat] + support_slices, dim=1)
            mean_support = self.atten_mean_imag[slice_index](mean_support)
            mu = self.cc_mean_transforms_imag[slice_index](mean_support)
            mu = mu[:, :, :, :]
            mu_imag_list.append(mu)
            scale_support = torch.cat([latent_scales] + [y_real_hat]+ support_slices, dim=1)
            scale_support = self.atten_scale_imag[slice_index](scale_support)
            scale = self.cc_scale_transforms_imag[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_imag_list.append(scale)
            _, y_slice_likelihood = self.gaussian_conditional_imag(y_slice, scale, mu)
            y_imag_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu
            # if self.training:
            #     lrp_support = torch.cat([mean_support + torch.randn(mean_support.size()).cuda().mul(scale_support), y_hat_slice], dim=1)
            # else:
            lrp_support = torch.cat([y_real_hat] +[mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms_imag[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_imag_hat_slices.append(y_hat_slice)

        y_imag_hat = torch.cat(y_imag_hat_slices, dim=1)
        means_imag = torch.cat(mu_imag_list, dim=1)
        scales_imag = torch.cat(scale_imag_list, dim=1)
        y_imag_likelihoods = torch.cat(y_imag_likelihood, dim=1)

        #y_hat = torch.cat((y_real_hat, y_imag_hat), 1)
        dwt_processed = torch.cat([y_real_hat, y_imag_hat], dim=1)

        # IDWT 小波逆变换
        y_hat = self.idwt(dwt_processed)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_real": y_real_likelihoods,"y_imag": y_imag_likelihoods, "z": z_likelihoods},
            "para":{"means_real": means_real, "scales_real":scales_real, "y":y}
        }

    def load_state_dict_real(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional_real,
            "gaussian_conditional_real",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict_real(state_dict)

    def load_state_dict_imag(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional_imag,
            "gaussian_conditional_imag",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict_imag(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a_real.0.weight"].size(0)
        M = state_dict["g_a_real.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(N, M)
        net.load_state_dict_real(state_dict)
        net.load_state_dict_imag(state_dict)
        return net

    def compress(self, x):

        #x = torch.fft.fft2(x, dim=(-2, -1))

        y = self.g_a(x)
        y_shape = y.shape[2:]

        y_output = self.dwt(y)

        low_freq = y_output[:, :320, :, :]
        high_freq = y_output[:, 320:, :, :]

        y_input = torch.cat([low_freq, high_freq], dim=1)
        #z = self.h_a(y_input)

        z = self.h_a(y_input)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        #_, z_likelihoods = self.entropy_bottleneck(z)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)


        y_real_slices = low_freq.chunk(self.num_slices, 1)
        y_real_hat_slices = []
        y_real_likelihood = []
        y_mu_real_list = []
        y_scale_real_list = []

        y_imag_slices = high_freq.chunk(self.num_slices, 1)
        y_imag_hat_slices = []
        y_imag_likelihood = []
        y_mu_imag_list = []
        y_scale_imag_list = []

        cdf_real = self.gaussian_conditional_real.quantized_cdf.tolist()
        cdf_real_lengths = self.gaussian_conditional_real.cdf_length.reshape(-1).int().tolist()
        offsets_real = self.gaussian_conditional_real.offset.reshape(-1).int().tolist()

        encoder_real = BufferedRansEncoder()
        symbols_real_list = []
        indexes_real_list = []
        y_real_strings = []

        for slice_index, y_slice in enumerate(y_real_slices):
            support_slices = (y_real_hat_slices if self.max_support_slices < 0 else y_real_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean_real[slice_index](mean_support)
            mu = self.cc_mean_transforms_real[slice_index](mean_support)
            mu = mu[:, :, :, :]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale_real[slice_index](scale_support)
            scale = self.cc_scale_transforms_real[slice_index](scale_support)
            scale = scale[:, :, :, :]

            index = self.gaussian_conditional_real.build_indexes(scale)
            y_q_slice = self.gaussian_conditional_real.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_real_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_real_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms_real[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_real_hat_slices.append(y_hat_slice)
            y_scale_real_list.append(scale)
            y_mu_real_list.append(mu)

        encoder_real.encode_with_indexes(symbols_real_list, indexes_real_list, cdf_real, cdf_real_lengths, offsets_real)
        y_real_string = encoder_real.flush()
        y_real_strings.append(y_real_string)

        cdf_imag = self.gaussian_conditional_imag.quantized_cdf.tolist()
        cdf_imag_lengths = self.gaussian_conditional_imag.cdf_length.reshape(-1).int().tolist()
        offsets_imag = self.gaussian_conditional_imag.offset.reshape(-1).int().tolist()

        y_real_hat = torch.cat(y_real_hat_slices, dim=1)
        encoder_imag = BufferedRansEncoder()
        symbols_imag_list = []
        indexes_imag_list = []
        y_imag_strings = []

        for slice_index, y_slice in enumerate(y_imag_slices):
            support_slices = (y_imag_hat_slices if self.max_support_slices < 0 else y_imag_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + [y_real_hat]+ support_slices, dim=1)
            mean_support = self.atten_mean_imag[slice_index](mean_support)
            mu = self.cc_mean_transforms_imag[slice_index](mean_support)
            mu = mu[:, :, :, :]

            scale_support = torch.cat([latent_scales] + [y_real_hat]+ support_slices, dim=1)
            scale_support = self.atten_scale_imag[slice_index](scale_support)
            scale = self.cc_scale_transforms_imag[slice_index](scale_support)
            scale = scale[:, :, :, :]

            index = self.gaussian_conditional_imag.build_indexes(scale)
            y_q_slice = self.gaussian_conditional_imag.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_imag_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_imag_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([y_real_hat]+ [mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms_imag[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_imag_hat_slices.append(y_hat_slice)
            y_scale_imag_list.append(scale)
            y_mu_imag_list.append(mu)

        encoder_imag.encode_with_indexes(symbols_imag_list, indexes_imag_list, cdf_imag, cdf_imag_lengths, offsets_imag)
        y_imag_string = encoder_imag.flush()
        y_imag_strings.append(y_imag_string)


        return {"strings": [y_real_strings, y_imag_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_real_string = strings[0][0]
        y_imag_string = strings[1][0]

        y_real_hat_slices = []
        cdf_real = self.gaussian_conditional_real.quantized_cdf.tolist()
        cdf_real_lengths = self.gaussian_conditional_real.cdf_length.reshape(-1).int().tolist()
        offsets_real = self.gaussian_conditional_real.offset.reshape(-1).int().tolist()

        decoder_real = RansDecoder()
        decoder_real.set_stream(y_real_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_real_hat_slices if self.max_support_slices < 0 else y_real_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean_real[slice_index](mean_support)
            mu = self.cc_mean_transforms_real[slice_index](mean_support)
            mu = mu[:, :, :, :]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale_real[slice_index](scale_support)
            scale = self.cc_scale_transforms_real[slice_index](scale_support)
            scale = scale[:, :, :, :]

            index = self.gaussian_conditional_real.build_indexes(scale)

            rv = decoder_real.decode_stream(index.reshape(-1).tolist(), cdf_real, cdf_real_lengths, offsets_real)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_real_hat_slice = self.gaussian_conditional_real.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_real_hat_slice], dim=1)
            lrp = self.lrp_transforms_real[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_real_hat_slice += lrp

            y_real_hat_slices.append(y_real_hat_slice)

        y_real_hat = torch.cat(y_real_hat_slices, dim=1)

        y_imag_hat_slices = []
        cdf_imag = self.gaussian_conditional_imag.quantized_cdf.tolist()
        cdf_imag_lengths = self.gaussian_conditional_imag.cdf_length.reshape(-1).int().tolist()
        offsets_imag = self.gaussian_conditional_imag.offset.reshape(-1).int().tolist()

        decoder_imag = RansDecoder()
        decoder_imag.set_stream(y_imag_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_imag_hat_slices if self.max_support_slices < 0 else y_imag_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + [y_real_hat] + support_slices, dim=1)
            mean_support = self.atten_mean_imag[slice_index](mean_support)
            mu = self.cc_mean_transforms_imag[slice_index](mean_support)
            mu = mu[:, :, :, :]

            scale_support = torch.cat([latent_scales]+[y_real_hat]+ support_slices, dim=1)
            scale_support = self.atten_scale_imag[slice_index](scale_support)
            scale = self.cc_scale_transforms_imag[slice_index](scale_support)
            scale = scale[:, :, :, :]

            index = self.gaussian_conditional_imag.build_indexes(scale)

            rv = decoder_imag.decode_stream(index.reshape(-1).tolist(), cdf_imag, cdf_imag_lengths, offsets_imag)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_imag_hat_slice = self.gaussian_conditional_imag.dequantize(rv, mu)

            lrp_support = torch.cat([y_real_hat]+[mean_support, y_imag_hat_slice], dim=1)
            lrp = self.lrp_transforms_imag[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_imag_hat_slice += lrp

            y_imag_hat_slices.append(y_imag_hat_slice)

        y_imag_hat = torch.cat(y_imag_hat_slices, dim=1)


        #y_hat = torch.cat((y_real_hat, y_imag_hat), 1)

        dwt_processed = torch.cat([y_real_hat, y_imag_hat], dim=1)

        # IDWT 
        y_hat = self.idwt(dwt_processed)

        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}