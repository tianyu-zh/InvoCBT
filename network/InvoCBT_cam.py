import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_

from network.amm import AMM


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y, attn_weight = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weight


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer()

        self.conv4 = nn.Conv2d(med_planes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x = x + residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class AMPDown(nn.Module):
    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(AMPDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.avg_pool = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)
        x = self.max_pool(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class AMPUp(nn.Module):
    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(AMPUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        out = F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

        return out


class Med_ConvBlock(nn.Module):
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer()

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = x + residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=1, res_conv=True,
                                          groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = AMPDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = AMPUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t, attn_weight = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t, attn_weight


class ConvBlock_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock_UNet, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class UpBlock_UNet(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock_UNet, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock_UNet(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)

    def forward(self, x1):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x = self.up(x1)
        return x


class StemTranspose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StemTranspose, self).__init__()
        self.maxpool = nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1,
                                          bias=False)
        self.conv1 = nn.ConvTranspose2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)

    def forward(self, x):
        return self.conv1(self.maxpool(x))


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.linear_layer = self.params['linear_layer']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_UNet(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=self.bilinear)
        self.up2 = UpBlock_UNet(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=self.bilinear)
        self.up3 = UpBlock_UNet(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=self.bilinear)
        self.up4 = UpBlock_UNet(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=self.bilinear)

        if self.linear_layer:
            self.out_conv = nn.Linear(self.ft_chns[0], self.n_class)
        else:
            self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        if self.linear_layer:
            x = x.permute([0, 2, 3, 1]).contiguous()
            output = self.out_conv(x)
            output = output.permute([0, 3, 1, 2]).contiguous()
        else:
            output = self.out_conv(x)

        return output


class Decoder_trans(nn.Module):
    def __init__(self, params):
        super(Decoder_trans, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.linear_layer = self.params['linear_layer']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=self.bilinear)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=self.bilinear)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=self.bilinear)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=self.bilinear)

        if self.linear_layer:
            self.out_conv = nn.Linear(self.ft_chns[0], self.n_class)
        else:
            self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature):
        x = feature

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        if self.linear_layer:
            x = x.permute([0, 2, 3, 1]).contiguous()
            output = self.out_conv(x)
            output = output.permute([0, 3, 1, 2]).contiguous()
        else:
            output = self.out_conv(x)

        return output


class Decoder_cam(nn.Module):
    def __init__(self, params):
        super(Decoder_cam, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.linear_layer = self.params['linear_layer']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_UNet(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=self.bilinear)
        self.up2 = UpBlock_UNet(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=self.bilinear)
        self.up3 = UpBlock_UNet(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=self.bilinear)
        self.up4 = UpBlock_UNet(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=self.bilinear)

        if self.linear_layer:
            self.out_conv = nn.Linear(self.ft_chns[0], self.n_class)
        else:
            self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        if self.linear_layer:
            x = x.permute([0, 2, 3, 1]).contiguous()
            output = self.out_conv(x)
            output = output.permute([0, 3, 1, 2]).contiguous()
        else:
            output = self.out_conv(x)

        return output


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(DownBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(out_channels)
        self.act1 = act_layer()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return self.act1(x)


class Encoder_cam(nn.Module):
    def __init__(self, params):
        super(Encoder_cam, self).__init__()
        self.params = params
        self.ft_chns = self.params['feature_chns']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)

        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.down4(self.down3(self.down2(self.down1(x[0]))))
        x1 = self.down4(self.down3(self.down2(x[1])))
        x2 = self.down4(self.down3(x[2]))
        x3 = self.down4(x[3])
        x4 = x[4]
        return [x0, x1, x2, x3, x4]


class Net(nn.Module):

    def __init__(self, linear_layer, bilinear,
                 patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=in_chans, outplanes=stage_1_channel // 4, res_conv=True, stride=1)
        self.amm_1 = AMM(stage_1_channel // 4, 16)
        self.conv_cls_head_1 = nn.Conv2d(stage_1_channel // 4, self.num_classes, 1, bias=False)

        self.conv_2 = ConvBlock(inplanes=stage_1_channel // 4, outplanes=stage_1_channel // 2, res_conv=True, stride=1)
        self.amm_2 = AMM(stage_1_channel // 2, 16)
        self.conv_cls_head_2 = nn.Conv2d(stage_1_channel // 2, self.num_classes, 1, bias=False)

        self.conv_3 = ConvBlock(inplanes=stage_1_channel // 2, outplanes=stage_1_channel, res_conv=True, stride=1)

        self.trans_patch_conv = nn.Conv2d(stage_1_channel, embed_dim, kernel_size=trans_dw_stride,
                                          stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )
        self.amm_3 = AMM(stage_1_channel, 16)
        self.conv_cls_head_3 = nn.Conv2d(stage_1_channel, self.num_classes, 1, bias=False)

        stage_2_channel = int(base_channel * channel_ratio * 2)
        init_stage = fin_stage
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )
        self.amm_4 = AMM(stage_2_channel, 16)
        self.conv_cls_head_4 = nn.Conv2d(stage_2_channel, self.num_classes, 1, bias=False)

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        init_stage = fin_stage
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )
        self.amm_5 = AMM(stage_3_channel, 16)
        self.conv_cls_head_5 = nn.Conv2d(stage_3_channel, self.num_classes, 1, bias=False)

        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

        params_cnn = {'in_chns': in_chans,
                      'feature_chns': [base_channel * channel_ratio // 4, base_channel * channel_ratio // 2,
                                       base_channel * channel_ratio // 1,
                                       base_channel * channel_ratio * 2, base_channel * channel_ratio * 4],
                      'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                      'class_num': num_classes,
                      'bilinear': bilinear,
                      'acti_func': 'relu',
                      'linear_layer': linear_layer}
        self.decoder_cnn = Decoder(params_cnn)
        params_trans = {'in_chns': embed_dim,
                        'feature_chns': [32, 64, 128, 256, 384],
                        'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                        'class_num': num_classes,
                        'bilinear': bilinear,
                        'acti_func': 'relu',
                        'linear_layer': linear_layer}
        self.decoder_trans = Decoder_trans(params_trans)
        params_cam = {'feature_chns': [num_classes] * 5,
                      'dropout': [0.05, 0.1, 0.2, 0.3, 0.5], }
        self.encoder_cam = Encoder_cam(params_cam)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def returnCAM(self, x, weight_softmax):
        x = x.permute([0, 2, 3, 1]).contiguous()
        output = weight_softmax(x)
        output = output.permute([0, 3, 1, 2]).contiguous()
        return output

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        feature_conv = []
        feature_trans = []
        x_cams = []
        conv_flats = []

        attn_weights = []

        x = self.conv_1(x, return_x_2=False)
        feature_conv.append(x)
        x_amm1 = self.amm_1(x)
        x_cam1 = self.conv_cls_head_1(x_amm1)
        x_cam1 = F.relu(x_cam1, )
        x_cams.append(x_cam1)
        conv_flat = x_cam1.flatten(start_dim=2)
        conv_flats.append(conv_flat)

        x = self.maxpool(x)
        x = self.conv_2(x, return_x_2=False)
        feature_conv.append(x)
        x_amm2 = self.amm_2(x)
        x_cam2 = self.conv_cls_head_2(x_amm2)
        x_cam2 = F.relu(x_cam2, )
        x_cams.append(x_cam2)
        conv_flat = x_cam2.flatten(start_dim=2)
        conv_flats.append(conv_flat)

        x = self.maxpool(x)
        x = self.conv_3(x, return_x_2=False)

        x_t = self.trans_patch_conv(x).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)

        x_t, attn_weight = self.trans_1(x_t)
        attn_weights.append(attn_weight)

        for i in range(2, self.fin_stage):
            x, x_t, attn_weight = eval('self.conv_trans_' + str(i))(x, x_t)
            attn_weights.append(attn_weight)

            if i % 4 == 0:
                feature_conv.append(x)
                feature_trans.append(x_t)
                x_amm = eval('self.amm_' + str(i // 4 + 2))(x)
                x_cam = eval('self.conv_cls_head_' + str(i // 4 + 2))(x_amm)
                x_cam = F.relu(x_cam, )
                x_cams.append(x_cam)
                conv_flat = x_cam.flatten(start_dim=2)
                conv_flats.append(conv_flat)

        x_patch = x_t[:, 1:]
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()

        seg_conv = self.decoder_cnn(feature_conv)
        seg_trans = self.decoder_trans(x_patch)
        seg_cam = self.encoder_cam(x_cams)

        return seg_conv, seg_trans, seg_cam
