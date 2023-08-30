# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, mode_upsampling=1):
        super(UnetUp3_CT, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 1:
            self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        else:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        out = self.conv(torch.cat([outputs1, outputs2], 1))
        return out


class Unet_encoder(nn.Module):
    def __init__(self, feature_scale=4, in_channels=3, is_batchnorm=True):
        super(Unet_encoder, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))


        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)

        features = [conv1, conv2, conv3, conv4, center]

        return features


class Unet_decoder(nn.Module):
    def __init__(self, feature_scale=4, n_classes=2, in_channels=3, is_batchnorm=True, up_type=0):
        super(Unet_decoder, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm, up_type)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm, up_type)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm, up_type)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm, up_type)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, features):
        conv1, conv2, conv3, conv4, center = features

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.final(up1)

        return final


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor, mode_upsampling=True):
        super(UnetDsv3, self).__init__()
        if mode_upsampling == True:
            self.dsv = nn.Sequential(nn.ConvTranspose3d(in_size, out_size, scale_factor, padding=0, stride=scale_factor))
        elif mode_upsampling == False:
            self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                     nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)


class Unet_decoder_1(nn.Module):
    def __init__(self, feature_scale=4, n_classes=2, in_channels=3, is_deconv=True, is_batchnorm=True):
        super(Unet_decoder_1, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], is_deconv, is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3], filters[2], is_deconv, is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2], filters[1], is_deconv, is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], is_deconv, is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dsv4 = UnetDsv3(filters[3], n_classes, 8, mode_upsampling=is_deconv)
        self.dsv3 = UnetDsv3(filters[2], n_classes, 4, mode_upsampling=is_deconv)
        self.dsv2 = UnetDsv3(filters[1], n_classes, 2, mode_upsampling=is_deconv)

        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, features):
        conv1, conv2, conv3, conv4, center = features

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)
        up2 = self.dropout2(up2)
        up3 = self.dropout2(up3)
        up4 = self.dropout2(up4)

        up4 = self.dsv4(up4)
        up3 = self.dsv3(up3)
        up2 = self.dsv2(up2)

        final = self.final(up1)

        return [up4, up3, up2, final]


class unet_3D_binycn(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, in_channels=3, is_batchnorm=True):
        super(unet_3D_binycn, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes

        self.encoder = Unet_encoder(self.feature_scale, self.in_channels, self.is_batchnorm)
        self.decoder_transpose = Unet_decoder(self.feature_scale, self.n_classes, self.in_channels, self.is_batchnorm, 0)
        self.decoder_linear = Unet_decoder(self.feature_scale, self.n_classes, self.in_channels, self.is_batchnorm, 1)


    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder_transpose(features)
        out_seg2 = self.decoder_linear(features)
        return out_seg1, out_seg2


class unet_3D_dv(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, in_channels=3, is_batchnorm=True):
        super(unet_3D_dv, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes

        self.encoder = Unet_encoder(self.feature_scale, self.in_channels, self.is_batchnorm)
        self.decoder_transpose = Unet_decoder_1(self.feature_scale, self.n_classes, self.in_channels, True, self.is_batchnorm)
        self.decoder_linear = Unet_decoder_1(self.feature_scale, self.n_classes, self.in_channels, False, self.is_batchnorm)


    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder_transpose(features)
        out_seg2 = self.decoder_linear(features)
        return out_seg1, out_seg2
