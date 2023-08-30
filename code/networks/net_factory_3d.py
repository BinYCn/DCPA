from networks.unet_3D import unet_3D
from networks.vnet import VNet
from networks.DCPA_3D import DCPA_3D
from networks.unet_3D_dv import unet_3D_dv
from networks.MCNet import MCNet3d_v2


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, mode="train"):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "DCPA3d" and mode == "train":
        net = DCPA_3D(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "DCPA3d" and mode == "test":
        net = DCPA_3D(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "unet_3D_dv" and mode == "train":
        net = unet_3D_dv(n_classes=class_num, in_channels=in_chns, is_batchnorm=True).cuda()
    elif net_type == "unet_3D_dv" and mode == "test":
        net = unet_3D_dv(n_classes=class_num, in_channels=in_chns, is_batchnorm=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "train":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "mcnet3d_v2" and mode == "test":
        net = MCNet3d_v2(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    else:
        net = None
    return net
