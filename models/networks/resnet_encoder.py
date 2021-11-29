import torch
import torch.nn as nn
import torchvision.models as models
from models.networks.layers01 import PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth

RESNETS = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}


class ResnetEncoder(nn.Module):
    """A ResNet that handles multiple input images and outputs skip connections"""

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super().__init__()

        if num_layers not in RESNETS:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.encoder = RESNETS[num_layers](pretrained)

        # Up until this point self.encoder handles 3 input channels.
        # For pose estimation we want to use two input images,
        # which means 6 input channels.
        # Extend the encoder in a way that makes it equivalent
        # to the single-image version when fed with an input image
        # repeated num_input_images times.
        # Further Information is found in the appendix Section B of:
        # https://arxiv.org/pdf/1806.01260.pdf
        # Mind that in this step only the weights are changed
        # to handle 6 (or even more) input channels
        # For clarity the attribute "in_channels" should be changed too,
        # although it seems to me that it has no influence on the functionality
        self.encoder.conv1.weight.data = self.encoder.conv1.weight.data.repeat(
            (1, num_input_images, 1, 1)
        ) / num_input_images

        # Change attribute "in_channels" for clarity
        self.encoder.conv1.in_channels = num_input_images * 3  # Number of channels for a picture = 3

        # Remove fully connected layer
        self.encoder.fc = None

        if num_layers > 34:
            self.num_ch_enc = (64, 256,  512, 1024, 2048)
        else:
            self.num_ch_enc = (64, 64, 128, 256, 512)

    def forward(self, l_0):
        l_0 = self.encoder.conv1(l_0)
        l_0 = self.encoder.bn1(l_0)
        l_0 = self.encoder.relu(l_0)

        l_1 = self.encoder.maxpool(l_0)
        l_1 = self.encoder.layer1(l_1)

        l_2 = self.encoder.layer2(l_1)
        l_3 = self.encoder.layer3(l_2)
        l_4 = self.encoder.layer4(l_3)

        return (l_0, l_1, l_2, l_3, l_4)
class PackNetSlim01(nn.Module):
    """
    PackNet network with 3d convolutions (version 01, from the CVPR paper).
    Slimmer version, with fewer feature channels

    https://arxiv.org/abs/1905.02693

    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, **kwargs):
        super().__init__()
        #self.version = version[1:]
        # Input/output channels
        in_channels = 3
        out_channels = 1
        # Hyper-parameters
        ni, no = 32, out_channels
        n1, n2, n3, n4, n5 = 32, 64, 128, 256, 512
        num_blocks = [2, 2, 3, 3]
        pack_kernel = [5, 3, 3, 3, 3]
        unpack_kernel = [3, 3, 3, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]
        num_3d_feat = 4
        # Initial convolutional layer
        self.pre_calc = Conv2D(in_channels, ni, 5, 1)
        # Support for different versions
        #if self.version == 'A':  # Channel concatenation
        n1o, n1i = n1, n1 + ni + no
        n2o, n2i = n2, n2 + n1 + no
        n3o, n3i = n3, n3 + n2 + no
        n4o, n4i = n4, n4 + n3
        n5o, n5i = n5, n5 + n4
        # elif self.version == 'B':  # Channel addition
        #     n1o, n1i = n1, n1 + no
        #     n2o, n2i = n2, n2 + no
        #     n3o, n3i = n3//2, n3//2 + no
        #     n4o, n4i = n4//2, n4//2
        #     n5o, n5i = n5//2, n5//2
        # else:
        #     raise ValueError('Unknown PackNet version {}'.format(version))

        # Encoder

        self.pack1 = PackLayerConv3d(n1, pack_kernel[0], d=num_3d_feat)
        self.pack2 = PackLayerConv3d(n2, pack_kernel[1], d=num_3d_feat)
        self.pack3 = PackLayerConv3d(n3, pack_kernel[2], d=num_3d_feat)
        self.pack4 = PackLayerConv3d(n4, pack_kernel[3], d=num_3d_feat)
        self.pack5 = PackLayerConv3d(n5, pack_kernel[4], d=num_3d_feat)

        self.conv1 = Conv2D(ni, n1, 7, 1)
        self.conv2 = ResidualBlock(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock(n4, n5, num_blocks[3], 1, dropout=dropout)

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.pre_calc(x)

        # Encoder

        x1 = self.conv1(x)
        x1p = self.pack1(x1)
        x2 = self.conv2(x1p)
        x2p = self.pack2(x2)
        x3 = self.conv3(x2p)
        x3p = self.pack3(x3)
        x4 = self.conv4(x3p)
        x4p = self.pack4(x4)
        x5 = self.conv5(x4p)
        x5p = self.pack5(x5)
        z= (x1p,x2p,x3p,x4p,x5p)
        return z
