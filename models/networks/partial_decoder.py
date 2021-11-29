import torch
import torch.nn as nn
from models.networks.layers01 import PackLayerConv3d, UnpackLayerConv3d, Conv2D, ResidualBlock, InvDepth,seg
class PreConvBlock(nn.Module):
    """Decoder basic block
    """

    def __init__(self, pos, n_in, n_out):
        super().__init__()
        self.pos = pos

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(n_in, n_out, 3)
        self.nl = nn.ELU()

    def forward(self, *x):
        if self.pos == 0:
            x_pre = x[:self.pos]
            x_cur = x[self.pos]
            x_pst = x[self.pos + 1:]
        else:
            x_pre = x[:self.pos]
            x_cur = x[self.pos - 1]
            x_pst = x[self.pos + 1:]

        x_cur = self.pad(x_cur)
        x_cur = self.conv(x_cur)
        x_cur = self.nl(x_cur)

        return x_pre + (x_cur, ) + x_pst

class UpSkipBlock(nn.Module):
    """Decoder basic block

    Perform the following actions:
        - Upsample by factor 2
        - Concatenate skip connections (if any)
        - Convolve
    """

    def __init__(self, pos, ch_in, ch_skip, ch_out):
        super().__init__()
        self.pos = pos

        self.up = nn.Upsample(scale_factor=2)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(ch_in + ch_skip, ch_out, 3)
        self.nl = nn.ELU()

    def forward(self, *x):
        if self.pos == 5:
            x_pre = x[:self.pos - 1]
            x_new = x[self.pos - 1]
            x_skp = tuple()
            x_pst = x[self.pos:]
        else:
            x_pre = x[:self.pos - 1]
            x_new = x[self.pos - 1]
            x_skp = x[self.pos]
            x_pst = x[self.pos:]

        # upscale the input:
        x_new = self.up(x_new)

        # Mix in skip connections from the encoder
        # (if there are any)
        if len(x_skp) > 0:
            x_new = torch.cat((x_new, x_skp), 1)

        # Combine up-scaled input and skip connections
        x_new = self.pad(x_new)
        x_new = self.conv(x_new)
        x_new = self.nl(x_new)

        return x_pre + (x_new, ) + x_pst

class PartialDecoder(nn.Module):
    """Decode some features encoded by a feature extractor

    Args:
        chs_dec: A list of decoder-internal channel counts
        chs_enc: A list of channel counts that we get from the encoder
        start: The first step of the decoding process this decoder should perform
        end: The last step of the decoding process this decoder should perform
    """

    def __init__(self, chs_dec, chs_enc, start=0, end=None):
        super().__init__()

        self.start = start
        self.end = (2 * len(chs_dec)) if (end is None) else end

        self.chs_dec = tuple(chs_dec)
        self.chs_enc = tuple(chs_enc)

        self.blocks = nn.ModuleDict()

        for step in range(self.start, self.end):
            macro_step = step // 2
            mini_step = step % 2
            pos_x = (step + 1) // 2

            # The decoder steps are interleaved ...
            if (mini_step == 0):
                n_in = self.chs_dec[macro_step - 1] if (macro_step > 0) else self.chs_enc[0]
                n_out = self.chs_dec[macro_step]

                # ... first there is a pre-convolution ...
                self.blocks[f'step_{step}'] = PreConvBlock(pos_x, n_in, n_out)

            else:
                # ... and then an upsampling and convolution with
                # the skip connections input.
                n_in = self.chs_dec[macro_step]
                n_skips = self.chs_enc[macro_step + 1] if ((macro_step + 1) < len(chs_enc)) else 0
                n_out = self.chs_dec[macro_step]

                self.blocks[f'step_{step}'] = UpSkipBlock(pos_x, n_in, n_skips, n_out)

    def chs_x(self):
        return self.chs_dec

    @classmethod
    def gen_head(cls, chs_dec, chs_enc, end=None):
        return cls(chs_dec, chs_enc, 0, end)

    @classmethod
    def gen_tail(cls, head, end=None):
        return cls(head.chs_dec, head.chs_enc, head.end, end)

    def forward(self, *x):
        for step in range(self.start, self.end):
            x = self.blocks[f'step_{step}'](*x)
        return x
class PackNetSlim02(nn.Module):
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
        out_channels = 20
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
        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0], d=num_3d_feat)
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1], d=num_3d_feat)
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2], d=num_3d_feat)
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3], d=num_3d_feat)
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4], d=num_3d_feat)

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)
        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.seg4_layer = seg(n4, out_channels=out_channels)
        self.seg3_layer = seg(n3, out_channels=out_channels)
        self.seg2_layer = seg(n2, out_channels=out_channels)
        self.seg1_layer = seg(n1, out_channels=out_channels)

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        zeg = self.pre_calc(x[5])

        # Skips

        skip1 = zeg
        skip2 = x[4]
        skip3 = x[3]
        skip4 = x[2]
        skip5 = x[1]
        unpack5 = self.unpack5(x[0])
        # if self.version == 'A':
        concat5 = torch.cat((unpack5, skip5), 1)
        # else:
        #     concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        # if self.version == 'A':
        concat4 = torch.cat((unpack4, skip4), 1)
        # else:
        #     concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        seg4 = self.seg4_layer(iconv4)
        udisp4 = self.unpack_disp4(seg4)

        unpack3 = self.unpack3(iconv4)
        # if self.version == 'A':
        concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        # else:
        #     concat3 = torch.cat((unpack3 + skip3, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        seg3 = self.seg3_layer(iconv3)
        udisp3 = self.unpack_disp3(seg3)

        unpack2 = self.unpack2(iconv3)
        # if self.version == 'A':
        concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        # else:
        #     concat2 = torch.cat((unpack2 + skip2, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        seg2 = self.seg2_layer(iconv2)
        udisp2 = self.unpack_disp2(seg2)

        unpack1 = self.unpack1(iconv2)
        # if self.version == 'A':
        concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        # else:
        #     concat1 = torch.cat((unpack1 +  skip1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        seg1 = self.seg1_layer(iconv1)

        # if self.training:
        return seg1
class PackNetSlim03(nn.Module):
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
        self.unpack5 = UnpackLayerConv3d(n5, n5o, unpack_kernel[0], d=num_3d_feat)
        self.unpack4 = UnpackLayerConv3d(n5, n4o, unpack_kernel[1], d=num_3d_feat)
        self.unpack3 = UnpackLayerConv3d(n4, n3o, unpack_kernel[2], d=num_3d_feat)
        self.unpack2 = UnpackLayerConv3d(n3, n2o, unpack_kernel[3], d=num_3d_feat)
        self.unpack1 = UnpackLayerConv3d(n2, n1o, unpack_kernel[4], d=num_3d_feat)

        self.iconv5 = Conv2D(n5i, n5, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1, out_channels=out_channels)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        zeg = self.pre_calc(x[5])

        # Encoder

        # x1 = self.conv1(x)
        # x1p = self.pack1(x1)
        # x2 = self.conv2(x1p)
        # x2p = self.pack2(x2)
        # x3 = self.conv3(x2p)
        # x3p = self.pack3(x3)
        # x4 = self.conv4(x3p)
        # x4p = self.pack4(x4)
        # x5 = self.conv5(x4p)
        # x5p = self.pack5(x5)

        # Skips

        skip1 = zeg
        skip2 = x[4]
        skip3 = x[3]
        skip4 = x[2]
        skip5 = x[1]
        unpack5 = self.unpack5(x[0])
        # if self.version == 'A':
        concat5 = torch.cat((unpack5, skip5), 1)
        # else:
        #     concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        # if self.version == 'A':
        concat4 = torch.cat((unpack4, skip4), 1)
        # else:
        #     concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4_layer(iconv4)
        udisp4 = self.unpack_disp4(disp4)

        unpack3 = self.unpack3(iconv4)
        # if self.version == 'A':
        concat3 = torch.cat((unpack3, skip3, udisp4), 1)
        # else:
        #     concat3 = torch.cat((unpack3 + skip3, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3_layer(iconv3)
        udisp3 = self.unpack_disp3(disp3)

        unpack2 = self.unpack2(iconv3)
        # if self.version == 'A':
        concat2 = torch.cat((unpack2, skip2, udisp3), 1)
        # else:
        #     concat2 = torch.cat((unpack2 + skip2, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2_layer(iconv2)
        udisp2 = self.unpack_disp2(disp2)

        unpack1 = self.unpack1(iconv2)
        # if self.version == 'A':
        concat1 = torch.cat((unpack1, skip1, udisp2), 1)
        # else:
        #     concat1 = torch.cat((unpack1 +  skip1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1_layer(iconv1)

        # if self.training:
        return [disp4, disp3, disp2, disp1]