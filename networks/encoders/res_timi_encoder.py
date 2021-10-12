import torch
import torch.nn as nn
import torch.nn.functional as F

from   networks.encoders.resnet_enc import ResNet_D
from   networks.ops import SpectralNorm, memory_attention


def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),
        nn.BatchNorm2d(out_channel))
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# The definition of the ResBlock，For more details, please refer to the diagram of the original resnet
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 16, 7, 2, 3, bias=False), nn.BatchNorm2d(16),
            nn.ReLU(True), nn.MaxPool2d(3, 2, 1))
        self.layer1 = make_layer(16, 32, 1)
        self.layer2 = make_layer(32, 64, 1, stride=2)


    def forward(self, x):  #512
        x = self.pre(x)    #256
        x = self.layer1(x) #128
        x = self.layer2(x) #64


        return x

class conv_bn_relu(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(conv_bn_relu,self).__init__()

        self.conv = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):

        x = self.relu(self.bn(self.conv(x)))

        return x


class Res_TIMI(ResNet_D):

    def __init__(self, block, layers, norm_layer=None, late_downsample=False):
        super(Res_TIMI, self).__init__(block, layers, norm_layer, late_downsample=late_downsample)
        first_inplane = 3 + 1

        self.shortcut_inplane = [first_inplane,
                                 self.midplanes,
                                 64 * block.expansion,
                                 128 * block.expansion,
                                 256 * block.expansion]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()

        self.res_module = Resnet()

        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))

        self.guidance_head1 = nn.Sequential( # N x 16 x 256 x 256
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(16),
        )
        self.guidance_head2 = nn.Sequential( # N x 32 x 128 x 128
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(32),
        )
        self.guidance_head3 = nn.Sequential( # N x 64 x 64 x 64
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(64)
        )

        self.memory = memory_attention(64)

        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(512, 256, 1, padding=0, dilation=dilations[0], BatchNorm=nn.BatchNorm2d)
        self.aspp2 = _ASPPModule(512, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=nn.BatchNorm2d)
        self.aspp3 = _ASPPModule(512, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=nn.BatchNorm2d)
        self.aspp4 = _ASPPModule(512, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=nn.BatchNorm2d)
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp_conv = conv_bn_relu(1280,512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes)
        )

    def forward(self, x):
        trimap_feature = self.res_module(x[:, 3:4, ...])

        im_fea1 = self.guidance_head1(x[:, :3, ...])
        im_fea2 = self.guidance_head2(im_fea1)
        image_feature = self.guidance_head3(im_fea2)


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out) # N x 32 x 256 x 256
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out) # N x 64 x 128 x 128

        #working
        x2 = self.layer1(out) # N x 64 x 128 x 128
        x3= self.layer2(x2) # N x 128 x 64 x 64 mainstream
        memory_attention = self.memory(x3,image_feature,trimap_feature)
        x4 = self.layer3(memory_attention) # N x 256 x 32 x 32

        x5 = self.layer_bottleneck(x4) # N x 512 x 16 x 16

        out = torch.cat([
            self.aspp1(x5),
            self.aspp2(x5),
            self.aspp3(x5),
            self.aspp4(x5),
            F.interpolate(self.aspp_pool(x5), size=x5.size()[2:], mode='bilinear', align_corners=True)
        ], dim=1)

        out = self.aspp_conv(out)

        fea1 = self.shortcut[0](x) # i
        fea2 = self.shortcut[1](x1) #36
        fea3 = self.shortcut[2](x2) #64
        fea4 = self.shortcut[3](x3) #128
        fea5 = self.shortcut[4](x4) #256

        return out, {
                    'shortcut': (fea1, fea2, fea3, fea4, fea5),
                     'image_feature': image_feature,
                     'trimap_feature': trimap_feature,
                     }

