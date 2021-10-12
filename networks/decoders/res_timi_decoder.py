import torch.nn as nn
from   networks.ops import memory_attention
from   networks.decoders.res_shortcut_dec import ResShortCut_D_Dec
import torch.nn.functional as F




class conv_bn_relu(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(conv_bn_relu,self).__init__()

        self.conv = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):

        x = self.relu(self.bn(self.conv(x)))

        return x


def upsample(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class Res_TIMI_de(ResShortCut_D_Dec):

    def __init__(self, block, layers, enc_expansion=1, norm_layer=None, large_kernel=False):
        super(Res_TIMI_de, self).__init__(block, layers, enc_expansion, norm_layer, large_kernel)

        self.memory_de = memory_attention(64)


    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        image_feature = mid_fea['image_feature']
        trimap_feature = mid_fea['trimap_feature']

        x = self.layer1(x) + fea5  # N x 256 x 32 x 32

        # scale-64-self-attention
        x  = self.layer2(x) + fea4  # N x 128 x 64 x 64

        x = self.memory_de(x,image_feature,trimap_feature)

        x = self.layer3(x) + fea3

        x = self.layer4(x) + fea2

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha

