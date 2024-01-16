import torch.nn as nn
import torch
import torch.nn.functional as F
from EncDec.parameter import *
from EncDec.pacconv import PacConv2d
# from DualGCNNet import DualGCNHead
# from modelsgcn.DualGCNNet import DualGCNHead
from .DualGCNNet import DualGCNHead
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
def corr_fun(Kernel_tmp, Feature):
    size = Kernel_tmp.size()
    CORR = []
    Kernel = []
    for i in range(len(Feature)):
        ker = Kernel_tmp[i:i + 1]
        fea = Feature[i:i + 1]
        ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
        ker = ker.unsqueeze(2).unsqueeze(3)

        co = F.conv2d(fea, ker.contiguous())
        CORR.append(co)
        ker = ker.unsqueeze(0)
        Kernel.append(ker)
    corr = torch.cat(CORR, 0)
    Kernel = torch.cat(Kernel, 0)
    return corr, Kernel


class CorrelationLayer(nn.Module):
    def __init__(self, feat_channel):
        super(CorrelationLayer, self).__init__()

        self.pool_layer = nn.AdaptiveAvgPool2d((corr_size, corr_size))

        self.corr_reduce = nn.Sequential(
            nn.Conv2d(corr_size * corr_size, feat_channel, kernel_size=1),
            nn.InstanceNorm2d(feat_channel),
            nn.ReLU(),
            nn.Conv2d(feat_channel, feat_channel, 3, padding=1),
        )
        self.Dnorm = nn.InstanceNorm2d(feat_channel)

        self.feat_adapt = nn.Sequential(
            nn.Conv2d(feat_channel * 2, feat_channel, 1),
            #nn.InstanceNorm2d(feat_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # calculate correlation map
        RGB_feat_downsize = F.normalize(self.pool_layer(x[0]))
        RGB_feat_norm = F.normalize(x[0])
        RGB_corr, _ = corr_fun(RGB_feat_downsize, RGB_feat_norm)

        Depth_feat_downsize = F.normalize(self.pool_layer(x[1]))
        Depth_feat_norm = F.normalize(x[1])
        Depth_corr, _ = corr_fun(Depth_feat_downsize, Depth_feat_norm)

        corr = (RGB_corr + Depth_corr) / 2
        Red_corr = self.corr_reduce(corr)

        # beta cond
        new_feat = torch.cat([x[0], Red_corr], 1)
        #new_feat = torch.cat([x[0],x[1]],dim=1)
        new_feat = self.feat_adapt(new_feat)

        Depth_feat = self.Dnorm(x[1])
        #Depth_feat = x[1]
        return new_feat, Depth_feat

class Scale_Selected(nn.Module):
    def __init__(self,in_channels):
        super(Scale_Selected, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,dilation=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,dilation=3,padding=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,dilation=5,padding=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.reg_layer = nn.Sequential(
            nn.Conv2d(in_channels*4,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self,x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x_4 = self.conv4(x)
        out = torch.cat([x_1,x_2,x_3,x_4],dim=1)

        return self.reg_layer(out)
class MSCA(nn.Module):
    def __init__(self, channels=32, r=2):
        super(MSCA, self).__init__()
        out_channels = int(channels//r)
        #local_att
        self.local_att =nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding= 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(channels)
        )

        #global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei
class MBR(nn.Module):
    expansion =1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(MBR, self).__init__()
        # branch1
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample1 = upsample
        self.stride1 = stride
        # barch2
        self.conv3 = nn.Conv2d(inplanes, inplanes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv4 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=5, stride=stride,
                                   padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.conv_cat = BasicConv2d(2 * inplanes, inplanes, 3, padding=1)
        self.upsample2 = upsample
        self.stride2 = stride

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)

        if self.upsample1 is not None:
            residual = self.upsample1(x)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        out2 = self.relu3(out2)

        out2 = self.conv4(out2)
        out2 = self.bn4(out2)

        if self.upsample2 is not None:
            residual = self.upsample2(x)
        out = self.conv_cat(torch.cat((out1, out2), 1))
        out += residual
        out = self.relu(out)

        return out

class DGCM(nn.Module):
    def __init__(self, channel=32):
        super(DGCM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)

        self.h2l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.h2h = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.mscah = MSCA()
        self.mscal = MSCA()

        self.upsample_add = upsample_add
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x):
        # first conv
        x_h = self.h2h(x)
        x_l = self.h2l(self.h2l_pool(x))
        x_h = x_h * self.mscah(x_h)
        x_l = x_l * self.mscal(x_l)
        out = self.upsample_add(x_l, x_h)
        # out = out + x_5
        out = self.conv(out)

        return out
class EncDecFusing(nn.Module):
    def __init__(self, in_channels):
        super(EncDecFusing, self).__init__()
        self.enc_fea_proc = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.sig = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.fusing_layer = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1)
        self.channel = in_channels
        self.reg_layer = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        #self.scale = Scale_Selected(in_channels)
        self.mbr = MBR(inplanes=in_channels,planes=in_channels)
        self.gcn = DualGCNHead(inplanes=in_channels,interplanes=in_channels,num_classes=in_channels)
    def forward(self, enc_fea, dec_fea):
        enc_fea = self.enc_fea_proc(enc_fea)

        if dec_fea.size(2) != enc_fea.size(2):
            dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear',
                                 align_corners=True)

        fuc_fea = torch.cat([enc_fea, dec_fea], dim=1)
        fuc_fea = self.conv(fuc_fea)

        output = self.gcn(fuc_fea)
        #output = fuc_fea
        # gate_fusion = torch.split(fuc_fea,self.channel,dim=1)
        #
        # r_gate_fusion = self.sig(enc_fea) * gate_fusion[0]
        # f_gate_fusion = self.sig(dec_fea) * gate_fusion[0]
        # fusion = r_gate_fusion*f_gate_fusion+r_gate_fusion+f_gate_fusion
        #
        #
        # #output = self.reg_layer(fusion)
        # output = self.reg_layer(fuc_fea)
        #output = self.fusing_layer(fusion)

        return output


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()

        if with_pac:
            self.pac = PacConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.norm = nn.InstanceNorm2d(in_channels)
        self.decoding = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, feat, guide):
        if with_pac:
            feat = self.norm(self.pac(feat, guide)).relu()
        output = self.decoding(feat)
        #print(feat.shape,output.shape)

        return output