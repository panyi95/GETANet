import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from droneRGBT.wt import SwinTransformerBlock3D

class mMMCCN_IAWT(nn.Module):
    def __init__(self):
        super(mMMCCN_IAWT, self).__init__()

        # ****************RGB_para****************
        self.RGB_para1_1x1 = Conv2d(64, 256, 1, same_padding=True, NL='relu')
        self.RGB_para2_1x1 = nn.Sequential(Conv2d(256, 512, 1, same_padding=True, NL='relu'), nn.MaxPool2d(kernel_size=2, stride=2))

        # *********T_para**********************
        self.T_para1_1x1 = Conv2d(64, 256, 1, same_padding=True, NL='relu')
        self.T_para2_1x1 = nn.Sequential(Conv2d(256, 512, 1, same_padding=True, NL='relu'), nn.MaxPool2d(kernel_size=2, stride=2))


        # *** prediction****
        self.de_pred1 = Conv2d(1024, 128, 1, same_padding=True, NL='relu')
        self.de_pred2 = Conv2d(128, 1, 1, same_padding=True, NL='relu')

        self.reduce = Conv2d(1024, 512, 1, same_padding=True, NL='relu')

        self.ma1 = ModalityAttention(64)
        self.ma2 = ModalityAttention(256)
        self.ma3 = ModalityAttention(512)


        self.conv1_rgb = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv1_t = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_rgb = make_res_layer(Bottleneck, 64, 64, 3, 1)
        self.conv3_rgb = make_res_layer(Bottleneck, 256, 128, 4, 2)

        self.conv2_t = make_res_layer(Bottleneck, 64, 64, 3, 1)
        self.conv3_t = make_res_layer(Bottleneck, 256, 128, 4, 2)

        self.conv4_x = make_res_layer(Bottleneck, 512, 256, 6)


        self.swin3dfuse = SwinTransformerBlock3D(dim=512, num_heads=8, window_size=(2,1,1))

    def forward(self, img):

        rgb, ir = img  # (B, 3, h, w)
        featR = self.conv1_rgb(rgb)
        featT = self.conv1_t(ir)   # (B, 64, h/4, w/4)

        featR, featT = self.ma1(featR, featT)     # (B, 64, h/4, w/4)

        # **********block 1*********
        feat_T = self.T_para1_1x1(featT)  # (B, 256, h/4, w/4)
        feat_R = self.RGB_para1_1x1(featR)   # (B, 256, h/4, w/4)

        feat_MT = self.conv2_t(featT)   # (B, 256, h/4, w/4)
        feat_MR = self.conv2_rgb(featT)   # (B, 256, h/4, w/4)

        featT = feat_MT + feat_T    # (B, 256, h/4, w/4)
        featR = feat_MR + feat_R    # (B, 256, h/4, w/4)
        featR, featT = self.ma2(featR, featT)   # (B, 256, h/4, w/4)

        # *********block 2 ********
        feat_T = self.T_para2_1x1(featT)  # (B, 512, h/8, w/8)
        feat_R = self.RGB_para2_1x1(featR)   # (B, 512, h/8, w/8)

        feat_MT = self.conv3_rgb(featT)    # (B, 512, h/8, w/8)
        feat_MR = self.conv3_t(featT)    # (B, 512, h/8, w/8)

        featR = feat_MR + feat_R   # (B, 512, h/8, w/8)
        featT = feat_MT + feat_T   # (B, 512, h/8, w/8)
        featR, featT = self.ma3(featR, featT)   # (B, 512, h/8, w/8)

        # ***********fusion************
        feat = torch.stack([featT, featR], 1)  # (B, 2, 512, h/8, w/8)
        feat = feat.permute(0, 1, 3, 4, 2)   # (B, 2, h/8, w/8, 512)
        feat = self.swin3dfuse(feat)   # (B, 2, h/8, w/8, 512)

        feat1, feat2 = feat.chunk(2, dim=1)  # (B, h/8, w/8, 512)
        feat = torch.cat([feat1, feat2], -1).squeeze(1)   # (B, h/8, w/8, 1024)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # (B, 1024, h/8, w/8)

        feat = self.reduce(feat)  # (B, 512, h/8, w/8)

        # ********block3 **********
        conv4_feat = self.conv4_x(feat)      # (B, 512, h/8, w/8)
        de_pred1_feat = self.de_pred1(conv4_feat)
        de_pred2_feat = self.de_pred2(de_pred1_feat)
        #feat = F.upsample(de_pred2_feat, scale_factor=8)
        feat = de_pred2_feat
        return feat



def make_res_layer(block, inplanes, planes, blocks, stride=1):

    downsample = None

    #inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   # (B, 2C, H, W)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))   # (B, 2C)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))   # (B, 2C)
        out = avg_out + max_out
        out = self.sigmoid(out)
        #print("Channel attention shape : {}".format(out.size()))
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        x = self.conv1(x)  # (B, 1, H, W)
        #print("spatial attention shape : {}".format(x.size()))
        return self.sigmoid(x)  # (B, 1, H, W)

class ModalityAttention(nn.Module):
    def __init__(self, in_planes):
       super(ModalityAttention, self).__init__()
       self.in_planes = in_planes
       self.mod_t = SpatialAttention()
       self.mod_r = SpatialAttention()
       self.cross = ChannelAttention(in_planes*2)

    def forward(self, r, t):  # (B, C, H, W),  (B, C, H, W)
        atten_r = self.mod_r(r) * r  # (B, 1, H, W) * (B, C, H, W) = (B, C, H, W)
        atten_t = self.mod_t(t) * t  # (B, 1, H, W) * (B, C, H, W) = (B, C, H, W)

        m = torch.cat([r, t], 1)  # (B, 2C, H, W)
        atten_m = torch.cat([atten_r, atten_t], 1)  # (B, 2C, H, W)

        feat = self.cross(m) * atten_m

        return feat[:, :self.in_planes, :, :], feat[:, self.in_planes:, :, :]    # (B, C, H, W),  (B, C, H, W)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
def fusion_model():
    model = mMMCCN_IAWT()
    return model

if __name__ == '__main__':
    model = mMMCCN_IAWT()
    rgb = torch.randn(2,3,640,480)
    t = torch.randn(2,3,640,480)
    a = model([rgb,t])
    print(a.shape)
