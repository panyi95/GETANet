import torch
from droneRGBT.wt import SwinTransformerBlock3D

from EncDec.utils import *
class Gatefusion3(nn.Module):
    def __init__(self, channel):
        super(Gatefusion3, self).__init__()
        self.channel = channel
        self.gate = nn.Sigmoid()


    def forward(self, x, y):
        first_fusion = torch.cat((x, y), dim=1)
        gate_fusion = self.gate(first_fusion)
        gate_fusion = torch.split(gate_fusion, self.channel, dim=1)
        fusion_x = gate_fusion[0] * x + x
        fusion_y = gate_fusion[1] * y + y
        fusion = fusion_x + fusion_y
        fusion = torch.abs((fusion)) * fusion + fusion
        #print(first_fusion.shape,fusion.shape)
        return x+y
        #return fusion

class ASPP(nn.Module):
    def __init__(self,in_channel,depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv = nn.Conv2d(in_channel,depth,1,1)
        self.atrous_block1 = nn.Conv2d(in_channel,depth,1,1)
        self.atrous_block2 = nn.Conv2d(in_channel,depth,3,1,padding=2,dilation=2)
        self.atrous_block4 = nn.Conv2d(in_channel,depth,3,1,padding=4,dilation=4)
        self.atrous_block8 = nn.Conv2d(in_channel,depth,3,1,padding=8,dilation=8)
        self.conv_1x1_output = nn.Conv2d(depth*5,depth,1,1)

    def forward(self,x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features,size=size,mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block2 = self.atrous_block2(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block8 = self.atrous_block8(x)

        output = self.conv_1x1_output(torch.cat([image_features,atrous_block1,atrous_block2,atrous_block4,atrous_block8],dim=1))
        return output

class ImageDecoder(nn.Module):
    def __init__(self):

        super(ImageDecoder, self).__init__()
        #channels = [64, 128, 256, 512, 512]
        channels = [96,192,384,768]
        self.channels = [96,192,384,768]
        self.with_corr = with_corr

        # feature fusing: encoder feature and decoder feature
        #self.enc_dec_fusing5 = EncDecFusing(channels[4])
        self.enc_dec_fusing4 = EncDecFusing(channels[3])
        self.enc_dec_fusing3 = EncDecFusing(channels[2])
        self.enc_dec_fusing2 = EncDecFusing(channels[1])
        self.enc_dec_fusing1 = EncDecFusing(channels[0])

        # correlation calculate
        #self.corr_layer6 = CorrelationLayer(feat_channel=channels[5])
        #self.corr_layer5 = CorrelationLayer(feat_channel=channels[4])
        # self.corr_layer4 = CorrelationLayer(feat_channel=channels[3])
        # self.corr_layer3 = CorrelationLayer(feat_channel=channels[2])
        # self.corr_layer2 = CorrelationLayer(feat_channel=channels[1])
        # self.corr_layer1 = CorrelationLayer(feat_channel=channels[0])
        self.corr_layer4 = Gatefusion3(channels[3])
        self.corr_layer3 = Gatefusion3(channels[2])
        self.corr_layer2 = Gatefusion3(channels[1])
        self.corr_layer1 = Gatefusion3(channels[0])
        # decoder modules
        #self.decoder6 = decoder(channels[5], channels[4])
        #self.decoder5 = decoder(channels[4], channels[3])
        self.decoder4 = decoder(channels[3], channels[2])
        self.decoder3 = decoder(channels[2], channels[1])
        self.decoder2 = decoder(channels[1], channels[0])
        self.decoder1 = decoder(channels[0], channels[0])

        # predict layers
        #self.conv_loss6 = nn.Conv2d(in_channels=channels[4], out_channels=1, kernel_size=3, padding=1)
        #self.conv_loss5 = nn.Conv2d(in_channels=channels[3], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss4 = nn.Conv2d(in_channels=channels[2], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss3 = nn.Conv2d(in_channels=channels[1], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss2 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss1 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)

        self.aspp = ASPP(384,384)
        self.aspp_3 = ASPP(192,192)
        self.aspp_2 = ASPP(96,96)

        self.swin3d_1 = SwinTransformerBlock3D(dim=384,num_heads=8,window_size=(2,1,1))
        self.swin3d_2 = SwinTransformerBlock3D(dim=192,num_heads=8,window_size=(2,1,1))
        self.swin3d_3 = SwinTransformerBlock3D(dim=96,num_heads=8,window_size=(2,1,1))

        self.reg_layer = nn.Sequential(
            nn.Conv2d(192,96,kernel_size=1,),
            nn.ReLU(),
            nn.Conv2d(96,48,1),
            nn.ReLU(),
            nn.Conv2d(48,1,1),
            nn.ReLU()

        )
        self.reg_layer2 = nn.Sequential(
            nn.Conv2d(192,96,1),
            nn.ReLU()
        )
        self.reg_layer_3 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(384,96,kernel_size=1,),
            nn.ReLU(),
        )
        self.reg_layer_4 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(576,192,1),
            nn.ReLU()
        )
        self.reg_layer_5 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3,stride=2,padding=1),
            nn.ReLU(),

        )
        self.reg_layer_6 = nn.Sequential(
            nn.Conv2d(384, 48,1 ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48,24,1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24,1,1),
            nn.ReLU()

        )
        self.reg_4 = nn.Sequential(
            nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(768,96,1),
            nn.ReLU(),
            nn.Conv2d(96,1,1),
            nn.ReLU()
        )
        self.reg_2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(384, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, 1, 1),
            nn.ReLU()
        )
        self.reg_1 = nn.Sequential(
            nn.Conv2d(192, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, 1, 1),
            nn.ReLU()
        )
        self.reg_0 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1),
            nn.ReLU()
        )
        self.up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(768,192,1),
            nn.ReLU()
        )
        self.up_2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(384,96,1),
            nn.ReLU()
        )
        # self.reg_10 = nn.Sequential(
        #     nn.Conv2d(96,48,kernel_size=3,stride=2,padding=1),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(),
        #     nn.Conv2d(48,24,1),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU(),
        #     nn.Conv2d(24,1,1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU()
        # )

    def forward(self, image_feats, depth_feats):

        #encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4, encoder_conv5, x6 = image_feats
        #depth_feat1, depth_feat2, depth_feat3, depth_feat4, depth_feat5, depth_feat6 = depth_feats
        encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4,= image_feats
        depth_feat1, depth_feat2, depth_feat3, depth_feat4,= depth_feats
        #print(encoder_conv1.shape,encoder_conv2.shape,encoder_conv3.shape,encoder_conv4.shape)
        lamda = 0.01
        # corr_fea_6, depth_feat6 = self.corr_layer6((x6, depth_feat6))
        # dec_fea_6 = self.decoder6(corr_fea_6, depth_feat6 * lamda)
        # mask6 = self.conv_loss6(dec_fea_6)
        #
        # fus_fea_5 = self.enc_dec_fusing5(encoder_conv5, depth_feat5)
        # corr_fea_5, depth_feat5 = self.corr_layer5((fus_fea_5, depth_feat5))
        # dec_fea_5 = self.decoder5(corr_fea_5, depth_feat5 * lamda)
        # mask5 = self.reg_layer_5(self.conv_loss5(dec_fea_5))
        #mask5 = dec_fea_5

        #fus_fea_4 = self.enc_dec_fusing4(encoder_conv4,depth_feat4)
        corr_fea_4 = self.corr_layer4(encoder_conv4, depth_feat4)
        dec_fea_4 = self.decoder4(corr_fea_4, depth_feat4 * lamda)
        #mask4 = self.reg_layer_4(self.conv_loss4(dec_fea_4))
        mask4 = dec_fea_4
        #print(mask4.shape)

        # image (decoder3)
        fus_fea_3 = self.enc_dec_fusing3(encoder_conv3, dec_fea_4)
        corr_fea_3 = self.corr_layer3(fus_fea_3, depth_feat3)
        dec_fea_3 = self.decoder3(corr_fea_3, depth_feat3 * lamda)
        #mask3 = self.reg_layer_3(self.conv_loss3(dec_fea_3))
        mask3 = dec_fea_3
        # image (decoder2)
        fus_fea_2 = self.enc_dec_fusing2(encoder_conv2, dec_fea_3)
        corr_fea_2 = self.corr_layer2(fus_fea_2, depth_feat2)
        dec_fea_2 = self.decoder2(corr_fea_2, depth_feat2 * lamda)
        #mask2 = self.reg_layer2(self.conv_loss2(dec_fea_2))
        mask2 = dec_fea_2
        # image (decoder1)
        fus_fea_1 = self.enc_dec_fusing1(encoder_conv1, dec_fea_2)
        corr_fea_1 = self.corr_layer1(fus_fea_1, depth_feat1)
        dec_fea_1 = self.decoder1(corr_fea_1, depth_feat1 * lamda)
        #mask1 = self.conv_loss1(dec_fea_1)
        #mask1 = self.reg_layer(mask1)
        mask1 = dec_fea_1
        #output = self.reg_10(mask1)
        #print(mask1.shape)
        #print(dec_fea_1.shape)
        #print(mask1.shape,mask2.shape,mask3.shape,mask4.shape,mask5.shape)
        #mask1 = torch.cat([mask1,self.aspp(mask1)],dim=1)
        #print(mask4.shape,mask3.shape,mask2.shape,mask1.shape)\
        # mask4 = mask4[:, :self.channels[2], :, :]
        # print(mask4.shape)
        # mask4 = self.swin3d_1(mask4)
        # mask4_1 , mask4_2 = mask4.chunk(2,dim=1)
        # mask4_f = torch.cat([mask4_1,mask4_2],dim=1)
        # print(mask4_f.shape)
        mask4 = (torch.stack([mask4,self.aspp(mask4)],dim=1))
        mask4 = mask4.permute(0, 1, 3, 4, 2)
        mask4 = self.swin3d_1(mask4)
        mask4_1, mask4_2 = mask4.chunk(2, dim=1)  # (B, h/8, w/8, 512)
        mask4 = torch.cat([mask4_1, mask4_2], -1).squeeze(1)  # (B, h/8, w/8, 1024)
        mask4 = mask4.permute(0, 3, 1, 2).contiguous()  # (B, 1024, h/8, w/8)
        #print(mask4.shape)
        mask4_1 = self.reg_4(mask4)
        mask3 = (torch.stack([self.aspp_3(mask3),self.up_1(mask4)],dim=1))
        mask3 = mask3.permute(0, 1, 3, 4, 2)
        mask3 = self.swin3d_2(mask3)
        mask3_1, mask3_2 = mask3.chunk(2, dim=1)  # (B, h/8, w/8, 512)
        mask3 = torch.cat([mask3_1, mask3_2], -1).squeeze(1)  # (B, h/8, w/8, 1024)
        mask3 = mask3.permute(0, 3, 1, 2).contiguous()  # (B, 1024, h/8, w/8)
        #print(mask3.shape)
        mask3_1 = self.reg_2(mask3)



        mask2 = torch.stack([self.up_2(mask3),self.aspp_2(mask2)],dim=1)
        mask2 = mask2.permute(0, 1, 3, 4, 2)
        mask2 = self.swin3d_3(mask2)
        #print(mask2.shape)
        mask2_1, mask2_2 = mask2.chunk(2, dim=1)
        mask2 = torch.cat([mask2_1,mask2_2],-1).squeeze(1)
        mask2 = mask2.permute(0,3,1,2).contiguous()
        #print(mask2.shape)
        mask2_1 = self.reg_1(mask2)

        #print(mask1.shape)

        mask1 = torch.stack([mask2,self.reg_0(self.aspp_2(mask1))],dim=1)
        mask1 = mask1.permute(0,1,3,4,2)
        mask1 = self.swin3d_2(mask1)
        mask1_1,mask1_2 = mask1.chunk(2,dim=1)
        mask1 = torch.cat([mask1_1,mask1_2],-1).squeeze(1)
        mask1 = mask1.permute(0,3,1,2).contiguous()
        #print(mask1.shape)
        mask1 = self.reg_layer_6(mask1)


        return mask1, mask2_1, mask3_1, mask4_1,
        #return mask1
        #return torch.abs(output)