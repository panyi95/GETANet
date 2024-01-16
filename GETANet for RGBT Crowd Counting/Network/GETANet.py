import torch

from EncDec import ImageDecoder
import torch.nn as nn
from convnext import convnext_tiny



class GETANet(nn.Module):
    def __init__(self):
        super(GETANet, self).__init__()

        # encoder part


        self.channels = [96,192,384,768]

        self.vgg_r = convnext_tiny(pretrained=True)
        self.vgg_d = convnext_tiny(pretrained=True)

        self.conv1_vgg_r = nn.Sequential(self.vgg_r.downsample_layers[0], self.vgg_r.stages[0])
        self.conv1_vgg_d = nn.Sequential(self.vgg_d.downsample_layers[0], self.vgg_d.stages[0])
        self.conv2_vgg_r = nn.Sequential(self.vgg_r.downsample_layers[1], self.vgg_r.stages[1])
        self.conv2_vgg_d = nn.Sequential(self.vgg_d.downsample_layers[1], self.vgg_d.stages[1])
        self.conv3_vgg_r = nn.Sequential(self.vgg_r.downsample_layers[2], self.vgg_r.stages[2])
        self.conv3_vgg_d = nn.Sequential(self.vgg_d.downsample_layers[2], self.vgg_d.stages[2])
        self.conv4_vgg_r = nn.Sequential(self.vgg_r.downsample_layers[3], self.vgg_r.stages[3])
        self.conv4_vgg_d = nn.Sequential(self.vgg_d.downsample_layers[3], self.vgg_d.stages[3])

        # decoder part
        self.ImageDecoder = ImageDecoder()


    def forward(self, RGBT):
    #def forward(self,image_Input,depth_Input):
        image_Input = RGBT[0]
        depth_Input = RGBT[1]


        image_feat_1 = self.conv1_vgg_r(depth_Input)
        image_feat_2 = self.conv2_vgg_r(image_feat_1)
        image_feat_3 = self.conv3_vgg_r(image_feat_2)
        image_feat_4 = self.conv4_vgg_r(image_feat_3)
        image_feat = [image_feat_1,image_feat_2,image_feat_3,image_feat_4]


        depth_feat_1 = self.conv1_vgg_d(image_Input)
        depth_feat_2 = self.conv2_vgg_d(depth_feat_1)
        depth_feat_3 = self.conv3_vgg_d(depth_feat_2)
        depth_feat_4 = self.conv4_vgg_d(depth_feat_3)
        depth_feat = [depth_feat_1,depth_feat_2,depth_feat_3,depth_feat_4]



        outputs_image = self.ImageDecoder(image_feat, depth_feat)



        return outputs_image

    def init_parameters(self, pretrain_vgg16_1024):
        # init all layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # load rgb encoder parameters
        rgb_conv_blocks = [self.ImageEncoder.conv1,
                           self.ImageEncoder.conv2,
                           self.ImageEncoder.conv3,
                           self.ImageEncoder.conv4,
                           self.ImageEncoder.conv5,
                           self.ImageEncoder.fc6]

        listkey = [['conv1_1', 'conv1_2'], ['conv2_1', 'conv2_2'], ['conv3_1', 'conv3_2', 'conv3_3'],
                   ['conv4_1', 'conv4_2', 'conv4_3'], ['conv5_1', 'conv5_2', 'conv5_3'], ['fc6']]

        for idx, conv_block in enumerate(rgb_conv_blocks):
            num_conv = 0
            for l2 in conv_block:
                if isinstance(l2, nn.Conv2d):
                    if 'fc' in listkey[idx][num_conv]:
                        l2.weight.data = pretrain_vgg16_1024[str(listkey[idx][num_conv]) + '.weight'][:512, :512]
                        l2.bias.data = pretrain_vgg16_1024[str(listkey[idx][num_conv])
                                                           + '.bias'][:, :, :, :512].squeeze()
                    else:
                        l2.weight.data = pretrain_vgg16_1024[str(listkey[idx][num_conv]) + '.weight']
                        l2.bias.data = pretrain_vgg16_1024[str(listkey[idx][num_conv]) + '.bias'].squeeze(
                            0).squeeze(
                            0).squeeze(0).squeeze(0)
                    num_conv += 1
        return self

def fusion_model():
    model = GETANet()
    return model


if __name__ == '__main__':
    import time
    net = GETANet()
    net.eval()  # 将网络设置为评估模式

    x = torch.randn(2, 3, 224, 224)

    # 预热网络
    for _ in range(10):
        _ = net([x, x])

    num_inferences = 100  # 您可以根据需要调整这个数字

    # 开始计时
    start_time = time.time()

    # 进行多次推理
    for _ in range(num_inferences):
        a = net([x, x])

    # 结束计时
    end_time = time.time()

    # 计算总推理时间
    total_inference_time = end_time - start_time

    # 计算FPS
    fps = num_inferences / total_inference_time
    print(f"FPS: {fps}")

    a1, a2, a3, a4 = a
    print(a1.shape, a2.shape, a3.shape, a4.shape)
