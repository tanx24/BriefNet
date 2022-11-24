import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
import copy

config_resnet = {'convert': [[64,256,512,1024,2048],[64,64,64,64,64],[128,128,128,128,128,128,256]],
                 'para': [[5,5,5,5,7],[2,2,2,2,3]]}

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.cfg      = cfg
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./resnet50.pth'), strict=False)

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False), nn.BatchNorm2d(self.inter_channel), nn.ReLU(inplace=True))
        self.conv_theta = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(self.inter_channel), nn.ReLU(inplace=True))
        self.conv_g = nn.Sequential(nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(self.inter_channel), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Sequential(nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(channel), nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out
    
    def initialize(self):   
        return

class EncoderCluster(nn.Module):
    def __init__(self, channel):
        super(EncoderCluster, self).__init__()

        self.nlb = NonLocalBlock(channel)
    
    def forward(self, x, P_h, P_w):
        N, C, H, W = x.size()
        Q_h, Q_w = H // P_h, W // P_w
        x = x.reshape(N, C, Q_h, P_h, Q_w, P_w)

        # Local
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(N*Q_h*Q_w, C, P_h, P_w)
        x = self.nlb(x)
        x = x.reshape(N, Q_h, Q_w, C, P_h, P_w)
        # Global
        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(N*P_h*P_w, C, Q_h, Q_w)
        x = self.nlb(x)
        x = x.reshape(N, P_h, P_w, C, Q_h, Q_w)

        return x.permute(0, 3, 4, 1, 5, 2).reshape(N, C, H, W)
 
    def initialize(self):   
        return

class Decoder(nn.Module):
    def __init__(self, list_cvt, list_para):
        super(Decoder, self).__init__()

        self.convs5 = nn.Sequential(nn.Conv2d(list_cvt[1][4], list_cvt[1][4], kernel_size=list_para[0][4], stride=1, padding=list_para[1][4]),nn.BatchNorm2d(list_cvt[1][4]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][4], list_cvt[1][4], kernel_size=list_para[0][4], stride=1, padding=list_para[1][4]),nn.BatchNorm2d(list_cvt[1][4]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][4], list_cvt[1][4], kernel_size=list_para[0][4], stride=1, padding=list_para[1][4]),nn.BatchNorm2d(list_cvt[1][4]), nn.ReLU(inplace=True))
        self.convs4 = nn.Sequential(nn.Conv2d(list_cvt[1][3], list_cvt[1][3], kernel_size=list_para[0][3], stride=1, padding=list_para[1][3]),nn.BatchNorm2d(list_cvt[1][3]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][3], list_cvt[1][3], kernel_size=list_para[0][3], stride=1, padding=list_para[1][3]),nn.BatchNorm2d(list_cvt[1][3]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][3], list_cvt[1][3], kernel_size=list_para[0][3], stride=1, padding=list_para[1][3]),nn.BatchNorm2d(list_cvt[1][3]), nn.ReLU(inplace=True))
        self.convs3 = nn.Sequential(nn.Conv2d(list_cvt[1][2], list_cvt[1][2], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[1][2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][2], list_cvt[1][2], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[1][2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][2], list_cvt[1][2], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[1][2]), nn.ReLU(inplace=True))
        self.convs2 = nn.Sequential(nn.Conv2d(list_cvt[1][1], list_cvt[1][1], kernel_size=list_para[0][1], stride=1, padding=list_para[1][1]),nn.BatchNorm2d(list_cvt[1][1]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][1], list_cvt[1][1], kernel_size=list_para[0][1], stride=1, padding=list_para[1][1]),nn.BatchNorm2d(list_cvt[1][1]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][1], list_cvt[1][1], kernel_size=list_para[0][1], stride=1, padding=list_para[1][1]),nn.BatchNorm2d(list_cvt[1][1]), nn.ReLU(inplace=True))
        self.convs1 = nn.Sequential(nn.Conv2d(list_cvt[1][0], list_cvt[1][0], kernel_size=list_para[0][0], stride=1, padding=list_para[1][0]),nn.BatchNorm2d(list_cvt[1][0]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][0], list_cvt[1][0], kernel_size=list_para[0][0], stride=1, padding=list_para[1][0]),nn.BatchNorm2d(list_cvt[1][0]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[1][0], list_cvt[1][0], kernel_size=list_para[0][0], stride=1, padding=list_para[1][0]),nn.BatchNorm2d(list_cvt[1][0]), nn.ReLU(inplace=True))

    def forward(self, out1s, out2s, out3s, out4s, out5s):
        convs5 = self.convs5(out5s)
        convs4 = self.convs4(out4s + F.interpolate(convs5, out4s.size()[2:], mode='bilinear', align_corners=True))
        convs3 = self.convs3(out3s + F.interpolate(convs4, out3s.size()[2:], mode='bilinear', align_corners=True))
        convs2 = self.convs2(out2s + F.interpolate(convs3, out2s.size()[2:], mode='bilinear', align_corners=True))
        convs1 = self.convs1(out1s + convs2)

        return convs1, convs2, convs3, convs4, convs5

    def initialize(self):
        weight_init(self)

class ICM(nn.Module):
    def __init__(self, list_cvt, list_para):
        super(ICM, self).__init__()

        self.edge1 = nn.Sequential(nn.Conv2d(list_cvt[1][0], list_cvt[1][0], kernel_size=list_para[0][0], stride=1, padding=list_para[1][0]),nn.BatchNorm2d(list_cvt[1][0]), nn.ReLU(inplace=True))
        self.edge2 = nn.Sequential(nn.Conv2d(list_cvt[1][1], list_cvt[1][1], kernel_size=list_para[0][1], stride=1, padding=list_para[1][1]),nn.BatchNorm2d(list_cvt[1][1]), nn.ReLU(inplace=True))
        self.edge3 = nn.Sequential(nn.Conv2d(list_cvt[1][2], list_cvt[1][2], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[1][2]), nn.ReLU(inplace=True))
        self.edge4 = nn.Sequential(nn.Conv2d(list_cvt[1][3], list_cvt[1][3], kernel_size=list_para[0][3], stride=1, padding=list_para[1][3]),nn.BatchNorm2d(list_cvt[1][3]), nn.ReLU(inplace=True))
        self.edge5 = nn.Sequential(nn.Conv2d(list_cvt[1][4], list_cvt[1][4], kernel_size=list_para[0][4], stride=1, padding=list_para[1][4]),nn.BatchNorm2d(list_cvt[1][4]), nn.ReLU(inplace=True))

        self.sal1 = nn.Sequential(nn.Conv2d(list_cvt[1][0], list_cvt[1][0], kernel_size=list_para[0][0], stride=1, padding=list_para[1][0]),nn.BatchNorm2d(list_cvt[1][0]), nn.ReLU(inplace=True))
        self.sal2 = nn.Sequential(nn.Conv2d(list_cvt[1][1], list_cvt[1][1], kernel_size=list_para[0][1], stride=1, padding=list_para[1][1]),nn.BatchNorm2d(list_cvt[1][1]), nn.ReLU(inplace=True))
        self.sal3 = nn.Sequential(nn.Conv2d(list_cvt[1][2], list_cvt[1][2], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[1][2]), nn.ReLU(inplace=True))
        self.sal4 = nn.Sequential(nn.Conv2d(list_cvt[1][3], list_cvt[1][3], kernel_size=list_para[0][3], stride=1, padding=list_para[1][3]),nn.BatchNorm2d(list_cvt[1][3]), nn.ReLU(inplace=True))
        self.sal5 = nn.Sequential(nn.Conv2d(list_cvt[1][4], list_cvt[1][4], kernel_size=list_para[0][4], stride=1, padding=list_para[1][4]),nn.BatchNorm2d(list_cvt[1][4]), nn.ReLU(inplace=True))

    def forward(self, out1s, out2s, out3s, out4s, out5s, convs1, convs2, convs3, convs4, convs5):
        edge1, conv1 = self.edge1(convs1 * out1s), self.sal1(convs1 + out1s)
        edge2, conv2 = self.edge2(convs2 * out2s), self.sal2(convs2 + out2s)
        edge3, conv3 = self.edge3(convs3 * out3s), self.sal3(convs3 + out3s)
        edge4, conv4 = self.edge4(convs4 * out4s), self.sal4(convs4 + out4s)
        edge5, conv5 = self.edge5(convs5 * out5s), self.sal5(convs5 + out5s)

        edge_feature, sal_feature = [], []

        edge_feature.append(edge1)
        edge_feature.append(edge2)
        edge_feature.append(edge3)
        edge_feature.append(edge4)
        edge_feature.append(edge5)

        sal_feature.append(conv1)
        sal_feature.append(conv2)
        sal_feature.append(conv3)
        sal_feature.append(conv4)
        sal_feature.append(conv5)

        return edge_feature, sal_feature

    def initialize(self):
        weight_init(self)

class ECM(nn.Module):
    def __init__(self, list_cvt, list_para):
        super(ECM, self).__init__()

        self.nlb1 = EncoderCluster(list_cvt[1][0])
        self.nlb2 = EncoderCluster(list_cvt[1][1])
        self.nlb3 = EncoderCluster(list_cvt[1][2])
        self.nlb4 = EncoderCluster(list_cvt[1][3])
        self.nlb5 = EncoderCluster(list_cvt[1][4])

    def forward(self, out1s, out2s, out3s, out4s, out5s, convs1, convs2, convs3, convs4, convs5):
        nlb1 = self.nlb1(out1s, 8, 8)
        nlb2 = self.nlb2(out2s, 8, 8)
        nlb3 = self.nlb3(out3s, 4, 4)
        nlb4 = self.nlb4(out4s, 2, 2)
        nlb5 = self.nlb5(out5s, 1, 1)

        ecm1 = nlb1 + convs1
        ecm2 = nlb2 + convs2
        ecm3 = nlb3 + convs3
        ecm4 = nlb4 + convs4
        ecm5 = nlb5 + convs5

        ecm_feature = []

        ecm_feature.append(ecm1)
        ecm_feature.append(ecm2)
        ecm_feature.append(ecm3)
        ecm_feature.append(ecm4)
        ecm_feature.append(ecm5)

        return ecm_feature

    def initialize(self):
        weight_init(self)

class MergeLayer(nn.Module):
    def __init__(self, list_cvt, list_para):
        super(MergeLayer, self).__init__()

        self.convm25 = nn.Sequential(nn.Conv2d(list_cvt[2][4], list_cvt[2][4], kernel_size=list_para[0][4], stride=1, padding=list_para[1][4]),nn.BatchNorm2d(list_cvt[2][4]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][4], list_cvt[2][4], kernel_size=list_para[0][4], stride=1, padding=list_para[1][4]),nn.BatchNorm2d(list_cvt[2][4]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][4], list_cvt[2][4], kernel_size=list_para[0][4], stride=1, padding=list_para[1][4]),nn.BatchNorm2d(list_cvt[2][4]), nn.ReLU(inplace=True))
        self.convm24 = nn.Sequential(nn.Conv2d(list_cvt[2][3], list_cvt[2][3], kernel_size=list_para[0][3], stride=1, padding=list_para[1][3]),nn.BatchNorm2d(list_cvt[2][3]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][3], list_cvt[2][3], kernel_size=list_para[0][3], stride=1, padding=list_para[1][3]),nn.BatchNorm2d(list_cvt[2][3]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][3], list_cvt[2][3], kernel_size=list_para[0][3], stride=1, padding=list_para[1][3]),nn.BatchNorm2d(list_cvt[2][3]), nn.ReLU(inplace=True))
        self.convm23 = nn.Sequential(nn.Conv2d(list_cvt[2][2], list_cvt[2][2], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[2][2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][2], list_cvt[2][2], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[2][2]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][2], list_cvt[2][2], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[2][2]), nn.ReLU(inplace=True))
        self.convm22 = nn.Sequential(nn.Conv2d(list_cvt[2][1], list_cvt[2][1], kernel_size=list_para[0][1], stride=1, padding=list_para[1][1]),nn.BatchNorm2d(list_cvt[2][1]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][1], list_cvt[2][1], kernel_size=list_para[0][1], stride=1, padding=list_para[1][1]),nn.BatchNorm2d(list_cvt[2][1]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][1], list_cvt[2][1], kernel_size=list_para[0][1], stride=1, padding=list_para[1][1]),nn.BatchNorm2d(list_cvt[2][1]), nn.ReLU(inplace=True))
        self.convm21 = nn.Sequential(nn.Conv2d(list_cvt[2][0], list_cvt[2][0], kernel_size=list_para[0][0], stride=1, padding=list_para[1][0]),nn.BatchNorm2d(list_cvt[2][0]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][0], list_cvt[2][0], kernel_size=list_para[0][0], stride=1, padding=list_para[1][0]),nn.BatchNorm2d(list_cvt[2][0]), nn.ReLU(inplace=True),
                                    nn.Conv2d(list_cvt[2][0], list_cvt[2][0], kernel_size=list_para[0][0], stride=1, padding=list_para[1][0]),nn.BatchNorm2d(list_cvt[2][0]), nn.ReLU(inplace=True))
           
        self.convm = nn.Sequential(nn.Conv2d(list_cvt[2][5], list_cvt[2][5], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[2][5]), nn.ReLU(inplace=True),
                                   nn.Conv2d(list_cvt[2][5], list_cvt[2][5], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[2][5]), nn.ReLU(inplace=True),
                                   nn.Conv2d(list_cvt[2][5], list_cvt[2][5], kernel_size=list_para[0][2], stride=1, padding=list_para[1][2]),nn.BatchNorm2d(list_cvt[2][5]), nn.ReLU(inplace=True))

    def forward(self, edge_feature, sal_feature, ecm_feature):
        merge11 = edge_feature[0]+sal_feature[0]
        merge12 = edge_feature[1]+sal_feature[1]
        merge13 = edge_feature[2]+sal_feature[2]
        merge14 = edge_feature[3]+sal_feature[3]
        merge15 = edge_feature[4]+sal_feature[4]

        merge21 = self.convm21(torch.cat([merge11, ecm_feature[0]], dim=1))
        merge22 = self.convm22(torch.cat([merge12, ecm_feature[1]], dim=1))
        merge23 = self.convm23(torch.cat([merge13, ecm_feature[2]], dim=1))
        merge24 = self.convm24(torch.cat([merge14, ecm_feature[3]], dim=1))
        merge25 = self.convm25(torch.cat([merge15, ecm_feature[4]], dim=1))

        sal13i = F.interpolate(merge23, sal_feature[0].size()[2:], mode='bilinear', align_corners=True)
        sal14i = F.interpolate(merge24, sal_feature[0].size()[2:], mode='bilinear', align_corners=True)
        sal15i = F.interpolate(merge25, sal_feature[0].size()[2:], mode='bilinear', align_corners=True)

        merge = self.convm(merge21 + merge22 + sal13i + sal14i + sal15i)

        merge_feature = []
        merge_feature.append(merge21)
        merge_feature.append(merge22)
        merge_feature.append(merge23)
        merge_feature.append(merge24)
        merge_feature.append(merge25)
        merge_feature.append(merge)

        return merge_feature

    def initialize(self):
        weight_init(self)

class BriefNet(nn.Module):
    def __init__(self, cfg):
        super(BriefNet, self).__init__()

        self.cfg = cfg
        self.bkbone = ResNet(cfg)

        list_cvt = config_resnet['convert']
        list_para = config_resnet['para']

        self.squeeze5s   = nn.Sequential(nn.Conv2d(list_cvt[0][4], list_cvt[1][4], kernel_size=1), nn.BatchNorm2d(list_cvt[1][4]), nn.ReLU(inplace=True), nn.Conv2d(list_cvt[1][4], list_cvt[1][4], kernel_size=list_para[0][4], padding=list_para[1][4]), nn.BatchNorm2d(list_cvt[1][4]), nn.ReLU(inplace=True))
        self.squeeze4s   = nn.Sequential(nn.Conv2d(list_cvt[0][3], list_cvt[1][3], kernel_size=1), nn.BatchNorm2d(list_cvt[1][3]), nn.ReLU(inplace=True), nn.Conv2d(list_cvt[1][3], list_cvt[1][3], kernel_size=list_para[0][3], padding=list_para[1][3]), nn.BatchNorm2d(list_cvt[1][3]), nn.ReLU(inplace=True))
        self.squeeze3s   = nn.Sequential(nn.Conv2d(list_cvt[0][2], list_cvt[1][2], kernel_size=1), nn.BatchNorm2d(list_cvt[1][2]), nn.ReLU(inplace=True), nn.Conv2d(list_cvt[1][2], list_cvt[1][2], kernel_size=list_para[0][2], padding=list_para[1][2]), nn.BatchNorm2d(list_cvt[1][2]), nn.ReLU(inplace=True))
        self.squeeze2s   = nn.Sequential(nn.Conv2d(list_cvt[0][1], list_cvt[1][1], kernel_size=1), nn.BatchNorm2d(list_cvt[1][1]), nn.ReLU(inplace=True), nn.Conv2d(list_cvt[1][1], list_cvt[1][1], kernel_size=list_para[0][1], padding=list_para[1][1]), nn.BatchNorm2d(list_cvt[1][1]), nn.ReLU(inplace=True))
        self.squeeze1s   = nn.Sequential(nn.Conv2d(list_cvt[0][0], list_cvt[1][0], kernel_size=1), nn.BatchNorm2d(list_cvt[1][0]), nn.ReLU(inplace=True), nn.Conv2d(list_cvt[1][0], list_cvt[1][0], kernel_size=list_para[0][0], padding=list_para[1][0]), nn.BatchNorm2d(list_cvt[1][0]), nn.ReLU(inplace=True))
        
        self.decoder = Decoder(list_cvt, list_para)
        self.icm = ICM(list_cvt, list_para)
        self.ecm = ECM(list_cvt, list_para)
        self.merge = MergeLayer(list_cvt, list_para)

        self.out_edge1 = nn.Conv2d(list_cvt[1][0], 1, kernel_size=list_para[0][0], padding=list_para[1][0])
        self.out_edge2 = nn.Conv2d(list_cvt[1][1], 1, kernel_size=list_para[0][1], padding=list_para[1][1])
        self.out_edge3 = nn.Conv2d(list_cvt[1][2], 1, kernel_size=list_para[0][2], padding=list_para[1][2])
        self.out_edge4 = nn.Conv2d(list_cvt[1][3], 1, kernel_size=list_para[0][3], padding=list_para[1][3])
        self.out_edge5 = nn.Conv2d(list_cvt[1][4], 1, kernel_size=list_para[0][4], padding=list_para[1][4])

        self.out_sal1 = nn.Conv2d(list_cvt[1][0], 1, kernel_size=list_para[0][0], padding=list_para[1][0])
        self.out_sal2 = nn.Conv2d(list_cvt[1][1], 1, kernel_size=list_para[0][1], padding=list_para[1][1])
        self.out_sal3 = nn.Conv2d(list_cvt[1][2], 1, kernel_size=list_para[0][2], padding=list_para[1][2])
        self.out_sal4 = nn.Conv2d(list_cvt[1][3], 1, kernel_size=list_para[0][3], padding=list_para[1][3])
        self.out_sal5 = nn.Conv2d(list_cvt[1][4], 1, kernel_size=list_para[0][4], padding=list_para[1][4])

        self.outf1 = nn.Sequential(nn.Conv2d(list_cvt[2][0], list_cvt[1][0], kernel_size=list_para[0][0], padding=list_para[1][0]), nn.BatchNorm2d(list_cvt[1][0]), nn.ReLU(inplace=True), nn.Conv2d(list_cvt[1][0], 1, kernel_size=list_para[0][0], padding=list_para[1][0]))
        self.outf = nn.Sequential(nn.Conv2d(list_cvt[2][5], list_cvt[1][4], kernel_size=list_para[0][2], padding=list_para[1][2]), nn.BatchNorm2d(list_cvt[1][4]), nn.ReLU(inplace=True), nn.Conv2d(list_cvt[1][4], 1, kernel_size=list_para[0][2], padding=list_para[1][2]))

        self.initialize()

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape
        
        out1, out2, out3, out4, out5 = self.bkbone(x)
        out1s, out2s, out3s, out4s, out5s = self.squeeze1s(out1), self.squeeze2s(out2), self.squeeze3s(out3), self.squeeze4s(out4), self.squeeze5s(out5)
        convs1, convs2, convs3, convs4, convs5 = self.decoder(out1s, out2s, out3s, out4s, out5s)
        edge_feature, sal_feature = self.icm(out1s, out2s, out3s, out4s, out5s, convs1, convs2, convs3, convs4, convs5)
        ecm_feature = self.ecm(out1s, out2s, out3s, out4s, out5s, convs1, convs2, convs3, convs4, convs5)
        merge_feature = self.merge(edge_feature, sal_feature, ecm_feature)

        up_edge, up_sal_f, up_sal = [], [], []
        up_sal_f.append(F.interpolate(self.outf1(merge_feature[0]), shape, mode='bilinear', align_corners=True))
        up_sal_f.append(F.interpolate(self.outf1(merge_feature[1]), shape, mode='bilinear', align_corners=True))
        up_sal_f.append(F.interpolate(self.outf1(merge_feature[2]), shape, mode='bilinear', align_corners=True))
        up_sal_f.append(F.interpolate(self.outf1(merge_feature[3]), shape, mode='bilinear', align_corners=True))
        up_sal_f.append(F.interpolate(self.outf1(merge_feature[4]), shape, mode='bilinear', align_corners=True))
        up_sal_f.append(F.interpolate(self.outf(merge_feature[5]), shape, mode='bilinear', align_corners=True))

        up_sal.append(F.interpolate(self.out_sal5(sal_feature[4]), shape, mode='bilinear', align_corners=True))
        up_sal.append(F.interpolate(self.out_sal4(sal_feature[3]), shape, mode='bilinear', align_corners=True))
        up_sal.append(F.interpolate(self.out_sal3(sal_feature[2]), shape, mode='bilinear', align_corners=True))
        up_sal.append(F.interpolate(self.out_sal2(sal_feature[1]), shape, mode='bilinear', align_corners=True))
        up_sal.append(F.interpolate(self.out_sal1(sal_feature[0]), shape, mode='bilinear', align_corners=True))

        up_edge.append(F.interpolate(self.out_edge5(edge_feature[4]), shape, mode='bilinear', align_corners=True))
        up_edge.append(F.interpolate(self.out_edge4(edge_feature[3]), shape, mode='bilinear', align_corners=True))
        up_edge.append(F.interpolate(self.out_edge3(edge_feature[2]), shape, mode='bilinear', align_corners=True))
        up_edge.append(F.interpolate(self.out_edge2(edge_feature[1]), shape, mode='bilinear', align_corners=True))
        up_edge.append(F.interpolate(self.out_edge1(edge_feature[0]), shape, mode='bilinear', align_corners=True))

        return up_edge, up_sal, up_sal_f

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)