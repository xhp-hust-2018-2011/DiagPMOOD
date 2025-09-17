import torch
import torch.nn as nn

# get the fc layer of the network
def get_fclayer(net):
    net_name = net.__class__.__name__
    if net_name in  ['ResNet','ShuffleNetV2','RegNet','GoogLeNet','Inception3']:
        fclayer = net.fc
    elif net_name in ['VisionTransformer']:
        fclayer = net.heads.head
    elif net_name in ['MobileNetV2','SqueezeNet','MNASNet']:
        fclayer = net.classifier[1]
    elif net_name in ['ConvNeXt']:
        fclayer = net.classifier[2]
    elif net_name in ['SwinTransformer']:
        fclayer = net.head
    elif net_name in ['DenseNet']:
        fclayer = net.classifier
    elif net_name in ['MobileNetV3','VGG','MaxVit']:
        fc_layer = net.classifier # this is a several layer fc, may also contain avgpool
    return fclayer

# initialize linear or convolutonal layer
def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)

# imglist file dataset