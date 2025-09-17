from torchvision.models.resnet import Bottleneck, ResNet
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(ResNet):
    def __init__(self,
                 block=Bottleneck,
                 layers=[3, 4, 6, 3],
                 num_classes=1000):
        super(ResNet50, self).__init__(block=block,
                                       layers=layers,
                                       num_classes=num_classes)
        self.feature_size = 2048

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)

        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    # return the pixel-wise output
    def pixel_forward(self, x, return_feature=False, return_feature_list=False):
        feature = self.relu(self.bn1(self.conv1(x)))
        feature = self.maxpool(feature)
        feature = self.layer1(feature)
        feature = self.layer2(feature)
        feature = self.layer3(feature)
        feature = self.layer4(feature)
        feature5 = feature
        feature = self.avgpool(feature)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        # feature 4 need to be reshape
        B,C,H,W = feature5.size()
        feature5 = feature5.permute([0,2,3,1]).view(B,H*W,C)
        pixel_logits_cls = self.fc(feature5) 

        # feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return_dict =  {'avg_feature':feature, 
                            'pixel_feature':feature5,
                            'pixel_logit_cls':pixel_logits_cls,
                            }
            return logits_cls, return_dict # this pixel-wise things
        # elif return_feature_list:
        #     return logits_cls, feature_list
        else:
            return logits_cls


    def forward_threshold(self, x, threshold):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature1 = self.maxpool(feature1)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def intermediate_forward(self, x, layer_index):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        if layer_index == 1:
            return out

        out = self.layer2(out)
        if layer_index == 2:
            return out

        out = self.layer3(out)
        if layer_index == 3:
            return out

        out = self.layer4(out)
        if layer_index == 4:
            return out

        raise ValueError

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc

    
