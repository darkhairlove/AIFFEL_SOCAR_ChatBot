from cv_lib import nn, smp
from Utils import device
'''
추후에 구조에 대한 개선 고민 필요
1. sigmoid function 추가 여부
2. encoder = efficientnet 사용 모델과의 비교
'''

'''
- num_classes = 탐지할 객체 수. 현재 모델의 경우 배경과 파손부위 2개로만 탐지
- encoder = resnet34
- pre_weight = imagenet
- in_channels = input data의 channel 
- Batch_Norm2d 적용, (3)은 채널수를 의미
'''

class Unet(nn.Module):
    def __init__(self, num_classes = 2, encoder = 'resnet34', pre_weight = 'imagenet'): # num_class가 뭘까?
        super().__init__()
        self.model = smp.Unet(classes = num_classes,
                              encoder_name = encoder, 
                              encoder_weights = pre_weight,
                              in_channels = 3).to(device)
        self.input_batchnorm = nn.BatchNorm2d(3).to(device)

        
    def forward(self, x):
        x = self.input_batchnorm(x)
        y = self.model(x)
        
        encoder_weights = "imagenet"
        return y
    
class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes = 2, encoder = 'efficientnet-b7', pre_weight = 'imagenet'): # num_class가 뭘까?
        super().__init__()
        self.model = smp.UnetPlusPlus(classes = num_classes,
                              encoder_name = encoder, 
                              encoder_weights = pre_weight,
                              in_channels = 3, 
                              ).to(device)

        
    def forward(self, x):
        y = self.model(x)
        
        encoder_weights = "imagenet"
        return y