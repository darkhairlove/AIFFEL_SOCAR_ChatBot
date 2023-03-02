from cv_lib import losses, ast
from Models import Unet
from Datasets import train_transforms, val_transforms
'''
parameters, path 모음
'''

from cv_lib import os, random, np, nn, torch, config

seed = 42

# path
main_path = config['DEFAULT']["data_folder"]

# 파손종류별 데이터 경로
dent_path = os.path.join(main_path, 'dent')
scratch_path = os.path.join(main_path, 'scratch')
spacing_path = os.path.join(main_path, 'spacing')

# check point 저장 경로
# 모델별로 각각 저장
test_num = config['TEST_NUM']['test_num']

ckpt_path = os.path.join(main_path, 'train_ckpt')
dent_ckpt_path = os.path.join(ckpt_path, 'dent_ckpt'+test_num)
scratch_ckpt_path = os.path.join(ckpt_path, 'scratch_ckpt'+test_num)
spacing_ckpt_path = os.path.join(ckpt_path, 'spacing_ckpt'+test_num)

# 모델 저장 경로
model_path = config['DEFAULT']["model_folder"]

# 모델 명 
dent_model = 'dent'+test_num
scratch_model = 'scratch'+test_num
spacing_model = 'spacing'+test_num


# 모델 파라미터
lr = float(config['HYPER_PARAM']["lr"])
batch_size = int(config['HYPER_PARAM']["batch_size"])
num_epoch = int(config['HYPER_PARAM']["num_epoch"])
num_workers = int(config['HYPER_PARAM']["num_workers"])
resize = ast.literal_eval(config['AUGMENTATION']["resize"])
train_continue = ast.literal_eval(config['HYPER_PARAM']["train_continue"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transforms 
augmentation = ast.literal_eval(config['AUGMENTATION']["augmentation"])
train_tf = train_transforms
val_tf = val_transforms


'''
성능 평가에 사용되는 모듈 (미완성)
loss function과 evaluation function 둘 모두로 사용가능

moU loss: mean-IoU.  https://gaussian37.github.io/vision-segmentation-miou/
       (예측한 영역)n(실제 영역)/(예측한 영역)U(실제 영역)   
       -> 현재 계산 train에서 계산 결과가 nan이 나오는 문제가 있음
Dice loss : mou와 유사한 loss function. 차이점은 TP에 더 큰 가중치를 주는 loss function
Focal : 라벨에 negative 영역이 너무 많아서 BCE가 positive 영역을 잘 학습하지 못하는 문제를 해결하기 위해 등장한 loss function.
        FP, FN loss에 가중치를 줘서 배경부분에 과도하게 피팅되는 문제를 해결함.
JaccardLoss : IoU와 동일
'''

# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()
#         self.BCE = nn.BCELoss().to(device) 

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = torch.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         # BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
#         # BCE = torch.nn.BCELoss(inputs, targets, reduction='mean')
#         BCE = self.BCE(inputs, targets)
#         Dice_BCE = BCE + dice_loss
        
#         return Dice_BCE

# # Dice + Focal loss     
# class DiceFocal(nn.Module):
#     def __init__(self,weight=None, size_average=True):
#         super(DiceFocal, self).__init__()
#         self.Focal = losses.FocalLoss(mode='binary').to(device)
#         self.Dice = losses.DiceLoss(mode='binary', from_logits=True).to(device)
        
#     def forward(self, inputs, targets):
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         focal = self.Focal(inputs, targets)
#         dice = self.Dice(inputs, targets)
        
#         dice_focal = focal + dice
        
#         return dice_focal
        
    

## set seed
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

# # model load function 
# # 모델 파라미터가 존재하면 불러오고 없으면 새로 unet 생성
# def load(model):
#     try:
#         net = Unet()
#         net.load_state_dict(torch.load(os.path.join(model_path, model+'.pt')))
#         return net

#     except:
#         net = Unet()
#         return net
    