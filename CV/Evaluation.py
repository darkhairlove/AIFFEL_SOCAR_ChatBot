from cv_lib import np, nn, torch, DataLoader, losses, os
from Utils import (device, batch_size, num_epoch, 
                   lr, num_workers, DiceBCELoss)
from Datasets import Datasets, train_transforms, val_transforms
from Models import Unet

class Evaluation():
    def __init__(self, 
                 net,
                 ckpt_path,
                 data_path,
                 val_transforms=val_transforms,
                 batch_size=batch_size, 
                 num_workers=num_workers, 
                 lr=lr):
        
        self.val_datasets = Datasets(data_path, train = 'test', transforms = val_transforms, forcsv = False)
        
        self.val_transforms = val_transforms
        
        self.ckpt_path=ckpt_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.net = net
        
        
        # 손실함수 정의
        self.BCEL_loss = nn.BCEWithLogitsLoss().to(device)   # 이진 분류 모델
        self.Focal_loss = losses.FocalLoss(mode='binary').to(device)
        self.Dice_loss = losses.DiceLoss(mode='binary', from_logits=True).to(device)
        self.Jaccard_loss = losses.JaccardLoss(mode='binary', from_logits=True).to(device)  
        

              
          
    # 데이터 로더 
    # datasets의 경우 미리 만들어서 넣어야 함
    def get_loader(self):   
        val_loader = DataLoader(self.val_datasets, 
                                  batch_size = self.batch_size,
                                  shuffle = False,
                                  num_workers = self.num_workers
                                 )
    
        
        return val_loader

            
    def validation(self): # mode를 test로 바꾸면 test 
        self.net = self.load_net()  # 가장 최신 모델 불러오기
        
        val_loader = self.get_loader()
        
        with torch.no_grad():
            self.net.eval()
            
            # loss 값 저장
            loss_bce_arr = []
            loss_focal_arr = []
            loss_dice_arr = []
            loss_jaccard_arr = []            
            
            for batch, data in enumerate(val_loader, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                
                output = self.net(input)
                
                # 손실함수 계산하기
                loss_bce = self.BCEL_loss(output, label)
                loss_focal = self.Focal_loss(output, label)
                loss_dice = self.Dice_loss(output, label)
                loss_jaccard = self.Jaccard_loss(output, label)

                loss_bce_arr += [loss_bce.item()]
                loss_focal_arr +=[loss_focal.item()]
                loss_dice_arr +=[loss_dice.item()]
                loss_jaccard_arr += [loss_jaccard.item()]    
                
            print(f"TEST: | {self.BCEL_loss} %.4f | {self.Focal_loss} %.4f | {self.Dice_loss} %.4f | {self.Jaccard_loss} %.4f" %
                 (np.mean(loss_bce_arr), np.mean(loss_focal_arr), np.mean(loss_dice_arr), np.mean(loss_jaccard_arr)))
            print('-------------------------------------------------')           
#         return np.mean(loss_bce_arr), np.mean(loss_focal_arr), np.mean(loss_dice_arr), np.mean(loss_jaccard_arr)
            
    
    def load_net(self):        
        ckpt_lst = os.listdir(self.ckpt_path)  # ckpt_path의 파일명 리스트
        ckpt_lst.sort(key = lambda f : int(''.join(filter(str.isdigit, f))))  # 뒷부분의 epoch을 기준으로 정렬
        
        dict_model = torch.load(f'{self.ckpt_path}/{ckpt_lst[-1]}', map_location = device) # ckpt_path에서 마지막 모델 load
        
        self.net.load_state_dict(dict_model['net'])
        return self.net
        
            
