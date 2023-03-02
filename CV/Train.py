from cv_lib import (os, np, losses, nn, torch, DataLoader)
from Utils import (
    device
    , num_epoch, num_workers
    , model_path
    , batch_size, lr
)
from Datasets import Datasets, DatasetsForCsv, train_transforms, val_transforms
from Models import Unet


class Trainer():
    def __init__(self, 
                 net,    # 모델불러오기
                 ckpt_path,
                 path=None,
                 train_df_name=None,  # csv 파일로 존재할 때
                 train_continue=False,
                 train_transforms=train_transforms,
                 val_transforms=val_transforms,
                 batch_size=batch_size, 
                 num_workers=num_workers, 
                 num_epoch=num_epoch, 
                 lr=lr):
        
        # CSV 파일일때
        if train_df_name:
            self.train_datasets = DatasetsForCsv(df_name = train_df_name, transforms=train_transforms)  # csv file을 활용할때
            self.valid_datasets = Datasets(path, train = 'valid', transforms = val_transforms, forcsv = False)
        
        # 이미지 경로일때    ex) dent_path
        else:
            self.train_datasets = Datasets(path, train = 'train', transforms = train_transforms, forcsv = False)
            self.valid_datasets = Datasets(path, train = 'valid', transforms = val_transforms, forcsv = False)
        
        self.train_continue = train_continue
        
        self.net = net
        self.ckpt_path=ckpt_path
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_workers = num_workers
        self.lr = lr
        
        # 손실함수 정의
        self.BCEL_loss = nn.BCEWithLogitsLoss().to(device)   # 이진 분류 모델
        self.Focal_loss = losses.FocalLoss(mode='binary').to(device)
        self.Dice_loss = losses.DiceLoss(mode='binary', from_logits=True).to(device)
        self.Jaccard_loss = losses.JaccardLoss(mode='binary', from_logits=True).to(device)  
        
        
        self.loss_fn = losses.DiceLoss(mode='binary', from_logits=True).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.5)
              
          
    # 데이터 로더 
    # datasets의 경우 미리 만들어서 넣어야 함
    def get_loader(self):  
        train_loader = DataLoader(self.train_datasets,                             
                                  batch_size = self.batch_size, 
                                  shuffle = True, 
                                  num_workers = self.num_workers
                                 )
        
        
        valid_loader = DataLoader(self.valid_datasets, 
                                  batch_size = self.batch_size,
                                  shuffle = True,
                                  num_workers = self.num_workers
                                 )
    
        
        return train_loader, valid_loader
        
    def train(self, train_continue = False):
        
        train_loader, self.valid_loader = self.get_loader()
        best_eval = np.inf   # best_eval = BCEWithLogitsLoss + DiceLoss + FoclaLoss +
        st_epoch = 0
        cnt = 0
        
        if self.train_continue:  # 이어서 학습하기
            self.net, self.optim, st_epoch = self.load_net()
            self.num_epoch = self.num_epoch + st_epoch  # 시작에폭과 끝 에폭을 그만큼 늘림
            print("이어서 학습")

        
        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            self.net.train()
            loss_arr = []

            for batch, data in enumerate(train_loader, 1):

                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = self.net(input)

                # backward pass 
                self.optim.zero_grad()
                loss = self.loss_fn(output, label)
                loss.backward()

                self.optim.step()

                # loss function 계산
                loss_arr +=[loss.item()]
                del label, input
                

            print(f"TRAIN: EPOCH %04d/%04d | {self.loss_fn} %.4f" %
                 (epoch, self.num_epoch, np.mean(loss_arr)))
            
            torch.cuda.empty_cache()
            
            self.lr_scheduler.step()  # scheduler step
           
            BCE, Focal, Dice, Jaccard = self.validation(epoch=epoch)

            new_eval = Focal + Dice + Jaccard
            
            if new_eval < best_eval:
                best_eval = new_eval
                self.save_net(epoch, best_eval)  # 모델 저장 경로에 저장
                cnt=0  # 카운트 초기화
     
            else:
                cnt +=1
      
            
            # early stop
            if cnt==10:
                break

            # 메모리 정리    
            torch.cuda.empty_cache()
            print('-------------------------------------------------')   
                
        return self.net
            
    def validation(self, epoch): # mode를 test로 바꾸면 test 
        
        
        with torch.no_grad():
            self.net.eval()
            
            # loss 값 저장
            loss_bce_arr = []
            loss_focal_arr = []
            loss_dice_arr = []
            loss_jaccard_arr = []            
            
            for batch, data in enumerate(self.valid_loader, 1):
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
                
            print(f"VALIDATION: EPOCH %04d/%04d | {self.BCEL_loss} %.4f | {self.Focal_loss} %.4f | {self.Dice_loss} %.4f | mIoU %.4f" %
                 (epoch, self.num_epoch, np.mean(loss_bce_arr), np.mean(loss_focal_arr), np.mean(loss_dice_arr), 1-np.mean(loss_jaccard_arr)))        
       
    
        return np.mean(loss_bce_arr), np.mean(loss_focal_arr), np.mean(loss_dice_arr), np.mean(loss_jaccard_arr)
    
            
    def save_net(self, epoch, best_eval):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
            
        torch.save({'net':self.net.state_dict(), 'optim':self.optim.state_dict()},
                   f"{self.ckpt_path}/model_{epoch}.pth")  # 모델과 optim을 모두 저장  # besteval 삭제
        print('Model Saved')
        
    def load_net(self):
        if not os.path.exists(self.ckpt_path): 
            epoch = 0
            return self.net, self.optim, epoch
        
        ckpt_lst = os.listdir(self.ckpt_path)  # ckpt_path의 파일명 리스트
        ckpt_lst.sort(key = lambda f : int(''.join(filter(str.isdigit, f))))  # 뒷부분의 epoch을 기준으로 정렬     
        dict_model = torch.load(f'{self.ckpt_path}/{ckpt_lst[-1]}', map_location = device) # ckpt_path에서 마지막 모델 load

        self.net.load_state_dict(dict_model['net'])
        self.optim.load_state_dict(dict_model['optim'])
        epoch = int(ckpt_lst[-1].split('_')[2].split('.pth')[0]) # 모델명에서 숫자만 가져옴
        return self.net, self.optim, epoch