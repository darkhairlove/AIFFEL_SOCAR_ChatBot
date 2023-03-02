from cv_lib import (
    os, random
    , np, pd
    , torch, transforms
    , cv2, ast)
from Utils import main_path, resize

class Datasets(torch.utils.data.Dataset):
    def __init__(self, path, train = 'train', transforms = None, forcsv = False):  # forcsv는 csv로 변환할지 여부 판단
        self.forcsv = forcsv
        self.train = train
        self.data_path = os.path.join(path, train)
        self.common_transforms = transforms['common']   # transforms 
        self.input_transforms = transforms['input']    # augmentations
        
        label_path = os.path.join(self.data_path, 'masks')
        input_path = os.path.join(self.data_path, 'images')
        
        self.label_path = label_path
        self.input_path = input_path
        
        lst_label = os.listdir(label_path)  # 이미지 파일명 리스트
        lst_input = os.listdir(input_path)

        self.lst_label = lst_label
        self.lst_input = lst_input
        
        
    def __len__(self):
        return len(self.lst_label)  # label 목록 길이 반환
    
    def __getitem__(self, index):
        if self.forcsv:  # 데이터 증강을 위한 csv파일 생성
            label_path = os.path.join(self.label_path, self.lst_label[index])
            input_path = os.path.join(self.input_path, self.lst_input[index])


            label = cv2.imread(label_path)
            input = cv2.imread(input_path)

            # 파손 여부 
            damage = label.sum()   # mask의 픽셀이 하나라도 존재하면 파손 데이터
            if damage == 0:
                isdamaged = 0     # 마스크에 합이 0 이면 미파손 차량이고
            else:
                isdamaged = 1     # 마스크에 합이 0 초과이면 파손 차량으로 분류
            
            data = {'input_path': input_path,'label_path':label_path,'isdamaged':isdamaged}
            
            return data
        
        else:  # 일반적인 학습을 위한 datasets출력
            label_path = os.path.join(self.label_path, self.lst_label[index])
            input_path = os.path.join(self.input_path, self.lst_input[index])


            label = cv2.imread(label_path)[:,:,:1]
            input = cv2.imread(input_path)
            
            # crop
            R = random.random()  # random 확률 발생
            if (self.train=="train")&(label.sum()>0)&(R>0.5):
                y0 = np.where(np.argmax(label, axis=1)>0)[0][0]  # axis=0은 세로
                y1 = np.where(np.argmax(label, axis=1)>0)[0][-1]  # axis=0은 세로

                x0 = np.where(np.argmax(label, axis=0)>0)[0][0] # 이게 x축 시작점
                x1 = np.where(np.argmax(label, axis=0)>0)[0][-1] # 이게 x축  마지막점

                y_term0 = int(y0/4)
                y_term1 = int((label.shape[0]-y1)/4)
                x_term0 = int(x0/4)
                x_term1 = int((label.shape[1]-x1)/4)
                
                input = input[y0-y_term0:y1+y_term1, x0-x_term0:x1+x_term1,:]
                label = label[y0-y_term0:y1+y_term1, x0-x_term0:x1+x_term1,:]
                       

            # normalized   # 픽셀 값을 0~1 사이로 정규화 하기 위한 코드
            label = label/255.0
            label0 = 1-label
            label = np.concatenate((label0, label), axis=2)
            input = input/255.0
    
                
                

            if self.common_transforms:  # transforms가 존재하면
                label = self.common_transforms(label)
                input = self.common_transforms(input)
            # validation이나 test의 경우 augmentation을 할 필요가 없다.      
            if self.input_transforms:
                input = self.input_transforms(input)
            
            label = np.ceil(label)
            
            
                
            
            # shape이 (256,256)인 경우 -> (256,256,1)으로 변경; 흑백의 경우가 해당
            if label.ndim == 2: 
                label = label[:, :, np.newaxis]
            if input.ndim == 2:
                input = input[:, :, np.newaxis]

            # data라는 dict에 저장



            data = {'input_path': input_path,'label_path':label_path,'input':input.float(), 'label':label.float()}


            return data

# csv형식에서 dataset으로 변환하기 위한 class
# csv 형식 : input_path label_path 피쳐가 필수적으로 존재해야함

class DatasetsForCsv(torch.utils.data.Dataset):
    def __init__(self, df_name, transforms = None):
        self.df = pd.read_csv(os.path.join(main_path, df_name))  # csv 경로
        self.input_path = self.df.input_path 
        self.label_path = self.df.label_path
        
        self.commom_transforms = transforms['common']   # input transform 
        self.input_transforms = transforms['input']    # label transform
        
    def __len__(self):
        return len(self.label_path)  # label 목록 길이 반환
    
    def __getitem__(self, index):
        label_path = self.label_path[index]
        input_path = self.input_path[index]
        
        
        input = cv2.imread(input_path)
        label = cv2.imread(label_path)[:,:,:1]  # imageio보다 cv2의 속도가 빠름
        
        

        
        # image crop
        R = random.random()  # random 확률 발생
        if (self.train=="train")&(label.sum()>0)&(R>0.5):
            y0 = np.where(np.argmax(label, axis=1)>0)[0][0]  # axis=0은 세로
            y1 = np.where(np.argmax(label, axis=1)>0)[0][-1]  # axis=0은 세로

            x0 = np.where(np.argmax(label, axis=0)>0)[0][0] # 이게 x축 시작점
            x1 = np.where(np.argmax(label, axis=0)>0)[0][-1] # 이게 x축  마지막점

            y_term0 = int(y0/4)
            y_term1 = int((label.shape[0]-y1)/4)
            x_term0 = int(x0/4)
            x_term1 = int((label.shape[1]-x1)/4)

            input = input[y0-y_term0:y1+y_term1, x0-x_term0:x1+x_term1,:]
            label = label[y0-y_term0:y1+y_term1, x0-x_term0:x1+x_term1,:]        
            
        # normalized   # 픽셀 값을 0~1 사이로 정규화 하기 위한 코드
        label = label/255.0
        label0 = 1-label
        label = np.concatenate((label0, label), axis=2)
        label = np.ceil(label)
        input = input/255.0
        
        if self.commom_transforms:
            label = self.commom_transforms(label)
            input = self.commom_transforms(input)
            
        # validation이나 test의 경우 augmentation을 할 필요가 없다.      
        if self.input_transforms:
            input = self.input_transforms(input)
                      
        
        label = np.ceil(label)
        
        # 흑백 사진의 경우 채널 차원 생성
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
        
        # data라는 dict에 저장
        data = {'input':input.float(), 'label':label.float()}
        

        return data
    
    
# transforms
# random 변환 요소를 image, mask 동일하게 적용
class RandomChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = random.choice(self.transforms)

    def __call__(self, img):
        return self.t(img)
    
train_transforms = {
    'common':transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize),
        transforms.RandomChoice([       
            transforms.RandomHorizontalFlip(), 
            transforms.RandomVerticalFlip()
        ])
    ]), 
    'input': transforms.Compose([
        transforms.ColorJitter(brightness=0.2, constrast=0.2) # (min, max) = (1-x, 1)
    ])
}

val_transforms = {
    'common':transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize),
    ]),
    'input':False
}