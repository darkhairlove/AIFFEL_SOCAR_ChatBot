from Datasets import Datasets, val_transforms
from Models import Unet
from Utils import device, dent_path, scratch_path, spacing_path, model_path, resize
from cv_lib import DataLoader, cv2, plt, torch, nn, os

class Check_Output():
    def __init__(self,
                 net, 
                 dent_model, # 모델 path
                 scratch_model, 
                 spacing_model,
                 train = 'test',
                 dent_path=dent_path,     # 데이터 path
                 scratch_path=scratch_path, 
                 spacing_path=spacing_path,):
        
        self.dent_datasets = Datasets(dent_path, train = train, transforms = val_transforms, forcsv = False)
        self.scratch_datasets = Datasets(scratch_path, train = train, transforms = val_transforms, forcsv = False)
        self.spacing_datasets = Datasets(spacing_path, train = train, transforms = val_transforms, forcsv = False)
        
        dent = net
        self.dent_model = net.torch.load(os.path.join(model_path, dent_model+'.pth'), map_location=device)['net']   # 저장되어있는 모델 load
        
        scratch = net
        self.scratch_model = net.torch.load(os.path.join(model_path, scratch_model+'.pth'), map_location=device)['net']  
        
        spacing = net
        self.spacing_model = net.torch.load(os.path.join(model_path, spacing_model+'.pth'), map_location=device)['net']
        

    
    def get_loader(self):   
        dent_loader = DataLoader(self.dent_datasets, 
                                  batch_size = 1,
                                  shuffle = True,  # 이미지는 랜덤 순서로 출력
                                  num_workers = 4
                                 )
        scratch_loader = DataLoader(self.scratch_datasets, 
                                  batch_size = 1,
                                  shuffle = True,
                                  num_workers = 4
                                 )
        spacing_loader = DataLoader(self.spacing_datasets, 
                                  batch_size = 1,
                                  shuffle = True,
                                  num_workers = 4
                                 )
        return dent_loader, scratch_loader, spacing_loader
    
    def check_output(self, max_count = 1):   # max_count 값에 따라 한 종류의 파손에서 몇개의 사진을 출력할지 결정
        dent_loader, scratch_loader, spacing_loader = self.get_loader()
        loaders = [dent_loader, scratch_loader, spacing_loader]
        labels = ['dent', 'scratch', 'spacing']
        models = [self.dent_model, self.scratch_model, self.spacing_model]
        m = nn.Threshold(0.3, 0) 
        
        for label, loader in zip(labels,loaders):
            count = 0  # 필요한 횟수만큼만 돌리자
            print(f'{label} imgaes----------------')
            for data in loader:
                fig, ax = plt.subplots(1,4, figsize = (10,30))
                img = cv2.imread(data['input_path'][0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, resize)

                mask = cv2.imread(data['label_path'][0])
                mask = cv2.resize(mask, resize)

                ax[0].imshow(img, alpha=0.5)
                ax[0].imshow(mask, alpha=0.5)
                ax[0].axis('off')
                
                for i, (label, model) in enumerate(zip(labels, models),1):
                    output = model(data['input'].to(device))
                    output = torch.sigmoid(output)
                    output = m(output)
                    
                    
                    img_output = torch.argmax(output, dim = 1).detach().cpu().numpy()
                    img_output = img_output.transpose([1,2,0])
                    
                    ax[i].set_title(label)
                    ax[i].imshow(img, alpha = 0.5)
                    ax[i].imshow(img_output, cmap = cm.gray, alpha = 0.5)
                    ax[i].axis('off')
                fig.set_tight_layout(True)
                plt.show()
                
                count +=1
                if count >= max_count:
                    break
                
