from src.Utils import (random, np, os, torch, model_path, main_path, gc, set_seed, lr,
                       dent_path, scratch_path, spacing_path,
                       dent_ckpt_path, scratch_ckpt_path, spacing_ckpt_path,
                       augmentation, loss_fn, train_tf, val_tf, train_continue, batch_size, num_workers, num_epoch,
                       dent_model, scratch_model, spacing_model,ast, config)
from src.Models import Unet, Pspnet
from src.Datasets import Datasets, DatasetsForCsv, train_transforms, val_transforms
from src.Evaluation import Evaluation
from src.Train import Trainer
import time
from src.Check_Output import Check_Output



if __name__ =="__main__":
    ## set seed
    def set_seed(seed:int):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore
    set_seed(42)


    models = ['dent', 'scratch', 'spacing']
    ckpt_paths = [dent_ckpt_path, scratch_ckpt_path, spacing_ckpt_path]
    for model, ckpt_path in zip(models, ckpt_paths):
        start = time.time()  # 시작시간 저장
        print(f'Start {model} model training...')  
        path = os.path.join(main_path, model)  # 모델 경로 지정
        
        train_df = None
        if ast.literal_eval(config['AUGMENTATION']["augmentation"]):    # augmentation True로 하면
            train_df = model+'_train.csv'    # csv로 저장된 파일을 
            train_transforms = train_transforms
            val_transforms = val_transforms
        
        trainer = Trainer(path = path,
                          net=Pspnet(),  # 모델을 하이퍼 파라미터로 추가
                          train_df_name=train_df,
                          ckpt_path=ckpt_path,
                          loss_fn=loss_fn,
                          train_transforms=train_tf, 
                          val_transforms=val_tf,
                          train_continue=train_continue,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          num_epoch=num_epoch)
        net = trainer.train()
        end = time.time()
        print(f"time : {(end-start)//60}분 {round((end-start)%60)}초")  # 현재 시간 - 시작 시간
        print()
        
        net, optim, _ = trainer.load_net()
        torch.save({'net':net.state_dict(), "optim":optim.state_dict()},
                   os.path.join(model_path, model+f'{ckpt_path[-2:]}.pth'))  # 모델을 파라미터만 저장하기로
        print(f"Final Model Save {model}{ckpt_path[-2:]}.pth")
        
        gc.collect()
        torch.cuda.empty_cache()

        del trainer, net, optim    
        
    
    for model,ckpt_path in zip(models,ckpt_paths):
        print()
        print()
        print(f'Start {model} model testing...')  
        ckpt_path = ckpt_path
        data_path = os.path.join(main_path, model)
        test = Evaluation(ckpt_path=ckpt_path, 
                          net = Pspnet(),  # 모델 선언
                          data_path=data_path,
                          val_transforms=val_transforms,
                          batch_size=batch_size, 
                          num_workers=num_workers,
                          lr=lr)
        test.validation()

        gc.collect()
        torch.cuda.empty_cache()

        del test
        
