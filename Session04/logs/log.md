```
[2022-10-04 16:16:20,528][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-10-04 16:16:20,540][src.utils.utils][INFO] - Printing config tree with Rich! <cfg.extras.print_config=True>
[2022-10-04 16:16:20,541][src.utils.rich_utils][WARNING] - Field 'logger' not found in config. Skipping 'logger' config printing...
CONFIG
├── datamodule
│   └── _target_: src.datamodules.cifar_datamodule.CIFARDataModule              
│       data_dir: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/data/             
│       batch_size: 10000                                                       
│       train_val_test_split:                                                   
│       - 45000                                                                 
│       - 5000                                                                  
│       - 10000                                                                 
│       num_workers: 16                                                         
│       pin_memory: false                                                       
│                                                                               
├── model
│   └── _target_: src.models.cifar_timm_module.CIFARLitModule                   
│       model_name: resnet18                                                    
│       optimizer:                                                              
│         _target_: torch.optim.Adam                                            
│         _partial_: true                                                       
│         lr: 0.001                                                             
│         weight_decay: 0.0                                                     
│                                                                               
├── callbacks
│   └── model_checkpoint:                                                       
│         _target_: pytorch_lightning.callbacks.ModelCheckpoint                 
│         dirpath: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/train/runs/2
│         filename: epoch_{epoch:03d}                                           
│         monitor: val/acc                                                      
│         verbose: false                                                        
│         save_last: true                                                       
│         save_top_k: 1                                                         
│         mode: max                                                             
│         auto_insert_metric_name: false                                        
│         save_weights_only: false                                              
│         every_n_train_steps: null                                             
│         train_time_interval: null                                             
│         every_n_epochs: null                                                  
│         save_on_train_epoch_end: null                                         
│       early_stopping:                                                         
│         _target_: pytorch_lightning.callbacks.EarlyStopping                   
│         monitor: val/acc                                                      
│         min_delta: 0.0                                                        
│         patience: 100                                                         
│         verbose: false                                                        
│         mode: max                                                             
│         strict: true                                                          
│         check_finite: true                                                    
│         stopping_threshold: null                                              
│         divergence_threshold: null                                            
│         check_on_train_epoch_end: null                                        
│       model_summary:                                                          
│         _target_: pytorch_lightning.callbacks.RichModelSummary                
│         max_depth: -1                                                         
│       rich_progress_bar:                                                      
│         _target_: pytorch_lightning.callbacks.RichProgressBar                 
│                                                                               
├── trainer
│   └── _target_: pytorch_lightning.Trainer                                     
│       default_root_dir: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/train
│       min_epochs: 1                                                           
│       max_epochs: 10                                                          
│       accelerator: gpu                                                        
│       devices: 1                                                              
│       deterministic: false                                                    
│                                                                               
├── paths
│   └── root_dir: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab                   
│       data_dir: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/data/             
│       log_dir: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/              
│       output_dir: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/train/runs/
│       work_dir: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab                   
│                                                                               
├── extras
│   └── ignore_warnings: false                                                  
│       enforce_tags: true                                                      
│       print_config: true                                                      
│                                                                               
├── task_name
│   └── train                                                                   
├── accelerator
│   └── gpu                                                                     
├── devices
│   └── 1                                                                       
├── tags
│   └── ['dev']                                                                 
├── train
│   └── True                                                                    
├── test
│   └── True                                                                    
├── ckpt_path
│   └── None                                                                    
└── seed
    └── None                                                                    
[2022-10-04 16:16:20,657][__main__][INFO] - Instantiating datamodule <src.datamodules.cifar_datamodule.CIFARDataModule>
[2022-10-04 16:16:20,667][__main__][INFO] - Instantiating model <src.models.cifar_timm_module.CIFARLitModule>
[2022-10-04 16:16:21,015][timm.models.helpers][INFO] - Loading pretrained weights from url (https://download.pytorch.org/models/resnet18-5c106cde.pth)
[2022-10-04 16:16:21,148][__main__][INFO] - Instantiating callbacks...
[2022-10-04 16:16:21,150][src.utils.utils][INFO] - Instantiating callback <pytorch_lightning.callbacks.ModelCheckpoint>
[2022-10-04 16:16:21,156][src.utils.utils][INFO] - Instantiating callback <pytorch_lightning.callbacks.EarlyStopping>
[2022-10-04 16:16:21,158][src.utils.utils][INFO] - Instantiating callback <pytorch_lightning.callbacks.RichModelSummary>
[2022-10-04 16:16:21,159][src.utils.utils][INFO] - Instantiating callback <pytorch_lightning.callbacks.RichProgressBar>
[2022-10-04 16:16:21,159][__main__][INFO] - Instantiating loggers...
[2022-10-04 16:16:21,160][src.utils.utils][WARNING] - Logger config is empty.
[2022-10-04 16:16:21,160][__main__][INFO] - Instantiating trainer <pytorch_lightning.Trainer>
Trainer already configured with model summary callbacks: [<class 'pytorch_lightning.callbacks.rich_model_summary.RichModelSummary'>]. Skipping setting a default `ModelSummary` callback.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
[2022-10-04 16:16:21,754][__main__][INFO] - Starting training!
Files already downloaded and verified
Files already downloaded and verified
Length of CIFAR train:  50000
Length of CIFAR test:  10000
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃     ┃ Name                      ┃ Type                 ┃ Params ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 0   │ net                       │ ResNet               │ 11.2 M │
│ 1   │ net.conv1                 │ Conv2d               │  9.4 K │
│ 2   │ net.bn1                   │ BatchNorm2d          │    128 │
│ 3   │ net.act1                  │ ReLU                 │      0 │
│ 4   │ net.maxpool               │ MaxPool2d            │      0 │
│ 5   │ net.layer1                │ Sequential           │  147 K │
│ 6   │ net.layer1.0              │ BasicBlock           │ 74.0 K │
│ 7   │ net.layer1.0.conv1        │ Conv2d               │ 36.9 K │
│ 8   │ net.layer1.0.bn1          │ BatchNorm2d          │    128 │
│ 9   │ net.layer1.0.drop_block   │ Identity             │      0 │
│ 10  │ net.layer1.0.act1         │ ReLU                 │      0 │
│ 11  │ net.layer1.0.aa           │ Identity             │      0 │
│ 12  │ net.layer1.0.conv2        │ Conv2d               │ 36.9 K │
│ 13  │ net.layer1.0.bn2          │ BatchNorm2d          │    128 │
│ 14  │ net.layer1.0.act2         │ ReLU                 │      0 │
│ 15  │ net.layer1.1              │ BasicBlock           │ 74.0 K │
│ 16  │ net.layer1.1.conv1        │ Conv2d               │ 36.9 K │
│ 17  │ net.layer1.1.bn1          │ BatchNorm2d          │    128 │
│ 18  │ net.layer1.1.drop_block   │ Identity             │      0 │
│ 19  │ net.layer1.1.act1         │ ReLU                 │      0 │
│ 20  │ net.layer1.1.aa           │ Identity             │      0 │
│ 21  │ net.layer1.1.conv2        │ Conv2d               │ 36.9 K │
│ 22  │ net.layer1.1.bn2          │ BatchNorm2d          │    128 │
│ 23  │ net.layer1.1.act2         │ ReLU                 │      0 │
│ 24  │ net.layer2                │ Sequential           │  525 K │
│ 25  │ net.layer2.0              │ BasicBlock           │  230 K │
│ 26  │ net.layer2.0.conv1        │ Conv2d               │ 73.7 K │
│ 27  │ net.layer2.0.bn1          │ BatchNorm2d          │    256 │
│ 28  │ net.layer2.0.drop_block   │ Identity             │      0 │
│ 29  │ net.layer2.0.act1         │ ReLU                 │      0 │
│ 30  │ net.layer2.0.aa           │ Identity             │      0 │
│ 31  │ net.layer2.0.conv2        │ Conv2d               │  147 K │
│ 32  │ net.layer2.0.bn2          │ BatchNorm2d          │    256 │
│ 33  │ net.layer2.0.act2         │ ReLU                 │      0 │
│ 34  │ net.layer2.0.downsample   │ Sequential           │  8.4 K │
│ 35  │ net.layer2.0.downsample.0 │ Conv2d               │  8.2 K │
│ 36  │ net.layer2.0.downsample.1 │ BatchNorm2d          │    256 │
│ 37  │ net.layer2.1              │ BasicBlock           │  295 K │
│ 38  │ net.layer2.1.conv1        │ Conv2d               │  147 K │
│ 39  │ net.layer2.1.bn1          │ BatchNorm2d          │    256 │
│ 40  │ net.layer2.1.drop_block   │ Identity             │      0 │
│ 41  │ net.layer2.1.act1         │ ReLU                 │      0 │
│ 42  │ net.layer2.1.aa           │ Identity             │      0 │
│ 43  │ net.layer2.1.conv2        │ Conv2d               │  147 K │
│ 44  │ net.layer2.1.bn2          │ BatchNorm2d          │    256 │
│ 45  │ net.layer2.1.act2         │ ReLU                 │      0 │
│ 46  │ net.layer3                │ Sequential           │  2.1 M │
│ 47  │ net.layer3.0              │ BasicBlock           │  919 K │
│ 48  │ net.layer3.0.conv1        │ Conv2d               │  294 K │
│ 49  │ net.layer3.0.bn1          │ BatchNorm2d          │    512 │
│ 50  │ net.layer3.0.drop_block   │ Identity             │      0 │
│ 51  │ net.layer3.0.act1         │ ReLU                 │      0 │
│ 52  │ net.layer3.0.aa           │ Identity             │      0 │
│ 53  │ net.layer3.0.conv2        │ Conv2d               │  589 K │
│ 54  │ net.layer3.0.bn2          │ BatchNorm2d          │    512 │
│ 55  │ net.layer3.0.act2         │ ReLU                 │      0 │
│ 56  │ net.layer3.0.downsample   │ Sequential           │ 33.3 K │
│ 57  │ net.layer3.0.downsample.0 │ Conv2d               │ 32.8 K │
│ 58  │ net.layer3.0.downsample.1 │ BatchNorm2d          │    512 │
│ 59  │ net.layer3.1              │ BasicBlock           │  1.2 M │
│ 60  │ net.layer3.1.conv1        │ Conv2d               │  589 K │
│ 61  │ net.layer3.1.bn1          │ BatchNorm2d          │    512 │
│ 62  │ net.layer3.1.drop_block   │ Identity             │      0 │
│ 63  │ net.layer3.1.act1         │ ReLU                 │      0 │
│ 64  │ net.layer3.1.aa           │ Identity             │      0 │
│ 65  │ net.layer3.1.conv2        │ Conv2d               │  589 K │
│ 66  │ net.layer3.1.bn2          │ BatchNorm2d          │    512 │
│ 67  │ net.layer3.1.act2         │ ReLU                 │      0 │
│ 68  │ net.layer4                │ Sequential           │  8.4 M │
│ 69  │ net.layer4.0              │ BasicBlock           │  3.7 M │
│ 70  │ net.layer4.0.conv1        │ Conv2d               │  1.2 M │
│ 71  │ net.layer4.0.bn1          │ BatchNorm2d          │  1.0 K │
│ 72  │ net.layer4.0.drop_block   │ Identity             │      0 │
│ 73  │ net.layer4.0.act1         │ ReLU                 │      0 │
│ 74  │ net.layer4.0.aa           │ Identity             │      0 │
│ 75  │ net.layer4.0.conv2        │ Conv2d               │  2.4 M │
│ 76  │ net.layer4.0.bn2          │ BatchNorm2d          │  1.0 K │
│ 77  │ net.layer4.0.act2         │ ReLU                 │      0 │
│ 78  │ net.layer4.0.downsample   │ Sequential           │  132 K │
│ 79  │ net.layer4.0.downsample.0 │ Conv2d               │  131 K │
│ 80  │ net.layer4.0.downsample.1 │ BatchNorm2d          │  1.0 K │
│ 81  │ net.layer4.1              │ BasicBlock           │  4.7 M │
│ 82  │ net.layer4.1.conv1        │ Conv2d               │  2.4 M │
│ 83  │ net.layer4.1.bn1          │ BatchNorm2d          │  1.0 K │
│ 84  │ net.layer4.1.drop_block   │ Identity             │      0 │
│ 85  │ net.layer4.1.act1         │ ReLU                 │      0 │
│ 86  │ net.layer4.1.aa           │ Identity             │      0 │
│ 87  │ net.layer4.1.conv2        │ Conv2d               │  2.4 M │
│ 88  │ net.layer4.1.bn2          │ BatchNorm2d          │  1.0 K │
│ 89  │ net.layer4.1.act2         │ ReLU                 │      0 │
│ 90  │ net.global_pool           │ SelectAdaptivePool2d │      0 │
│ 91  │ net.global_pool.flatten   │ Flatten              │      0 │
│ 92  │ net.global_pool.pool      │ AdaptiveAvgPool2d    │      0 │
│ 93  │ net.fc                    │ Linear               │  5.1 K │
│ 94  │ criterion                 │ CrossEntropyLoss     │      0 │
│ 95  │ train_acc                 │ Accuracy             │      0 │
│ 96  │ val_acc                   │ Accuracy             │      0 │
│ 97  │ test_acc                  │ Accuracy             │      0 │
│ 98  │ train_loss                │ MeanMetric           │      0 │
│ 99  │ val_loss                  │ MeanMetric           │      0 │
│ 100 │ test_loss                 │ MeanMetric           │      0 │
│ 101 │ val_acc_best              │ MaxMetric            │      0 │
│ 102 │ normalize                 │ Normalize            │      0 │
│ 103 │ resize                    │ Resize               │      0 │
└─────┴───────────────────────────┴──────────────────────┴────────┘
Trainable params: 11.2 M                                                        
Non-trainable params: 0                                                         
Total params: 11.2 M                                                            
Total estimated model params size (MB): 44                                      
/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:566: 
UserWarning: This DataLoader will create 16 worker processes in total. Our 
suggested max number of worker in current system is 2, which is smaller than 
what this DataLoader is going to create. Please be aware that excessive worker 
creation might get DataLoader running slow or even freeze, lower the worker 
number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Epoch 0    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: nan val/loss:
                                                             2.342 val/acc: 0.12
Epoch 0    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:09 • -:--:-- 0.00it/s loss: nan val/loss:
                                                             2.342 val/acc: 0.12
Epoch 0    ━━━━━━━━━━━━━━━━━━━━━━━━━━ 6/6 0:00:17 • 0:00:00 0.62it/s loss: 1.63 
Epoch 0    ━━━━━━━━━━━━━━━━━━━━━━━━━━ 6/6 0:00:17 • 0:00:00 0.62it/s loss: 1.63 
Epoch 1    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 1.63         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 1.63         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:10 • -:--:-- 0.00it/s loss: 1.63         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:11 • -:--:-- 0.00it/s loss: 1.51         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:11 • 0:00:06 0.71it/s loss: 1.51         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.71it/s loss: 1.41         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.71it/s loss: 1.41         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:14 • 0:00:05 0.71it/s loss: 1.33         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:14 • 0:00:03 0.71it/s loss: 1.33         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:15 • 0:00:03 0.71it/s loss: 1.27         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:15 • 0:00:02 0.73it/s loss: 1.27         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.73it/s loss: 1.21         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:18 • 0:00:02 0.73it/s loss: 1.21         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:18 • 0:00:02 0.73it/s loss: 1.21         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:18 • 0:00:00 0.58it/s loss: 1.21         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 1    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:18 • 0:00:00 0.58it/s loss: 1.21         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
                                                             train/acc: 0.417   
Epoch 1    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:18 • 0:00:00 0.58it/s loss: 1.21         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
                                                             train/acc: 0.417   
Epoch 1    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:18 • 0:00:00 0.58it/s loss: 1.21         
                                                             val/loss: 3.3      
                                                             val/acc: 0.233     
                                                             val/acc_best: 0.233
                                                             train/loss: 1.631  
Epoch 2    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 1.21         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 1.21         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:10 • -:--:-- 0.00it/s loss: 1.21         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:12 • -:--:-- 0.00it/s loss: 1.16         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.71it/s loss: 1.16         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:13 • 0:00:06 0.71it/s loss: 1.11         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.70it/s loss: 1.11         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:14 • 0:00:05 0.70it/s loss: 1.07         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:15 • 0:00:03 0.70it/s loss: 1.07         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:16 • 0:00:03 0.70it/s loss: 1.04         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.72it/s loss: 1.04         
                                                             val/loss: 1.269    
                                                             val/acc: 0.594     
                                                             val/acc_best: 0.594
                                                             train/loss: 0.796  
Epoch 2    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:17 • 0:00:02 0.72it/s loss: 1 val/loss:  
                                                             1.269 val/acc:     
                                                             0.594 val/acc_best:
                                                             0.594 train/loss:  
                                                             0.796 train/acc:   
Epoch 2    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.72it/s loss: 1 val/loss:  
                                                             1.269 val/acc:     
                                                             0.594 val/acc_best:
                                                             0.594 train/loss:  
                                                             0.796 train/acc:   
Epoch 2    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.72it/s loss: 1 val/loss:  
                                                             1.269 val/acc:     
                                                             0.594 val/acc_best:
                                                             0.594 train/loss:  
                                                             0.796 train/acc:   
Epoch 2    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 1 val/loss:  
                                                             1.269 val/acc:     
                                                             0.594 val/acc_best:
                                                             0.594 train/loss:  
                                                             0.796 train/acc:   
Epoch 2    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 1 val/loss:  
                                                             1.269 val/acc:     
                                                             0.594 val/acc_best:
                                                             0.594 train/loss:  
                                                             0.796 train/acc:   
                                                             0.721              
Epoch 2    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 1 val/loss:  
                                                             1.269 val/acc:     
                                                             0.594 val/acc_best:
                                                             0.594 train/loss:  
                                                             0.796 train/acc:   
                                                             0.721              
Epoch 2    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 1 val/loss:  
                                                             1.269 val/acc:     
                                                             0.594 val/acc_best:
                                                             0.594 train/loss:  
                                                             0.796 train/acc:   
Epoch 3    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 1 val/loss:  
                                                             0.859 val/acc:     
                                                             0.713 val/acc_best:
                                                             0.713 train/loss:  
                                                             0.579 train/acc:   
Epoch 3    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 1 val/loss:  
                                                             0.859 val/acc:     
                                                             0.713 val/acc_best:
                                                             0.713 train/loss:  
                                                             0.579 train/acc:   
Epoch 3    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:10 • -:--:-- 0.00it/s loss: 1 val/loss:  
                                                             0.859 val/acc:     
                                                             0.713 val/acc_best:
                                                             0.713 train/loss:  
                                                             0.579 train/acc:   
Epoch 3    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:11 • -:--:-- 0.00it/s loss: 0.968        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:11 • 0:00:06 0.70it/s loss: 0.968        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.70it/s loss: 0.938        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.70it/s loss: 0.938        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:14 • 0:00:05 0.70it/s loss: 0.91         
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:14 • 0:00:03 0.70it/s loss: 0.91         
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:15 • 0:00:03 0.70it/s loss: 0.884        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.72it/s loss: 0.884        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.72it/s loss: 0.861        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.72it/s loss: 0.861        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.72it/s loss: 0.861        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.55it/s loss: 0.861        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 3    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.55it/s loss: 0.861        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
                                                             train/acc: 0.799   
Epoch 3    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.55it/s loss: 0.861        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
                                                             train/acc: 0.799   
Epoch 3    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.55it/s loss: 0.861        
                                                             val/loss: 0.859    
                                                             val/acc: 0.713     
                                                             val/acc_best: 0.713
                                                             train/loss: 0.579  
Epoch 4    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.861        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.861        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:10 • -:--:-- 0.00it/s loss: 0.861        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:11 • -:--:-- 0.00it/s loss: 0.751        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:11 • 0:00:06 0.69it/s loss: 0.751        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.69it/s loss: 0.673        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.69it/s loss: 0.673        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:14 • 0:00:05 0.69it/s loss: 0.616        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:14 • 0:00:03 0.69it/s loss: 0.616        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:15 • 0:00:03 0.69it/s loss: 0.569        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.71it/s loss: 0.569        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.71it/s loss: 0.533        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:18 • 0:00:02 0.71it/s loss: 0.533        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:18 • 0:00:02 0.71it/s loss: 0.533        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 0.533        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 4    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 0.533        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
                                                             train/acc: 0.847   
Epoch 4    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 0.533        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
                                                             train/acc: 0.847   
Epoch 4    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 0.533        
                                                             val/loss: 0.731    
                                                             val/acc: 0.749     
                                                             val/acc_best: 0.749
                                                             train/loss: 0.437  
Epoch 5    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.533        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.533        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:09 • -:--:-- 0.00it/s loss: 0.533        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:11 • -:--:-- 0.00it/s loss: 0.5 val/loss:
                                                             0.668 val/acc: 0.78
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:11 • 0:00:06 0.69it/s loss: 0.5 val/loss:
                                                             0.668 val/acc: 0.78
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.69it/s loss: 0.468        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:12 • 0:00:05 0.69it/s loss: 0.468        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.69it/s loss: 0.44         
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:14 • 0:00:03 0.69it/s loss: 0.44         
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:15 • 0:00:03 0.69it/s loss: 0.414        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:15 • 0:00:02 0.71it/s loss: 0.414        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.71it/s loss: 0.387        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:18 • 0:00:02 0.71it/s loss: 0.387        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:18 • 0:00:02 0.71it/s loss: 0.387        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:18 • 0:00:00 0.56it/s loss: 0.387        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 5    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:18 • 0:00:00 0.56it/s loss: 0.387        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
                                                             train/acc: 0.89    
Epoch 5    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:18 • 0:00:00 0.56it/s loss: 0.387        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
                                                             train/acc: 0.89    
Epoch 5    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:18 • 0:00:00 0.56it/s loss: 0.387        
                                                             val/loss: 0.668    
                                                             val/acc: 0.78      
                                                             val/acc_best: 0.78 
                                                             train/loss: 0.319  
Epoch 6    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.387        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.387        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:10 • -:--:-- 0.00it/s loss: 0.387        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:11 • -:--:-- 0.00it/s loss: 0.363        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.69it/s loss: 0.363        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:13 • 0:00:06 0.69it/s loss: 0.341        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.69it/s loss: 0.341        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:14 • 0:00:05 0.69it/s loss: 0.318        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:14 • 0:00:03 0.68it/s loss: 0.318        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:16 • 0:00:03 0.68it/s loss: 0.296        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.71it/s loss: 0.296        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.71it/s loss: 0.275        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.71it/s loss: 0.275        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.71it/s loss: 0.275        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.275        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 6    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.275        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
                                                             train/acc: 0.928   
Epoch 6    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.275        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
                                                             train/acc: 0.928   
Epoch 6    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.275        
                                                             val/loss: 0.615    
                                                             val/acc: 0.801     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.215  
Epoch 7    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.275        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.275        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:10 • -:--:-- 0.00it/s loss: 0.275        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:11 • -:--:-- 0.00it/s loss: 0.256        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:11 • 0:00:06 0.69it/s loss: 0.256        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.69it/s loss: 0.237        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.68it/s loss: 0.237        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:14 • 0:00:05 0.68it/s loss: 0.218        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:14 • 0:00:03 0.68it/s loss: 0.218        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:15 • 0:00:03 0.68it/s loss: 0.201        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:15 • 0:00:02 0.71it/s loss: 0.201        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.71it/s loss: 0.184        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:18 • 0:00:02 0.71it/s loss: 0.184        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:18 • 0:00:02 0.71it/s loss: 0.184        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 0.184        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 7    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 0.184        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
                                                             train/acc: 0.96    
Epoch 7    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 0.184        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
                                                             train/acc: 0.96    
Epoch 7    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.57it/s loss: 0.184        
                                                             val/loss: 0.663    
                                                             val/acc: 0.799     
                                                             val/acc_best: 0.801
                                                             train/loss: 0.131  
Epoch 8    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.184        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.184        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:10 • -:--:-- 0.00it/s loss: 0.184        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:11 • -:--:-- 0.00it/s loss: 0.169        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:11 • 0:00:06 0.69it/s loss: 0.169        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.69it/s loss: 0.154        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.68it/s loss: 0.154        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:14 • 0:00:05 0.68it/s loss: 0.14         
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:14 • 0:00:03 0.68it/s loss: 0.14         
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:15 • 0:00:03 0.68it/s loss: 0.127        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.70it/s loss: 0.127        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.70it/s loss: 0.113        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.70it/s loss: 0.113        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.70it/s loss: 0.113        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.113        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 8    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.113        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
                                                             train/acc: 0.98    
Epoch 8    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.113        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
                                                             train/acc: 0.98    
Epoch 8    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.113        
                                                             val/loss: 0.694    
                                                             val/acc: 0.806     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.072  
Epoch 9    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.113        
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━━━━━━━ 0/6 0:00:00 • -:--:-- 0.00it/s loss: 0.113        
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:10 • -:--:-- 0.00it/s loss: 0.113        
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━╺━━━━━━━━━━━━━━ 1/6 0:00:11 • -:--:-- 0.00it/s loss: 0.103        
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:12 • 0:00:06 0.68it/s loss: 0.103        
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━╺━━━━━━━━━━━ 2/6 0:00:13 • 0:00:06 0.68it/s loss: 0.0931       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:13 • 0:00:05 0.68it/s loss: 0.0931       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━╺━━━━━━━━ 3/6 0:00:14 • 0:00:05 0.68it/s loss: 0.0833       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:15 • 0:00:03 0.68it/s loss: 0.0833       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━╺━━━━━ 4/6 0:00:16 • 0:00:03 0.68it/s loss: 0.0737       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:16 • 0:00:02 0.71it/s loss: 0.0737       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:17 • 0:00:02 0.71it/s loss: 0.0643       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.71it/s loss: 0.0643       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━━━━╺━━ 5/6 0:00:19 • 0:00:02 0.71it/s loss: 0.0643       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.0643       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
Epoch 9    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.0643       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
                                                             train/acc: 0.992   
Epoch 9    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.0643       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
                                                             train/acc: 0.992   
Epoch 9    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.0643       
                                                             val/loss: 0.769    
                                                             val/acc: 0.804     
                                                             val/acc_best: 0.806
                                                             train/loss: 0.036  
                                                             train/acc: 0.992   `Trainer.fit` stopped: `max_epochs=10` reached.
Epoch 9    ━━━━━━━━━━━━━━━━━━ 6/6 0:00:19 • 0:00:00 0.56it/s loss: 0.0643       
                                                             val/loss: 0.846    
                                                             val/acc: 0.811     
                                                             val/acc_best: 0.811
                                                             train/loss: 0.018  
                                                             train/acc: 0.996   
[2022-10-04 16:20:01,507][__main__][INFO] - Scripting Model
[2022-10-04 16:20:02,393][__main__][INFO] - Saving model to /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/train/runs/2022-10-04_16-16-20
[2022-10-04 16:20:02,393][__main__][INFO] - Starting testing!
Files already downloaded and verified
Files already downloaded and verified
Length of CIFAR train:  50000
Length of CIFAR test:  10000
Restoring states from the checkpoint path at /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/train/runs/2022-10-04_16-16-20/checkpoints/epoch_009.ckpt
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loaded model weights from checkpoint at /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/train/runs/2022-10-04_16-16-20/checkpoints/epoch_009.ckpt
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/acc          │    0.8084999918937683     │
│         test/loss         │    0.8463168144226074     │
└───────────────────────────┴───────────────────────────┘
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:00:00 • 0:00:00 0.00it/s 
[2022-10-04 16:20:11,620][__main__][INFO] - Best ckpt path: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/train/runs/2022-10-04_16-16-20/checkpoints/epoch_009.ckpt
[2022-10-04 16:20:11,633][src.utils.utils][INFO] - Closing loggers...
[2022-10-04 16:20:11,636][src.utils.utils][INFO] - Output dir: /content/gdrive/MyDrive/EMLO2/S4_Main_Colab/logs/train/runs/2022-10-04_16-16-20
[2022-10-04 16:20:11,637][src.utils.utils][INFO] - Metric name is None! Skipping metric value retrieval...
```
