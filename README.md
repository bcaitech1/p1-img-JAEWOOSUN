pstage_01_image_classification
=====

## 1) 목표

Mask를 쓴 사람의 Gender와 Age를 분류하는 Classificiation 문제입니다.

총 18개이 Label로 분류합니다.

* Mask (3개)
  * Correct Mask
  * Incorrect Mask
  * Normal (마스크 착용X)
* Gender (2개)
  * Male
  * Female
* Age (3개)
  * ~ 30
  * 30 ~ 60
  * 60 ~ 



## 2) 사용법   
      
### [Dependencies]

* torch==1.6.0

* torchvision==0.7.0

* nni (Auto ML)

### [Install Requirements]   

    pip install -r requirements.txt

### [Training]

* Terminal 실행   
  
  parser를 통해 실행할 수 있습니다.

```
python train.py --epochs 30 --batch_size 16 --criterion cross_entropy --classification multi --model Efficientnet_b4
```

* nni 실행   
  
  nnictl를 통해 실행할 수 있습니다.

```
nnictl create --config /opt/ml/code/baseline/config_remote.yml --port 5000
```

## 3) 파일 설명

* train.py   
  
  model의 실행, 결과 csv 예측

* model.py
  
  model 구현

* dataset.py
  
  dataset 구현, Augmentation 구현

* loss.py
  
  loss 구현

* soft_votes_ensemble.ipynb
  
  Inference된 csv 파일들의 soft voting ensemble code

* config_remote.yml
  
  nni 설정

* search_space.json
  
  nni에서 찾을 hypereparameter 설정


## 4) 주요 코드 설명

## [train.py]   

- make_csv 함수  
  
  Inference 단계에서 hard submission과 soft submission을 생성해줍니다.

  (280줄)

```
make_csv(model, test_loader, epoch, submission, save_dir, train_acc, loss_avg_value)
```

- logger
  
  이번 train에 사용된 hyperparameter와 여러가지 parameter들을 저장합니다.

  epoch 마다의 accuracy와 loss를 저장합니다.

  (231줄)

```
logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=4)
```

- Learning rate 줄이기
  
  중간 중간에 일정한 조건이 만족되면 learning rate를 줄여줍니다.

  (285줄)

```
if epoch in [10, 20, 25]:
    for g in optimizer.param_groups:
        g['lr'] /= 10
    print("Loss 1/10")
```

- nni report
  
  한 epoch이 끝날 때마다 nni에 report합니다. report 값은 사용자가 정할 수 있습니다.

  (319줄)

```
nni.report_intermediate_result(val_acc)
```

- DataLoader
  
  Custom Dataset을 불러와 DataLoader을 만들 수 있습니다.


```
train_loader = DataLoader(
        train_set,
        batch_size=args['batch_size'],
        num_workers=2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
```

## [model.py]

- Efficientnet_b6
  
  Pre-trained된 model을 사용해서 fc layer를 수정해서 fine-tuning으로 사용했습니다.

  (41줄)

```
class Efficientnet_b6(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained == True:
            self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b6', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
```

## [dataset.py]

- Augmentation
  
  torch.transforms으로 가지고 옵니다. 원하는 Augmentation 방식을 넣을 수 있습니다.

  (22줄)

```
class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.CenterCrop([350, 350]),
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)
```

- MaskBaseDataset
  
  Data들을 가지고 오는 Custom dataset입니다.

  (80줄)

- MaskSplitByProfileDataset
  
  train / valid를 나누는 기준이 random이 아닌, 사람 (profile) 기준으로 나누게 됩니다.

  valid를 random으로 나누게 되면 valid에서 train data를 이미 학습해버리는 데이터 유출이 일어나게 됩니다.

  (237줄)

- TestDataset
  
  Test data에 대한 dataset입니다.

  (294줄)


## [loss.py]

- FocalLoss
  
  Focal loss에 대해 정의합니다.

  (7줄)

- create_criterion 함수
  
  위에서 정의된 criterion class를 가지고 오는 함수입니다.

  이때, F1, Label smoothing은 class의 개수를 정해줘야 합니다.

  (91줄)

```
def create_criterion(criterion_name, num_classes, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
        if criterion_name in ['label_smoothing', 'f1']:
            criterion.set_classes(num_classes)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion
``` 

## [soft_votes_ensemble.ipynb]

생략

## [Config_remote.yml]

```
# config_remote.yml

authorName: default
experimentName: pstage1-env1
trialConcurrency: 1
maxExecDuration: 12h
maxTrialNum: 50
trainingServicePlatform: remote
searchSpacePath: search_space.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 train.py
  codeDir: .
  gpuNum: 1
#machineList can be empty if the platform is local
machineList:
  - ip: 101.101.216.**
    port : 2222
    username: root
    sshKeyPath: /opt/ml/key
    pythonPath: /opt/conda/bin/python
```
- experimentName : 실험 제목

- maxExecDuration : 실험 최대 수행 시간
  
  최대 시간이 끝나면 실험이 종료됩니다.

- maxTrialNum : 실험 최대 수행 횟수
  
  마찬가지로 최대 수행 횟수에 도달하면 실험이 종료됩니다.

- trainingServicePlatform : 실험 할 platform
  
  여기서 remote로 설정해야합니다.
  
- searchSpacePath : 실험할 변수들을 가지고 있는 파일
  
  search_space.json 파일을 가지고 옵니다.

- command : 실행할 명령
  
  terminal 형식으로 실행되기 때문에 python [PYTHON_NAME].py로 해주세요.

- gpuNum : 실행할 gpu 개수

- machineList : remote server 정보
  
  remote server를 사용할 것이라면 가장 중요한 정보입니다.

    (1) ip : remote server ip를 입력

    (2) port : remote server에 연결된 port 번호

    (3) username : 대부분 root (따로 지정한 username이 있다면 username 작성)

    (4) sshKeyPath : ssh key를 사용하고 있다면 ssh key가 저장되어 있는 path를 적습니다. 만약 password를 사용하고 있다면 sshKeyPath 대신 passwd를 적어줍니다.

    (5) PythonPath : python interpreter가 있는 path 


## [search_space.json]

원하는 hyperparameter를 넣어줄 수 있습니다.

```
{
    "batch_size": {"_type":"choice", "_value": [8, 16, 32, 64, 128]},
    "lr":{"_type":"choice","_value":[0.001, 0.01, 0.1]},
    "model": {"_type" : "choice", "_value" : ["Efficientnet_b6", "Efficientnet_b0", "Efficientnet_b3"]},
    "criterion": {"_type": "choice", "_value" : ["focal", "label_smoothing", "cross_entropy", "f1"]},
    "augmentation": {"_type": "choice", "_value" : ["BaseAugmentation", "AffineAugmentation", "GaussianAugmentation"]}
}
```


