import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class Efficientnet_b6(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained == True:
            self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b6', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class Efficientnet_b4(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained == True:
            self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        else:
            self.model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class nfnet_f0(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained == True:
            self.model = timm.create_model('dm_nfnet_f0', pretrained=True)
        else:
            self.model = timm.create_model('dm_nfnet_f0', pretrained=False)

        self.model.head.fc = nn.Linear(in_features=3072, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)


class ecaresnet50t(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained == True:
            self.model = timm.create_model('ecaresnet50t', pretrained=True)
        else:
            self.model = timm.create_model('ecaresnet50t', pretrained=False)

        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)

class seresnet152d(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained == True:
            self.model = timm.create_model('seresnet152d', pretrained=True)
        else:
            self.model = timm.create_model('seresnet152d', pretrained=False)

        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model(x)