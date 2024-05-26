import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms
import torch.optim as optim





# 数据加载和预处理
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np


class ExpressionDataset(Dataset):
    def __init__(self, folders, transform=None):
        self.folders = folders
        self.transform = transform
        self.images = []
        self.labels = []

        self.load_images()

    def load_images(self):
        for label, folder in enumerate(self.folders):
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Keep RGB
        img = cv2.resize(img, (299, 299))  # Resize to GoogLeNet input size
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])






# Folder paths our dataset
folders = [
    "E:/PyCharm/workspace/archive 2/Sad_augmented",
    "E:/PyCharm/workspace/archive 2/Angry_augmented",
    "E:/PyCharm/workspace/archive 2/Happy_augmented"

]


# # A big data set
# # Folder paths
# folders = [
#     "E:/PyCharm/workspace/archive/Sad",
#     "E:/PyCharm/workspace/archive/Angry",
#     "E:/PyCharm/workspace/archive/happy"
#
# ]




# Dataset and DataLoader
dataset = ExpressionDataset(folders=folders, transform=transform)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)





# cutmix
def cutmix(data, target, alpha, minority_classes=[0, 2]):
    batch_size = data.size(0)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    for i in range(batch_size):
        if target[i] in minority_classes or np.random.rand() < 0.5:
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
            data[i, :, bbx1:bbx2, bby1:bby2] = shuffled_data[i, :, bbx1:bbx2, bby1:bby2]
            target_a = target.clone()
            target_b = shuffled_target.clone()
            target_a[i] = target[i] * lam + target_b[i] * (1 - lam)  # 只对selected indices做adjust
            target_b[i] = target_b[i] * lam + target_a[i] * (1 - lam)
    return data, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class SEBlock(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SEBlock, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )


    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()

        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # 打印应用SE权重前的通道统计信息
        # print("权重应用前的通道平均值:", x.mean(dim=[2, 3]))
        # Fscale操作：将得到的权重乘以原来的特征图x
        output = x * y.expand_as(x)
        # print("权重应用后的通道平均值:", output.mean(dim=[2, 3]))

        return output








class GoogLeNetV3(nn.Module):
    def __init__(self, num_classes=3, aux_logits=True, init_weights=False):
        super(GoogLeNetV3, self).__init__()
        self.aux_logits = aux_logits
        # 3个3×3卷积替代7×7卷积
        self.conv1_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv1_2 = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv1_3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 80, kernel_size=3)
        self.conv3 = BasicConv2d(80, 192, kernel_size=3, stride=2)
        self.conv4 = BasicConv2d(192, 192, kernel_size=3, padding=1)

        self.inception3a = InceptionV3A(192, 64, 48, 64, 64, 96, 32)
        self.inception3b = InceptionV3A(256, 64, 48, 64, 64, 96, 64)
        self.inception3c = InceptionV3A(288, 64, 48, 64, 64, 96, 64)

        self.inception4a = InceptionV3D(288, 0, 384, 384, 64, 96, 0, total_channels=768)
        self.inception4b = InceptionV3B(768, 192, 128, 192, 128, 192, 192)
        self.inception4c = InceptionV3B(768, 192, 160, 192, 160, 192, 192)
        self.inception4d = InceptionV3B(768, 192, 160, 192, 160, 192, 192)
        self.inception4e = InceptionV3D(768, 0, 384, 384, 64, 128, 0,total_channels=1280)

        if self.aux_logits == True:
            self.aux = InceptionAux(in_channels=768, out_channels=num_classes)

        self.inception5a = InceptionV3C(1280, 320, 384, 384, 448, 384, 192)
        self.inception5b = InceptionV3C(2048, 320, 384, 384, 448, 384, 192)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print("Input size:", x.size())
        # N x 3 x 299 x 299
        x = self.conv1_1(x)
        # print("After conv1_1 size:", x.size())
        # N x 32 x 149 x 149
        x = self.conv1_2(x)
        # print("After conv1_2 size:", x.size())
        # N x 32 x 147 x 147
        x = self.conv1_3(x)
        # print("After conv1_3 size:", x.size())
        #  N x 32 x 147 x 147
        x = self.maxpool1(x)
        # print("After maxpool1 size:", x.size())
        # N x 64 x 73 x 73
        x = self.conv2(x)
        # print("After conv2 size:", x.size())
        # N x 80 x 71 x 71
        x = self.conv3(x)
        # print("After conv3 size:", x.size())
        # N x 192 x 35 x 35
        x = self.conv4(x)
        # print("After conv4 size:", x.size())
        # N x 192 x 35 x 35
        x = self.inception3a(x)
        # print("After inception3a size:", x.size())
        # N x 256 x 35 x 35
        x = self.inception3b(x)
        # print("After inception3b size:", x.size())
        # N x 288 x 35 x 35
        x = self.inception3c(x)
        # print("After inception3c size:", x.size())
        # N x 288 x 35x 35
        x = self.inception4a(x)
        # print("After inception4a size:", x.size())
        # N x 768 x 17 x 17
        x = self.inception4b(x)
        # print("After inception4b size:", x.size())
        # N x 768 x 17 x 17
        x = self.inception4c(x)
        # print("After inception4c size:", x.size())
        # N x 768 x 17 x 17
        x = self.inception4d(x)
        # print("After inception4d size:", x.size())
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:    # eval model lose this layer
            aux = self.aux(x)
            # print("Aux output size:", aux.size())
        # N x 768 x 17 x 17
        x = self.inception4e(x)
        # print("After inception4e size:", x.size())
        # N x 1280 x 8 x 8
        x = self.inception5a(x)
        # print("After inception5a size:", x.size())
        # N x 2048 x 8 x 8
        x = self.inception5b(x)
        # print("After inception5b size:", x.size())
        # N x 2048 x 7 x 7
        x = self.avgpool(x)
        # print("After avgpool size:", x.size())
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # print("After flatten size:", x.size())
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # print("Final output size:", x.size())
        # N x 1000(num_classes)
        if self.training and self.aux_logits:  # 训练阶段使用
            return x, aux
        return x
    # 对模型的权重进行初始化操作
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # 同样地，确保偏置存在
                    nn.init.constant_(m.bias, 0)

# InceptionV3A:BasicConv2d+MaxPool2d
class InceptionV3A(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3A, self).__init__()
        # 1×1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1卷积+3×3卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )
        # 1×1卷积++3×3卷积+3×3卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=3, padding=1)         # 保证输出大小等于输入大小
        )
        # 3×3池化+1×1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
        # 计算总通道数以配置SE Block
        total_channels = ch1x1 + ch3x3 + ch3x3X2 + pool_proj
        self.se_block = SEBlock(total_channels)



    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 拼接
        outputs = [branch1, branch2, branch3, branch4]
        outputs = torch.cat(outputs, 1)
        # 应用SE Block
        # print("A Output size before SEBlock:", outputs.size())
        outputs = self.se_block(outputs)
        return outputs

# InceptionV3B:BasicConv2d+MaxPool2d
class InceptionV3B(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3B, self).__init__()
        # 1×1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1卷积+1×3卷积+3×1卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3, ch3x3, kernel_size=[3, 1], padding=[1, 0])   # 保证输出大小等于输入大小
        )
        # 1×1卷积+1×3卷积+3×1卷积+1×3卷积+3×1卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0])  # 保证输出大小等于输入大小
        )
        # 3×3池化+1×1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

        # 计算总通道数以配置SE Block
        total_channels = ch1x1 + ch3x3 + ch3x3X2 + pool_proj
        self.se_block = SEBlock(total_channels)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 拼接
        outputs = [branch1, branch2, branch3, branch4]
        outputs = torch.cat(outputs, 1)
        # 应用SE Block
        outputs = self.se_block(outputs)
        return outputs

# InceptionV3C:BasicConv2d+MaxPool2d
class InceptionV3C(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3C, self).__init__()
        # 1×1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1卷积+1×3卷积+3×1卷积
        self.branch2_0 = BasicConv2d(in_channels, ch3x3red, kernel_size=1)
        self.branch2_1 = BasicConv2d(ch3x3red, ch3x3, kernel_size=[1, 3], padding=[0, 1])
        self.branch2_2 = BasicConv2d(ch3x3red, ch3x3, kernel_size=[3, 1], padding=[1, 0])

        # 1×1卷积+3×3卷积+1×3卷积+3×1卷积
        self.branch3_0 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),
        )
        self.branch3_1 = BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1])
        self.branch3_2 = BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0])

        # 3×3池化+1×1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

        total_channels = ch1x1 + 2*ch3x3 + 2*ch3x3X2 + pool_proj
        self.se_block = SEBlock(total_channels)
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2_0 = self.branch2_0(x)
        branch2 = torch.cat([self.branch2_1(branch2_0), self.branch2_2(branch2_0)], dim=1)
        branch3_0 = self.branch3_0(x)
        branch3 = torch.cat([self.branch3_1(branch3_0), self.branch3_2(branch3_0)], dim=1)
        branch4 = self.branch4(x)
        # 拼接
        outputs = [branch1, branch2, branch3, branch4]
        outputs = torch.cat(outputs, 1)
        # 应用SE Block
        # print("C Output size before SEBlock:", outputs.size())
        outputs = self.se_block(outputs)
        return outputs
# InceptionV3D:BasicConv2d+MaxPool2d
class InceptionV3D(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj, total_channels):
        super(InceptionV3D, self).__init__()
        # ch1x1:没有1×1卷积
        # 1×1卷积+3×3卷积,步长为2
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, stride=2)
        )
        # 1×1卷积+3×3卷积+3×3卷积,步长为2
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),   # 保证输出大小等于输入大小
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=3, stride=2)
        )
        # 3×3池化,步长为2
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2))
        # pool_proj:池化层后不再接卷积层

        # total_channels = ch3x3 + ch3x3X2

        #
        self.se_block = SEBlock(total_channels)




    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        # 拼接
        outputs = [branch1, branch2, branch3]
        outputs = torch.cat(outputs, 1)

        # 打印输出尺寸，确保它是预期的尺寸
        # print("D Output size before SEBlock:", outputs.size())

        # 应用SE Block
        outputs = self.se_block(outputs)
        # print("After SEBlock size:", x.size())
        return outputs

# 辅助分类器:AvgPool2d+BasicConv2d+Linear+dropout
class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()

        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.conv2 = BasicConv2d(in_channels=128, out_channels=768, kernel_size=5, stride=1)
        self.dropout = nn.Dropout(p=0.7)
        self.linear = nn.Linear(in_features=768, out_features=out_channels)
    def forward(self, x):
        # N x 768 x 17 x 17
        x = self.averagePool(x)
        # N x 768 x 5 x 5
        x = self.conv1(x)
        # N x 128 x 5 x 5
        x = self.conv2(x)
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        out = self.linear(self.dropout(x))
        # N x num_classes
        return out

# 卷积组: Conv2d+BN+ReLU
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GoogLeNetV3().to(device)
    summary(model, input_size=(3, 299, 299))


# 之后的步骤
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GoogLeNetV3(num_classes=3, aux_logits=True, init_weights=True).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import matplotlib.pyplot as plt

# 在训练循环外初始化存储损失和准确率的列表
loss_history = []
accuracy_history = []

# 假设 `dataset` 是你的数据集
# 定义超参数

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

num_epochs = 150
k_folds = 5
criterion = nn.CrossEntropyLoss()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义K折交叉验证
kfold = KFold(n_splits=k_folds, shuffle=True)

# 获取数据集的索引和标签
dataset_size = len(dataset)
indices = list(range(dataset_size))
labels = [dataset[i][1] for i in range(dataset_size)]

# 类别名称
classes = ['Sad', 'Angry', 'Happy']

# 存储所有折的准确率
fold_class_accuracies = {class_name: [] for class_name in classes}
fold_total_accuracies = []

# K折交叉验证
for fold, (train_ids, val_ids) in enumerate(kfold.split(indices, labels)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # 创建数据加载器
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)

    # 初始化模型
    model = GoogLeNetV3(num_classes=3, aux_logits=True, init_weights=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 初始化列表存储损失和准确率
    loss_history = []
    accuracy_history = []

    # 训练模型
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            if np.random.rand() < 0.5:  # 例如50%的概率应用CutMix
                inputs, targets_a, targets_b, lam = cutmix(inputs, labels, alpha=1.0)
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1 - lam)
                loss2 = criterion(aux_outputs, targets_a) * lam + criterion(aux_outputs, targets_b) * (1 - lam)
            else:
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)

            loss = loss1 + 0.4 * loss2

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total

        # 存储损失和准确率以供后续绘图
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')

    # 绘制损失和准确率图表
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim([0, 0.06])  # 设置y轴的范围

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # 评估模型
    model.eval()  # 设置模型为评估模式
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 打印每个类别的准确率
    for i in range(3):
        accuracy = 100 * class_correct[i] / class_total[i]
        fold_class_accuracies[classes[i]].append(accuracy)
        print(f'Accuracy of {classes[i]:5s} : {accuracy:.2f} %')

    # 计算总体准确率并打印
    val_accuracy = 100 * total_correct / total_samples
    fold_total_accuracies.append(val_accuracy)
    print(f'Total accuracy: {val_accuracy:.2f}%')
    print('--------------------------------')

# 输出所有折的平均准确率和标准差
print(f'K-Fold Cross-Validation results:')
for class_name in classes:
    avg_accuracy = np.mean(fold_class_accuracies[class_name])
    std_accuracy = np.std(fold_class_accuracies[class_name])
    print(f'{class_name} Accuracy: {avg_accuracy:.2f} % ± {std_accuracy:.2f} %')

avg_total_accuracy = np.mean(fold_total_accuracies)
std_total_accuracy = np.std(fold_total_accuracies)
print(f'Total Accuracy: {avg_total_accuracy:.2f} % ± {std_total_accuracy:.2f} %')

# 保存模型
torch.save(model.state_dict(), 'facial_expression_model.pth')
