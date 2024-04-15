import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



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
    #transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the training data
train_data = datasets.ImageFolder(root='cats/test', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Load the testing data
test_data = datasets.ImageFolder(root='cats/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)






class GoogLeNetV3(nn.Module):
    def __init__(self, num_classes=3, aux_logits=True, init_weights=False):
        super(GoogLeNetV3, self).__init__()
        self.aux_logits = aux_logits
        self.conv1_1 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv1_2 = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv1_3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 80, kernel_size=3)
        self.conv3 = BasicConv2d(80, 192, kernel_size=3, stride=2)
        self.conv4 = BasicConv2d(192, 192, kernel_size=3, padding=1)

        self.inception3a = InceptionV3A(192, 64, 48, 64, 64, 96, 32)
        self.inception3b = InceptionV3A(256, 64, 48, 64, 64, 96, 64)
        self.inception3c = InceptionV3A(288, 64, 48, 64, 64, 96, 64)

        self.inception4a = InceptionV3D(288, 0, 384, 384, 64, 96, 0)
        self.inception4b = InceptionV3B(768, 192, 128, 192, 128, 192, 192)
        self.inception4c = InceptionV3B(768, 192, 160, 192, 160, 192, 192)
        self.inception4d = InceptionV3B(768, 192, 160, 192, 160, 192, 192)
        self.inception4e = InceptionV3D(768, 0, 384, 384, 64, 128, 0)

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
        # N x 3 x 299 x 299
        x = self.conv1_1(x)
        # N x 32 x 149 x 149
        x = self.conv1_2(x)
        # N x 32 x 147 x 147
        x = self.conv1_3(x)
        #  N x 32 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.conv2(x)
        # N x 80 x 71 x 71
        x = self.conv3(x)
        # N x 192 x 35 x 35
        x = self.conv4(x)
        # N x 192 x 35 x 35
        x = self.inception3a(x)
        # N x 256 x 35 x 35
        x = self.inception3b(x)
        # N x 288 x 35 x 35
        x = self.inception3c(x)
        # N x 288 x 35x 35
        x = self.inception4a(x)
        # N x 768 x 17 x 17
        x = self.inception4b(x)
        # N x 768 x 17 x 17
        x = self.inception4c(x)
        # N x 768 x 17 x 17
        x = self.inception4d(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:    # eval model lose this layer
            aux = self.aux(x)
        # N x 768 x 17 x 17
        x = self.inception4e(x)
        # N x 1280 x 8 x 8
        x = self.inception5a(x)
        # N x 2048 x 8 x 8
        x = self.inception5b(x)
        # N x 2048 x 7 x 7
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000(num_classes)
        if self.training and self.aux_logits:  
            return x, aux
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# InceptionV3A:BasicConv2d+MaxPool2d
class InceptionV3A(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3A, self).__init__()
        # 1×1
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1+3×3
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   
        )
        # 1×1++3×3+3×3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=3, padding=1)         
        )
        # 3×3+1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 拼接
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# InceptionV3B:BasicConv2d+MaxPool2d
class InceptionV3B(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3B, self).__init__()
        # 1×1
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1+1×3+3×1
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3, ch3x3, kernel_size=[3, 1], padding=[1, 0])   
        )
        # 1×1+1×3+3×1+1×3+3×1
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1]),
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0])  
        )
        # 3×3+1×1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# InceptionV3C:BasicConv2d+MaxPool2d
class InceptionV3C(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3C, self).__init__()
        # 1×1
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        # 1×1+1×3+3×1
        self.branch2_0 = BasicConv2d(in_channels, ch3x3red, kernel_size=1)
        self.branch2_1 = BasicConv2d(ch3x3red, ch3x3, kernel_size=[1, 3], padding=[0, 1])
        self.branch2_2 = BasicConv2d(ch3x3red, ch3x3, kernel_size=[3, 1], padding=[1, 0])

        # 1×1+3×3+1×3+3×1
        self.branch3_0 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),
        )
        self.branch3_1 = BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[1, 3], padding=[0, 1])
        self.branch3_2 = BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=[3, 1], padding=[1, 0])


        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2_0 = self.branch2_0(x)
        branch2 = torch.cat([self.branch2_1(branch2_0), self.branch2_2(branch2_0)], dim=1)
        branch3_0 = self.branch3_0(x)
        branch3 = torch.cat([self.branch3_1(branch3_0), self.branch3_2(branch3_0)], dim=1)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# InceptionV3D:BasicConv2d+MaxPool2d
class InceptionV3D(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3redX2, ch3x3X2, pool_proj):
        super(InceptionV3D, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, stride=2)
        )
        # 1×1+3×3+3×3
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3redX2, kernel_size=1),
            BasicConv2d(ch3x3redX2, ch3x3X2, kernel_size=3, padding=1),   # 保证输出大小等于输入大小
            BasicConv2d(ch3x3X2, ch3x3X2, kernel_size=3, stride=2)
        )
        # 3×3
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2))
        # pool_proj

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1,branch2, branch3]
        return torch.cat(outputs, 1)

# AvgPool2d+BasicConv2d+Linear+dropout
class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()

        self.averagePool = nn.AvgPool2d(kernel_size=3, stride=3)
        self.conv1 = BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.conv2 = BasicConv2d(in_channels=128, out_channels=768, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(p=0.7)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Add this line
        self.linear = nn.Linear(in_features=768, out_features=out_channels)
    def forward(self, x):
        # N x 768 x 17 x 17
        x = self.averagePool(x)
        # N x 768 x 5 x 5
        x = self.conv1(x)
        # N x 128 x 5 x 5
        x = self.conv2(x)
        x = self.adaptive_pool(x)  # Add this line
        x = torch.flatten(x, 1)
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        out = self.linear(self.dropout(x))
        # N x num_classes
        return out

#  Conv2d+BN+ReLU
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GoogLeNetV3(num_classes=3, aux_logits=True, init_weights=True).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

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

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_acc}')



model.eval()  # Set model to evaluation mode
correct = 0
total = 0
num_classes = 3
# Initialize the confusion matrix
confusion_matrix = torch.zeros(num_classes, num_classes)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
                 confusion_matrix[t.long(), p.long()] += 1

print(f'Test Accuracy: {100 * correct / total}%')

for i in range(num_classes):
    correct = confusion_matrix[i, i]
    total = confusion_matrix[i].sum()
    accuracy = 100 * correct / total
    print('Accuracy of class {}: {:.2f}%'.format(i, accuracy))

print('Finished Training')


torch.save(model.state_dict(), 'facial_expression_model.pth')

