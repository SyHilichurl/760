import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np

# 导入模型
from v3 import GoogLeNetV3

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并将其移到指定的设备上
model = GoogLeNetV3(num_classes=3, aux_logits=True, init_weights=False).to(device)

# 加载模型权重
model.load_state_dict(torch.load('facial_expression_model.pth', map_location=device))

# 将模型设置为评估模式
model.eval()

# 定义图像转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载并转换图像
img_path = "E:/PyCharm/workspace/archive 2/Sad/aug-92-009.jpg"  # 更新为你的图像路径
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换颜色空间
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    # GoogLeNet可能返回一个tuple，因此我们需要主输出
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    probabilities = F.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    confidence_scores = probabilities[0].cpu().numpy()

emotion_classes = {0: 'happy', 1: 'sad', 2: 'angry'}
predicted_emotion = emotion_classes[predicted_class.item()]

print(f"Predicted Emotion: {predicted_emotion}")
print(f"Confidence [happy, sad, angry]: {confidence_scores}")

# 格式化输出每个类别的置信度
confidence_formatted = " ".join([f"{class_name}: {score:.8f}" for class_name, score in zip(emotion_classes.values(), confidence_scores)])
print(confidence_formatted)
