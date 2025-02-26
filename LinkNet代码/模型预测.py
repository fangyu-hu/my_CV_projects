import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks
from data_loader import load_data
from model_loader import LinkNet

# 数据的加载
train_dl, test_dl = load_data()

# 模型的加载
model = LinkNet()
model_state_dict = torch.load(os.path.join('model_data', 'model.pth'), map_location='cpu')
model.load_state_dict(model_state_dict)

# 开始进行预测
images, labels = next(iter(test_dl))
index = 2
with torch.no_grad():
    pred = model(images)
    pred = torch.argmax(input=pred, dim=1)
    result = draw_segmentation_masks(image=torch.as_tensor(data=images[index] * 255, dtype=torch.uint8),
                                     masks=torch.as_tensor(data=pred[index], dtype=torch.bool),
                                     alpha=1.0, colors=['purple'])
    plt.figure()
    plt.axis('off')
    plt.imshow(result.permute(1, 2, 0))
    plt.savefig('result.png')
    plt.show()

