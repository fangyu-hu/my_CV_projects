import torch
from data_loader import load_data
from model_loader import LinkNet
from torch import nn
from torch import optim
import tqdm
import os

# 环境变量的配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_dl, test_dl = load_data()

# 加载模型
model = LinkNet()
model = model.to(device=device)

# 训练的相关配置
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.7)

# 开始进行训练
for epoch in range(100):
    train_tqdm = tqdm.tqdm(iterable=train_dl, total=len(train_dl))
    train_tqdm.set_description_str('Train epoch: {:3d}'.format(epoch))
    train_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
    train_iou_sum = torch.tensor(data=[], dtype=torch.float, device=device)
    for train_images, train_labels in train_tqdm:
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        pred = model(train_images)
        loss = loss_fn(pred, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            intersection = torch.logical_and(input=train_labels, other=torch.argmax(input=pred, dim=1))
            union = torch.logical_or(input=train_labels, other=torch.argmax(input=pred, dim=1))
            batch_iou = torch.true_divide(torch.sum(intersection), torch.sum(union))

            train_iou_sum = torch.cat([train_iou_sum, torch.unsqueeze(input=batch_iou, dim=-1)], dim=-1)
            train_loss_sum = torch.cat([train_loss_sum, torch.unsqueeze(input=loss, dim=-1)], dim=-1)
            train_tqdm.set_postfix({
                'train loss': train_loss_sum.mean().item(),
                'train iou': train_iou_sum.mean().item()
            })
    train_tqdm.close()

    lr_scheduler.step()

    with torch.no_grad():
        test_tqdm = tqdm.tqdm(iterable=test_dl, total=len(test_dl))
        test_tqdm.set_description_str('Test epoch: {:3d}'.format(epoch))
        test_loss_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        test_iou_sum = torch.tensor(data=[], dtype=torch.float, device=device)
        for test_images, test_labels in test_tqdm:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            test_pred = model(test_images)
            test_loss = loss_fn(test_pred.softmax(dim=1), test_labels)

            test_intersection = torch.logical_and(input=test_labels, other=torch.argmax(input=test_pred, dim=1))
            test_union = torch.logical_or(input=test_labels, other=torch.argmax(input=test_pred, dim=1))
            test_batch_iou = torch.true_divide(torch.sum(test_intersection), torch.sum(test_union))

            test_iou_sum = torch.cat([test_iou_sum, torch.unsqueeze(input=test_batch_iou, dim=-1)], dim=-1)
            test_loss_sum = torch.cat([test_loss_sum, torch.unsqueeze(input=test_loss, dim=-1)], dim=-1)
            test_tqdm.set_postfix({
                'test loss': test_loss_sum.mean().item(),
                'test iou': test_iou_sum.mean().item()
            })
        test_tqdm.close()

# 模型的保存
if not os.path.exists(os.path.join('model_data')):
    os.mkdir(os.path.join('model_data'))
torch.save(model.state_dict(), os.path.join('model_data', 'model.pth'))

