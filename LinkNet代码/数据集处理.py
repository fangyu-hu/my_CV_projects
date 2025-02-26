import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks


class MaskDataset(data.Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        super(MaskDataset, self).__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.mask_paths[index]

        pil_img = Image.open(image_path)
        pil_img = pil_img.convert('RGB')
        img_tensor = self.transform(pil_img)

        pil_label = Image.open(label_path)
        label_tensor = self.transform(pil_label)
        label_tensor[label_tensor > 0] = 1
        label_tensor = torch.squeeze(input=label_tensor).type(torch.LongTensor)

        return img_tensor, label_tensor

    def __len__(self):
        return len(self.mask_paths)


def load_data():
    
    DATASET_PATH = r'D:/mask/mask'
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'training')
    TEST_DATASET_PATH = os.path.join(DATASET_PATH, 'testing')

    train_file_names = os.listdir(TRAIN_DATASET_PATH)
    test_file_names = os.listdir(TEST_DATASET_PATH)

    train_image_names = [name for name in train_file_names if
                         'matte' in name and name.split('_')[0] + '.png' in train_file_names]
    train_image_paths = [os.path.join(TRAIN_DATASET_PATH, name.split('_')[0] + '.png') for name in
                         train_image_names]
    train_label_paths = [os.path.join(TRAIN_DATASET_PATH, name) for name in train_image_names]

    test_image_names = [name for name in test_file_names if
                        'matte' in name and name.split('_')[0] + '.png' in test_file_names]
    test_image_paths = [os.path.join(TEST_DATASET_PATH, name.split('_')[0] + '.png') for name in test_image_names]
    test_label_paths = [os.path.join(TEST_DATASET_PATH, name) for name in test_image_names]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    BATCH_SIZE = 8

    train_ds = MaskDataset(image_paths=train_image_paths, mask_paths=train_label_paths, transform=transform)
    test_ds = MaskDataset(image_paths=test_image_paths, mask_paths=test_label_paths, transform=transform)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=BATCH_SIZE)

    return train_dl, test_dl


if __name__ == '__main__':
    train_my, test_my = load_data()
    images, labels = next(iter(train_my))
    indexx = 5
    images = images[indexx]
    labels = labels[indexx]
    labels = torch.unsqueeze(input=labels, dim=0)

    result = draw_segmentation_masks(image=torch.as_tensor(data=images * 255, dtype=torch.uint8),
                                     masks=torch.as_tensor(data=labels, dtype=torch.bool),
                                     alpha=1.0, colors=['purple'])
    plt.imshow(result.permute(1, 2, 0).numpy())
    plt.show()

