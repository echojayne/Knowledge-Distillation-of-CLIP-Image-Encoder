from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

image_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(image_mean, image_std)
])

def get_datasets(image_transform, batch_size, dataset_name='cifar100'):
    """
    Returns train and test datasets and dataloaders

    Args:
        image_transform: torchvision.transforms
        batch_size: int

    Returns:
        train_loader: torch.utils.data.DataLoader
        test_loader: torch.utils.data.DataLoader
    """
    if dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100('path/to/cifar100', 'train', transform=image_transform, download=False)
        test_dataset = datasets.CIFAR100('path/to/cifar100', 'test', transform=image_transform, download=False)
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10('path/to/cifar100', 'train', transform=image_transform, download=True)
        test_dataset = datasets.CIFAR10('path/to/cifar100', 'test', transform=image_transform, download=True)
    elif dataset_name == 'stl10':
        train_dataset = datasets.STL10('path/to/stl10', 'train', transform=image_transform, download=True)
        test_dataset = datasets.STL10('path/to/stl10', 'test', transform=image_transform, download=True)
    elif dataset_name == 'imagenet':
        train_dataset = datasets.ImageNet('path/to/imagenet', 'train', transform=image_transform, download=True)
        test_dataset = datasets.ImageNet('path/to/imagenet', 'val', transform=image_transform, download=True)
    elif dataset_name == 'voc2007':
        train_dataset = datasets.VOCDetection('path/to/VOC2007', 'train', transform=image_transform, download=True)
        test_dataset = datasets.VOCDetection('path/to/VOC2007', 'val', transform=image_transform, download=True)
    elif dataset_name == 'Birdsnap':
        train_dataset = datasets.ImageFolder('path/to/Birdsnap', transform=image_transform)
        test_dataset = datasets.ImageFolder('path/to/Birdsnap', transform=image_transform)
        
    else:
        raise ValueError('Invalid dataset name')

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_datasets(image_transform, 32, 'stl10')
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape, labels.shape)
        if i == 0:
            break
    for i, (images, labels) in enumerate(test_loader):
        print(images.shape, labels.shape)
        if i == 0:
            break