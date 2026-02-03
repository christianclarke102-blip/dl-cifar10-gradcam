from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_loaders(batch_size=128, num_workers=2, image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root="data", train=True, download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tf)

    val_size = 5000
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # Validation should not use augmentation
    val_set.dataset.transform = test_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=False)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,pin_memory=False)

    return train_loader, val_loader, test_loader
