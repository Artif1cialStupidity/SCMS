# data/cifar.py

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Standard normalization constants for CIFAR datasets
# Calculated from the training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616) # Alt: (0.2023, 0.1994, 0.2010) often used

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def get_cifar_transforms(dataset_name: str, augment: bool = True):
    """Gets the appropriate transforms for CIFAR datasets."""
    if dataset_name == 'cifar10':
        mean = CIFAR10_MEAN
        std = CIFAR10_STD
    elif dataset_name == 'cifar100':
        mean = CIFAR100_MEAN
        std = CIFAR100_STD
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Base transform: Convert to tensor and normalize
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    # Augmentation for training data
    if augment:
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + transform_list
        train_transform = transforms.Compose(train_transform_list)
    else:
        # Use simple transform if no augmentation needed (e.g., for testing)
         train_transform = transforms.Compose(transform_list)

    # Test transform (no augmentation)
    test_transform = transforms.Compose(transform_list)

    return train_transform, test_transform


def get_cifar_loaders(dataset_name: str,
                        batch_size: int,
                        data_dir: str = './data_files',
                        num_workers: int = 4,
                        augment_train: bool = True,
                        pin_memory: bool = True) -> tuple[DataLoader, DataLoader]:
    """
    Loads CIFAR-10 or CIFAR-100 dataset and returns training and testing DataLoaders.

    Args:
        dataset_name (str): 'cifar10' or 'cifar100'.
        batch_size (int): The batch size for the DataLoaders.
        data_dir (str): Directory where the dataset is stored or will be downloaded.
        num_workers (int): Number of subprocesses to use for data loading.
        augment_train (bool): Whether to apply data augmentation to the training set.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory
                           before returning them.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader
                                       and the testing DataLoader.
    """
    if dataset_name not in ['cifar10', 'cifar100']:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose 'cifar10' or 'cifar100'.")

    train_transform, test_transform = get_cifar_transforms(dataset_name, augment=augment_train)

    if dataset_name == 'cifar10':
        DatasetClass = torchvision.datasets.CIFAR10
    else: # dataset_name == 'cifar100'
        DatasetClass = torchvision.datasets.CIFAR100

    # Load training dataset
    train_dataset = DatasetClass(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    # Load testing dataset
    test_dataset = DatasetClass(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, # Can use larger batch size for testing if memory allows
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Loaded {dataset_name.upper()}:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Testing samples: {len(test_dataset)}")
    print(f"  Training loader batches: {len(train_loader)}")
    print(f"  Testing loader batches: {len(test_loader)}")


    return train_loader, test_loader

# Example usage (for testing this file independently)
if __name__ == '__main__':
    print("Testing CIFAR-10 loading...")
    train_loader_10, test_loader_10 = get_cifar_loaders(
        dataset_name='cifar10',
        batch_size=64,
        data_dir='./data_cifar10',
        augment_train=True
    )

    # Fetch one batch to check shapes
    data_batch, labels_batch = next(iter(train_loader_10))
    print(f"CIFAR-10 Batch shape: {data_batch.shape}") # Should be [64, 3, 32, 32]
    print(f"CIFAR-10 Labels shape: {labels_batch.shape}") # Should be [64]

    print("\nTesting CIFAR-100 loading...")
    train_loader_100, test_loader_100 = get_cifar_loaders(
        dataset_name='cifar100',
        batch_size=128,
        data_dir='./data_cifar100',
        augment_train=False # Example without augmentation
    )

    data_batch_100, labels_batch_100 = next(iter(test_loader_100))
    print(f"CIFAR-100 Batch shape: {data_batch_100.shape}")
    print(f"CIFAR-100 Labels shape: {labels_batch_100.shape}")