# dataset.py

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SEED

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),
    ])

    return train_tf, test_tf


def get_dataloaders(val_split=0.2):
    torch.manual_seed(SEED)

    train_tf, test_tf = get_transforms()

    full_train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tf)

    # Split training dataset into train and validation
    val_size = int(len(full_train_ds) * val_split)
    train_size = len(full_train_ds) - val_size

    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_dl, val_dl, test_dl, full_train_ds.classes
