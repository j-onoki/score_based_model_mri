import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import dataset


# データのロード
def load():
    # # 訓練データ
    # train_dataset = dataset.get_dataset("/Dataset/Coil/Dataset/knee_singlecoil_train/singlecoil_train", "train")

    # # 検証データ
    # test_dataset = dataset.get_dataset("/Dataset/Coil/Dataset/knee_singlecoil_val/singlecoil_val", "test")

    # 推論データ
    test_dataset = dataset.get_dataset("/Dataset/maki/brats/Task01_BrainTumour/imagesTr", "test")
    test_label = dataset.get_dataset("/Dataset/maki/brats/Task01_BrainTumour/labelsTr", "test")

    return test_dataset, test_label  # train_dataset, test_dataset


# ミニバッチの作成
def loader(train_dataset, test_dataset):
    batch_size = 8

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size // 2, shuffle=True)

    return train_loader, test_loader
