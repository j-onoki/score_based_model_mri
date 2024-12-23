import glob
import os
import pickle
import re

from natsort import natsorted
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from transformer import get_transform


class MRIDataset(Dataset):
    """Create MRI Dataset
    ```
    class MRIDataset(
        path_list(List): List of dataset path
        transform(transforms.Compose): Preprocessing flow
        return_path(Bool): Whether the image path is included in the return value
        config(Namespace): "Namespace of config"
    )
    ```
    """

    def __init__(self, path_list, transform=None, return_file_path=False):  # slice_len_list, transform=None, return_file_path=False):
        self.path_list = path_list
        self.transform = transform
        self.image_list = []
        self.slice_list = []
        self.file_list = []
        self.return_file_path = return_file_path

        print("Import Dataset ...")
        for file in tqdm(self.path_list):
            image = Image.open(file).convert("L")
            image_tensor = self.transform(image)

            # key = file.split("_slice")[0].replace("images/", "")
            slice = int(re.search(r"slice(?P<slice>[0-9]+)", file).group("slice"))
            # if slice >= 60 and slice < int(slice_len_list[key]) - 25:
            self.image_list.append(image_tensor)
            self.slice_list.append(slice)
            self.file_list.append(file)

        print(f"{len(self.image_list)} images is loaded.")
        print(f"{len(self.slice_list)} slice data is made.")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        slices = torch.tensor(self.slice_list[index])

        if self.return_file_path:
            return self.image_list[index], slices, self.file_list[index]

        return self.image_list[index], slices


def search_image(data_dir):
    r"""search image paths in directory
    ```
    search_image(data_dir: str) -> list
    ```

    Args:
        data_dir(str): Paths of directories
    """
    file_list = []

    print(f"Import images in {os.path.basename(data_dir)}")
    file_list += natsorted(glob.glob(f"{data_dir}/images/**/*.png"))
    if len(file_list) == 0:
        file_list += natsorted(glob.glob(f"{data_dir}/images/*.png"))
    # file_list.sort()
    return file_list


def inverse_data_transform(config):
    r"""inverse data transform
    ```
    inverse_data_transform(config: Namespace) -> func
    ```

    Args:
        config(Namespace): "Namespace of config"
            This function uses following values in `config`

                config.data.inverse: Whether require inverse transform before imshow
    """
    if config.data.inverse:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def get_dataset(data_dir, mode, return_file_path=False):
    """Create and Get Dataset
    ```
    get_dataset(args: Namespace, config: Namespace, mode: str) -> tuple(train_dataset: Dataset, test_dataset: Dataset)
    ```
    """
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()

    data_paths = search_image(data_dir)
    # slice_list = dict()
    # with open(f"{data_dir}/images/slices.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     slice_list = {**slice_list, **data}

    if mode == "train" or "validation":
        train_dataset = MRIDataset(
            data_paths,
            # slice_list,
            train_transform,
            return_file_path,
        )
        return train_dataset
    elif mode == "test":
        test_dataset = MRIDataset(
            data_paths,
            # slice_list,
            test_transform,
            return_file_path,
        )
        return test_dataset
    else:
        raise ValueError(f"Mode:{mode} is not supported.")
