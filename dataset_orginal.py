import glob
import os
import pickle
import re

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
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

    def __init__(self, logger, config, path_list, blacklist, slice_len_list, transform=None, return_file_path=False):
        self.config = config
        self.logger = logger
        self.path_list = path_list
        self.blacklist = blacklist
        self.transform = transform
        self.image_list = []
        self.slice_list = []
        self.file_list = []
        self.return_file_path = return_file_path

        print("Import Dataset ...")
        self.logger.info(f"blacklist: \n {self.blacklist}")
        for file in tqdm(self.path_list):
            image = Image.open(file).convert("L")
            image_tensor = self.transform(image)

            key = file.split("_slice")[0].replace("images/", "")
            slice = int(re.search(r"slice(?P<slice>[0-9]+)", file).group("slice"))
            if slice >= 5 and slice < int(slice_len_list[key]) - 5:
                if os.path.basename(key) in self.blacklist:
                    self.logger.info(f"ignore file: {file}")
                else:
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
    for dir in data_dir:
        print(f"Import images in {os.path.basename(dir)}")
        file_list += glob.glob(f"{dir}/images/*.png")
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


def get_dataset(config, logger, data_dir, mode, return_file_path=False):
    """Create and Get Dataset
    ```
    get_dataset(args: Namespace, config: Namespace, mode: str) -> tuple(train_dataset: Dataset, test_dataset: Dataset)
    ```
    """
    train_transform, test_transform = get_transform(config)

    data_paths = search_image(data_dir)
    blacklist = []
    slice_list = dict()
    for dir in data_dir:
        with open(f"{dir}/blacklist.txt", "r") as f:
            blacklist += [s.rstrip() for s in f.readlines()]
        with open(f"{dir}/images/slices.pkl", "rb") as f:
            data = pickle.load(f)
            slice_list = {**slice_list, **data}

    if mode == "train" or "validation":
        train_dataset = MRIDataset(
            logger,
            config,
            data_paths,
            blacklist,
            slice_list,
            train_transform,
            return_file_path,
        )
        return train_dataset
    elif mode == "test":
        test_dataset = MRIDataset(
            logger,
            config,
            data_paths,
            blacklist,
            slice_list,
            test_transform,
            return_file_path,
        )
        return test_dataset
    else:
        raise ValueError(f"Mode:{mode} is not supported.")
