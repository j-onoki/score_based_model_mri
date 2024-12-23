# conda environment: pt

import glob
import os
import sys

# sys.path.append("/home/sfukutomi/module")

import argparse
import pickle

import h5py
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def convert(array):
    max = array.max()
    min = array.min()

    return ((array - min) / (max - min)) * 255


def load_wrapper(args: argparse.Namespace, path: str) -> torch.Tensor:
    # h5data = h5py.File(path)
    # h5data_np = h5data["reconstruction_rss"][()]
    img = nib.load(path)
    img_data = np.transpose(img.get_fdata(), (2, 1, 0))

    h5data_tensor = torch.from_numpy(img_data)[:, None, ...].to(args.device)
    return h5data_tensor


def gen_image(args: argparse.Namespace, file_list: list) -> None:
    file_slice_list = dict()

    for filename in tqdm(file_list):
        image_tensor = load_wrapper(args, filename)
        file_slice_list[str(filename)] = image_tensor.shape[0]
        # print(file_slice_list)

        for slice in tqdm(range(image_tensor.shape[0]), leave=False):
            slice_array = convert(np.transpose(image_tensor[slice].to("cpu").numpy(), [1, 2, 0])[:, :, 0])
            pil_image = Image.fromarray(slice_array).convert("L")
            pil_image.save(f"/Dataset/IXI_Dataset/{args.path}/images/{os.path.basename(filename)}_slice{slice}.png")

    with open(f"/Dataset/IXI_Dataset/{args.path}/images/slices.pkl", "wb") as f:
        pickle.dump(file_slice_list, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="directory path")

    args = parser.parse_args()

    """Checking Cuda Availability"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    args.device = device

    if os.path.isdir(f"/Dataset/IXI_Dataset/{args.path}"):
        os.makedirs(f"/Dataset/IXI_Dataset/{args.path}/images", exist_ok=True)
        file_list = [file for file in glob.glob(f"/Dataset/IXI_Dataset/{args.path}/**/*.nii.gz", recursive=True) if os.path.isfile(file)]
    else:
        raise ValueError(f"path: /Dataset/IXI_Dataset/{args.path} is not directory.")

    gen_image(args, file_list)


if __name__ == "__main__":
    main()
