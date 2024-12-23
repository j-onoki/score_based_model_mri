import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

if __name__ == "__main__":
    img = nib.load("/Dataset/maki/brats/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz")
    img_data = img.get_fdata()
    print(img_data.shape)

    plt.figure()
    plt.imshow(img_data[:, :, 80, 0], cmap="gray", origin="lower")
    plt.savefig("image/demo5_0.png")
    # slice_pickle = pd.read_pickle("/Dataset/IXI_Dataset/IXI_T2/images/slices.pkl")

    # print(slice_pickle["/Dataset/IXI_Dataset/IXI_T2/IXI154-Guys-0821-T2.nii.gz"])
