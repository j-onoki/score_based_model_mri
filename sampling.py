import NCSN
import function as f
import torch
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NCSN.UNet().to(device)

    model.load_state_dict(torch.load("model260.pth"))

    batchsize = 10
    sigmab = 0.01
    epsilon = 0.00002
    T = 20
    dim1 = 320
    dim2 = 320

    # 条件付確率からサンプリング
    with torch.no_grad():
        x = f.sampling2(T, epsilon, batchsize, dim1, dim2, model, device)

    n = batchsize
    n1 = 1
    n2 = 10
    fig, ax = plt.subplots(n1, n2, gridspec_kw=dict(wspace=0.1, hspace=0.1), figsize=(28, 6))
    for i in range(n):
        ax[i].imshow(x[i].cpu().reshape(dim1, dim2), cmap="gray")
        ax[i].axes.xaxis.set_visible(False)
        ax[i].axes.yaxis.set_visible(False)
    plt.savefig("./image/knee_sampling.png")
