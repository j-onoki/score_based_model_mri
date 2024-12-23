import NCSN
import function as f
import torch
import matplotlib.pyplot as plt
import random
import os
import data_load

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NCSN.UNet().to(device)

    model.load_state_dict(torch.load("model260.pth"))

    model.eval()

    # データのダウンロード
    test_dataset = data_load.load()
    x = test_dataset[1][40][0]

    batchsize = 5
    N = x.numel()
    M = 30000
    dim1 = 320
    dim2 = 320
    sigmab = 0.01
    epsilon = 0.00002
    delta = 0.00002
    T = 5
    patchsize = 4
    patchsizes = torch.tensor([8, 16, 32, 64])

    # 初期マスクの生成
    initialized_masks, visualized_masks, initialized_masks_index = f.gen_initialized_masks(patchsizes, dim1, dim2)

    A = initialized_masks[5]
    visualized_A = visualized_masks[5]
    x_miss = visualized_A * x

    # # 欠損行列Aの生成(ピクセル単位で欠損)
    # Aindex = []
    # while len(Aindex) < M:
    #     n = random.randint(0, N - 1)
    #     if n not in Aindex:
    #         Aindex.append(n)
    # Aindex = torch.tensor(Aindex)

    # A = torch.zeros(size=(M, N))
    # for i in range(M):
    #     A[i, Aindex[i]] = 1

    # y=Ax+b(欠損とノイズ付与)
    b = sigmab * torch.randn(size=(N, 1))
    y = torch.mm(A, x.reshape(x.numel(), 1) + b)
    ones = torch.ones((1, batchsize))
    Y = y * ones

    # # Aの可視化
    # A_mask = torch.zeros((N,))
    # x_miss = torch.zeros((N,))
    # x_vec = x.reshape((N, 1)) + b
    # for i in range(N):
    #     if i in Aindex:
    #         x_miss[i] = x_vec[i]
    #         A_mask[i] = 1

    with torch.no_grad():
        xhat = f.posterior_sampling2(T, epsilon, delta, batchsize, dim1, dim2, model, Y, A, sigmab, device)

    plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.imshow(x.cpu().reshape(dim1, dim1), cmap="gray")
    plt.savefig("./image/PosteriorSampling/test_x.png", bbox_inches="tight")

    plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.imshow(torch.abs(x_miss).cpu().reshape(dim1, dim1), cmap="gray")
    plt.savefig("./image/PosteriorSampling/test_x_miss.png", bbox_inches="tight")

    plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.imshow(visualized_A.cpu().reshape(dim1, dim1), cmap="gray")
    plt.savefig("./image/PosteriorSampling/test_A_mask.png", bbox_inches="tight")

    plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.imshow(xhat[0].cpu().reshape(dim1, dim1), cmap="gray")
    plt.savefig("./image/PosteriorSampling/test_xhat.png", bbox_inches="tight")

    n = batchsize
    n1 = 1
    n2 = 5
    fig, ax = plt.subplots(n1, n2, gridspec_kw=dict(wspace=0.1, hspace=0.1), figsize=(16, 2))
    for i in range(n):
        ax[i].imshow(xhat[i].cpu().reshape(dim1, dim1), cmap="gray")
        ax[i].axes.xaxis.set_visible(False)
        ax[i].axes.yaxis.set_visible(False)
    plt.savefig("./image/PosteriorSampling/test_xhats.png", bbox_inches="tight")
