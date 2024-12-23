import torch
import math
from tqdm import tqdm

L = 200
bata_max = 50


def addNoise(images, device):
    sigma = torch.logspace(start=math.log10(bata_max), end=-2, steps=L, base=10).to(device)
    smalll = torch.randint(L, size=(images.size()[0],)).to(device)
    sigmal = sigma[smalll]
    output = torch.zeros(size=images.size()).to(device)
    dim1 = images.size()[2]
    dim2 = images.size()[3]

    for i in range(images.size()[0]):
        output[i, 0, :] = images[i, 0, :] + sigmal[i] * torch.randn(size=(dim1, dim2)).to(device)

    return output, sigmal


def criterion(images, inputs, outputs, sigma):
    sigma = sigma.reshape(sigma.size()[0], 1, 1, 1)
    temp = torch.mul(outputs, sigma)
    temp2 = (images - inputs) / sigma
    loss = torch.nn.functional.mse_loss(temp, temp2)
    return loss


# バッチサイズ１
def sampling(T, epsilon, dim1, dim2, model, device):
    sigma = torch.logspace(start=math.log10(bata_max), end=-2, steps=L, base=10).to(device)
    x0 = sigma[0] * torch.randn(size=(1, 1, dim1, dim2)).to(device)
    xt = x0
    for i in tqdm(range(L)):
        alphai = epsilon * sigma[i] ** 2 / sigma[L - 1] ** 2
        for t in range(T):
            zt = torch.randn(size=(dim1, dim2)).to(device)
            xt = xt + alphai * model(xt, sigma[i : i + 1]) / 2 + torch.sqrt(alphai) * zt
        # print(i)
    return xt


def posterior_sampling(T, epsilon, delta, dim1, dim2, model, y, A, sigmab, device):
    sigma = torch.logspace(start=math.log10(bata_max), end=-2, steps=L, base=10).to(device)
    x0 = sigma[0] * torch.randn(size=(1, 1, dim1, dim2)).to(device)
    xt = x0
    for i in tqdm(range(L)):
        alphai = epsilon * sigma[i] ** 2 / sigma[L - 1] ** 2
        ganmai = delta * sigma[i] ** 2 / sigma[L - 1] ** 2
        for t in range(T):
            zt = torch.randn(size=(dim1, dim2)).to(device)
            xt = xt + alphai * (model(xt, sigma[i : i + 1]) + (torch.mm(A.T, (y - torch.mm(A, xt.reshape(xt.numel(), 1)))) / (ganmai**2 + sigmab**2)).reshape(xt.size())) / 2 + torch.sqrt(alphai) * zt
        # print(i)
    return xt


# バッチサイズ任意
def sampling2(T, epsilon, batch, dim1, dim2, model, device):
    sigma = torch.logspace(start=math.log10(bata_max), end=-2, steps=L, base=10).to(device)
    x0 = sigma[0] * torch.randn(size=(batch, 1, dim1, dim2)).to(device)
    xt = x0
    for i in tqdm(range(L)):
        alphai = epsilon * sigma[i] ** 2 / sigma[L - 1] ** 2
        for t in range(T):
            zt = torch.randn(size=(batch, 1, dim1, dim2)).to(device)
            score = model(xt, sigma[i : i + 1])
            xt = xt + alphai * score / 2 + torch.sqrt(alphai) * zt
        # print(i)
    return xt


def posterior_sampling2(T, epsilon, delta, batch, dim1, dim2, model, y, A, sigmab, device):
    sigma = torch.logspace(start=math.log10(bata_max), end=-2, steps=L, base=10)
    x0 = sigma[0] * torch.randn(size=(batch, 1, dim1, dim2))
    xt = x0
    for i in tqdm(range(L)):
        alphai = epsilon * sigma[i] ** 2 / sigma[L - 1] ** 2
        ganmai = delta * sigma[i] ** 2 / sigma[L - 1] ** 2 + sigma[i]
        for t in range(T):
            zt = torch.randn(size=(batch, 1, dim1, dim2))
            with torch.cuda.amp.autocast():
                score = model(xt.to(device), sigma[i : i + 1].to(device)).cpu()
            xt = xt + alphai * (score + (torch.mm(A.T, (y - torch.mm(A, xt.reshape(batch, dim1 * dim2).permute(1, 0)))) / (ganmai**2 + sigmab**2)).permute(1, 0).reshape(xt.size())) / 2 + torch.sqrt(alphai) * zt
    return xt


# マスクの生成
def gen_initial_mask_index(patchsize, dim1, dim2):
    Ablockindex1 = []
    Ablockindex2 = []
    n_block = int(dim1 * dim2 / patchsize**2)  # ブロックの数
    for i in range(n_block):
        if (i // (dim2 / patchsize)) % 2 == 0:
            if i % 2 == 0:
                Ablockindex1.append(i)
            else:
                Ablockindex2.append(i)
        else:
            if i % 2 == 0:
                Ablockindex2.append(i)
            else:
                Ablockindex1.append(i)

    Ablockindex1 = torch.tensor(Ablockindex1)
    Ablockindex2 = torch.tensor(Ablockindex2)

    return Ablockindex1, Ablockindex2


def visualize_mask(Ablockindex, patchsize, dim1, dim2):
    A_mask = torch.zeros(size=(dim1, dim2))
    for i in range(len(Ablockindex)):
        index = Ablockindex[i]
        a = index // (dim1 / patchsize)
        b = index % (dim2 / patchsize)
        for j in range(patchsize):
            for k in range(patchsize):
                A_mask[int(patchsize * a + j), int(patchsize * b + k)] = 1
    return A_mask


def gen_mask_matrix(A_mask, M):
    Aindex = torch.where(A_mask.reshape(A_mask.numel()) == 1)
    A = torch.zeros(size=(M, A_mask.numel()))
    for i in range(M):
        A[i, Aindex[0][i]] = 1
    return A, Aindex


# 初期マスクの生成
def gen_initialized_masks(patchsizes, dim1, dim2):
    initialized_masks = []
    initialized_masks_index = []
    visualized_masks = []
    for i in range(patchsizes.size()[0]):
        Ablockindex1, Ablockindex2 = gen_initial_mask_index(patchsizes[i], dim1, dim2)
        A_mask1 = visualize_mask(Ablockindex1, patchsizes[i], dim1, dim2)
        A_mask2 = visualize_mask(Ablockindex2, patchsizes[i], dim1, dim2)
        A1, A1index = gen_mask_matrix(A_mask1, len(Ablockindex1) * patchsizes[i] ** 2)
        A2, A2index = gen_mask_matrix(A_mask2, len(Ablockindex2) * patchsizes[i] ** 2)
        A1 = A1
        A2 = A2
        initialized_masks.append(A1)
        initialized_masks.append(A2)
        visualized_masks.append(A_mask1)
        visualized_masks.append(A_mask2)
        initialized_masks_index.append(A1index)
        initialized_masks_index.append(A2index)
    return initialized_masks, visualized_masks, initialized_masks_index


def anomaly_detection(sigmab, N, A, x, batchsize, device, Aindex, T, epsilon, delta, dim1, dim2, model):
    # y=Ax+b(欠損とノイズ付与)
    b = sigmab * torch.randn(size=(N, 1)).cpu()
    y = torch.mm(A, x.reshape(x.numel(), 1).cpu() + b)
    ones = torch.ones((1, batchsize))
    Y = y * ones

    # yの可視化
    x_miss = torch.zeros((N, 1))
    x_vec = x.reshape((N, 1)) + b
    for i in range(N):
        if i in Aindex[0]:
            x_miss[i] = x_vec[i]

    # 条件付確率からサンプリング
    with torch.no_grad():
        xhat = posterior_sampling2(T, epsilon, delta, batchsize, dim1, dim2, model, Y, A, sigmab, device)

    ano = torch.sum((xhat - x) ** 2, dim=0, keepdim=True) / batchsize

    return ano, x_miss, xhat


if __name__ == "__main__":
    sigma = torch.logspace(start=math.log10(2), end=-2, steps=L, base=10)
    print(sigma)
