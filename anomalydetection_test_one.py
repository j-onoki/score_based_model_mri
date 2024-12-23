import NCSN
import function as f
import torch
import matplotlib.pyplot as plt
import os
import data_load


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import AnomalyDetection as ad

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = NCSN.UNet().to(device)

    model.load_state_dict(torch.load("model550.pth"))

    # ano_images, ano_labels = data_load.load()
    # ano_image = torch.nn.functional.pad(ano_images[543][0], (40, 40, 40, 40))
    # ano_label = torch.nn.functional.pad(ano_labels[543][0], (40, 40, 40, 40))
    # print(ano_image.size())
    # torch.save(ano_image, "./image/SSM/exp2_knee/ano_image.pt")
    # torch.save(ano_label, "./image/SSM/exp2_knee/ano_label.pt")

    ano_image = torch.load("./image/SSM/exp2_knee/ano_image.pt")
    ano_label = torch.load("./image/SSM/exp2_knee/ano_label.pt")

    dim = 320
    plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
    plt.savefig("./image/SSM/exp2_knee/test_image.png")

    plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.imshow(ano_label.cpu().reshape(dim, dim), cmap="gray")
    plt.savefig("./image/SSM/exp2_knee/test_label.png")

    # パラメータ
    batchsize = 5
    N = ano_image.numel()
    M = 512
    dim1 = ano_image.size()[1]
    dim2 = ano_image.size()[2]
    sigmab = 0.01
    epsilon = 0.00002
    delta = 0.00002
    T = 20
    patchsizes = torch.tensor([10, 20, 40, 80])
    print(dim1)
    print(dim2)
    torch.save(ano_image, "./image/SSM/exp2_knee/ano_image.pt")
    torch.save(ano_label, "./image/SSM/exp2_knee/ano_label.pt")

    # 初期マスクの生成
    initialized_masks, visualized_masks, initialized_masks_index = f.gen_initialized_masks(patchsizes, dim1, dim2)

    # 異常検知（ステップ１）
    dir_path = "./image/SSM/exp2_knee/step1"
    os.mkdir(dir_path)
    ano_ini = torch.zeros(size=(patchsizes.size()[0] * 2, dim1, dim2))
    ano2_ini = torch.zeros(size=(patchsizes.size()[0] * 2, dim1, dim2))
    for i in range(patchsizes.size()[0] * 2):
        A = initialized_masks[i]
        A_mask = visualized_masks[i]
        Aindex = initialized_masks_index[i]

        # 復元と異常検知
        ano, x_miss, xhat = f.anomaly_detection(sigmab, N, A, ano_image, batchsize, device, Aindex, T, epsilon, delta, dim1, dim2, model)

        # ピクセルごとの分散
        mean = torch.sum(xhat, dim=0, keepdim=True) / batchsize
        res = xhat - mean
        res = res * res
        var = torch.sum(res, dim=0) / batchsize

        # AnomalyScore
        ano2 = ano / var
        ano_ini[i, :, :] = ano
        ano2_ini[i, :, :] = ano2

        # 各マスクの結果保存
        dir_path = "./image/SSM/exp2_knee/step1/mask" + str(i + 1)
        os.mkdir(dir_path)

        plt.figure()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.imshow(ano.cpu().reshape(dim, dim), cmap="gray")
        plt.savefig("./image/SSM/exp2_knee/step1/mask" + str(i + 1) + "/test_ano.png")
        plt.close()

        plt.figure()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.imshow(ano2.cpu().reshape(dim, dim), cmap="gray")
        plt.savefig("./image/SSM/exp2_knee/step1/mask" + str(i + 1) + "/test_ano2.png")
        plt.close()

        plt.figure()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
        plt.savefig("./image/SSM/exp2_knee/step1/mask" + str(i + 1) + "/test_x.png")
        plt.close()

        plt.figure()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.imshow(x_miss.cpu().reshape(dim, dim), cmap="gray")
        plt.savefig("./image/SSM/exp2_knee/step1/mask" + str(i + 1) + "/test_x_miss.png")
        plt.close()

        plt.figure()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.imshow(A_mask.cpu().reshape(dim, dim), cmap="gray")
        plt.savefig("./image/SSM/exp2_knee/step1/mask" + str(i + 1) + "/test_A_mask.png")
        plt.close()

        n = batchsize
        n1 = 1
        n2 = 5
        fig, ax = plt.subplots(n1, n2, gridspec_kw=dict(wspace=0.1, hspace=0.1), figsize=(32, 4))
        for j in range(n):
            ax[j].imshow(xhat[j].cpu().reshape(dim, dim), cmap="gray")
            ax[j].axes.xaxis.set_visible(False)
            ax[j].axes.yaxis.set_visible(False)
        plt.savefig("./image/SSM/exp2_knee/step1/mask" + str(i + 1) + "/test_xhat.png")
        plt.close()

        plt.figure()
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        plt.imshow(var.cpu().reshape(dim, dim), cmap="gray")
        plt.savefig("./image/SSM/exp2_knee/step1/mask" + str(i + 1) + "/test_xhat_var.png")
        plt.close()

    # 初期アノマリースコア
    ano_ini_mean = torch.sum(ano_ini, dim=0) / patchsizes.size()[0]
    ano2_ini_mean = torch.sum(ano2_ini, dim=0) / patchsizes.size()[0]

    plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.imshow(ano_ini_mean.cpu().reshape(dim, dim), cmap="gray")
    plt.savefig("./image/SSM/exp2_knee/step1/test_ano_ini_mean.png")
    plt.close()

    plt.figure()
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    plt.imshow(ano2_ini_mean.cpu().reshape(dim, dim), cmap="gray")
    plt.savefig("./image/SSM/exp2_knee/step1/test_ano2_ini_mean.png")
    plt.close()

    # 一応ptファイルでも保存
    torch.save(ano_ini_mean, "./image/SSM/exp2_knee/step1/test_ano_ini_mean.pt")
    torch.save(ano2_ini_mean, "./image/SSM/exp2_knee/step1/test_ano2_ini_mean.pt")

    print(torch.max(ano_ini_mean))
    print(torch.max(ano2_ini_mean))

    ano_ini_mean = torch.load("./image/SSM/exp2_knee/step1/test_ano_ini_mean.pt")
    ano2_ini_mean = torch.load("./image/SSM/exp2_knee/step1/test_ano2_ini_mean.pt")

    # Step2以降
    eta = 0.01
    max_step = 10
    previous_mask = visualized_masks[0]
    ano = ano_ini_mean.reshape(1, 1, dim1, dim2)

    with torch.no_grad():
        for step in range(max_step):

            # マスクの更新
            refined_mask = torch.ones(size=(dim1, dim2))
            M = 0
            # eta = torch.max(ano) / 4
            for i in range(dim1):
                for j in range(dim2):
                    if ano[:, :, i, j] > eta:
                        refined_mask[i, j] = 0
                    else:
                        M += 1

            refined_mask_matrix, refined_mask_index = f.gen_mask_matrix(refined_mask, M)

            if torch.mean(refined_mask - previous_mask) == 0:
                print("mask does not change")
                break

            # 復元と異常検知
            ano, x_miss, xhat = f.anomaly_detection(sigmab, N, refined_mask_matrix, ano_image, batchsize, device, refined_mask_index, T, epsilon, delta, dim1, dim2, model)

            # ピクセルごとの分散
            mean = torch.sum(xhat, dim=0, keepdim=True) / batchsize
            res = xhat - mean
            res = res * res
            var = torch.sum(res, dim=0) / batchsize

            # AnomalyScore
            ano2 = ano / var

            # マスクの保持（次回ステップの比較用）
            previous_mask = refined_mask

            # 各ステップの結果保存
            dir_path = "./image/SSM/exp2_knee/step" + str(step + 2)
            os.mkdir(dir_path)

            plt.figure()
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            plt.imshow(ano.cpu().reshape(dim, dim), cmap="gray")
            plt.savefig("./image/SSM/exp2_knee/step" + str(step + 2) + "/test_ano.png")
            plt.close()

            plt.figure()
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            plt.imshow(ano2.cpu().reshape(dim, dim), cmap="gray")
            plt.savefig("./image/SSM/exp2_knee/step" + str(step + 2) + "/test_ano2.png")
            plt.close()

            plt.figure()
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
            plt.savefig("./image/SSM/exp2_knee/step" + str(step + 2) + "/test_x.png")
            plt.close()

            plt.figure()
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            plt.imshow(x_miss.cpu().reshape(dim, dim), cmap="gray")
            plt.savefig("./image/SSM/exp2_knee/step" + str(step + 2) + "/test_x_miss.png")
            plt.close()

            plt.figure()
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            plt.imshow(refined_mask.cpu().reshape(dim, dim), cmap="gray")
            plt.savefig("./image/SSM/exp2_knee/step" + str(step + 2) + "/test_A_mask.png")
            plt.close()

            n = batchsize
            n1 = 1
            n2 = 5
            fig, ax = plt.subplots(n1, n2, gridspec_kw=dict(wspace=0.1, hspace=0.1), figsize=(16, 2))
            for j in range(n):
                ax[j].imshow(xhat[j].cpu().reshape(dim, dim), cmap="gray")
                ax[j].axes.xaxis.set_visible(False)
                ax[j].axes.yaxis.set_visible(False)
            plt.savefig("./image/SSM/exp2_knee/step" + str(step + 2) + "/test_xhat.png")
            plt.close()

            plt.figure()
            plt.imshow(var.cpu().reshape(dim, dim), cmap="gray")
            plt.savefig("./image/SSM/exp2_knee/step" + str(step + 2) + "/test_xhat_var.png")
            plt.close()

            torch.save(ano, "./image/SSM/exp2_knee/step" + str(step + 2) + "/test_ano.pt")
            torch.save(ano2, "./image/SSM/exp2_knee/step" + str(step + 2) + "/test_ano2.pt")
