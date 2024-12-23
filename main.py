import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import NCSN
import data_load
import learning
import torch
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # nnがスカさないように
    torch.backends.cudnn.benchmark = True

    # GPUが使えるか確認
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # MRI画像のダウンロード
    train_dataset, test_dataset = data_load.load()

    # モデルのインスタンス化
    model = NCSN.UNet().to(device)
    print(model)

    # パラメータカウント
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()

    print(params)

    # ミニバッチの作成
    train_loader, test_loader = data_load.loader(train_dataset, test_dataset)

    # 最適化法の選択(Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 1000
    train_loss_list, test_loss_list = learning.learning(model, train_loader, test_loader, optimizer, num_epochs, device)

    plt.plot(range(len(train_loss_list)), train_loss_list, c="b", label="train loss")
    plt.plot(range(len(test_loss_list)), test_loss_list, c="r", label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("./image/loss.png")

    # モデルを保存する。
    torch.save(model.state_dict(), "model.pth")
