import torch
import function
from tqdm import tqdm
import matplotlib.pyplot as plt


# 1epoch分の訓練を行う関数
def train_model(model, train_loader, optimizer, device):
    train_loss = 0.0

    # 学習モデルに変換
    model.train()

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        # images = images.reshape(images.size()[0], 1, images.size()[1], images.size()[2])

        inputs, sigma = function.addNoise(images, device)

        # 勾配を初期化
        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # 推論(順伝播)
            outputs = model(inputs, sigma)

            # 損失の算出
            loss = function.criterion(images, inputs, outputs, sigma)

        # 誤差逆伝播
        loss.backward()

        print(loss.item())

        # 勾配をClip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # パラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()

    # lossの平均値を取る
    train_loss = train_loss / len(train_loader)

    return train_loss


# モデル評価を行う関数
def test_model(model, test_loader, optimizer, device):
    test_loss = 0.0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad():  # 勾配計算の無効化
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images, labels = images.to(device), labels.to(device)
            # images = images.reshape(images.size()[0], 1, images.size()[1], images.size()[2])
            inputs, sigma = function.addNoise(images, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(inputs, sigma)
                loss = function.criterion(images, inputs, outputs, sigma)
                test_loss += loss.item()

    # lossの平均値を取る
    test_loss = test_loss / len(test_loader)

    return test_loss


def learning(model, train_loader, test_loader, optimizer, num_epochs, device):
    train_loss_list = []
    test_loss_list = []

    # epoch数分繰り返す
    for epoch in range(1, num_epochs + 1, 1):
        train_loss = train_model(model, train_loader, optimizer, device)
        test_loss = test_model(model, test_loader, optimizer, device)

        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}".format(epoch, train_loss, test_loss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        if epoch % 10 == 0:
            filename = "./models/model3_knee/model" + str(epoch) + ".pth"
            torch.save(model.state_dict(), filename)
            plt.plot(range(len(train_loss_list)), train_loss_list, c="b", label="train loss")
            plt.plot(range(len(test_loss_list)), test_loss_list, c="r", label="test loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig("./image/loss.png")

    return train_loss_list, test_loss_list
