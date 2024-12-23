import torch
import matplotlib.pyplot as plt

ano_image = torch.load("./image/SSM/exp1_knee/ano_image.pt")

ano1 = torch.load("./image/SSM/exp1_knee/step1/test_ano_ini_mean.pt")
ano2 = torch.load("./image/SSM/exp1_knee/step4/test_ano.pt")
ano3 = torch.load("./image/SSM/exp1_knee/step5/test_ano.pt")
ano4 = torch.load("./image/SSM/exp1_knee/step6/test_ano.pt")
ano5 = torch.load("./image/SSM/exp1_knee/step7/test_ano.pt")
ano6 = torch.load("./image/SSM/exp1_knee/step8/test_ano.pt")

dim = 320

plt.figure()
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
plt.imshow(ano1.cpu().reshape(dim, dim), cmap="jet", alpha=0.6)
plt.savefig("./image/ano1.png", bbox_inches="tight")

plt.figure()
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
plt.imshow(ano2.cpu().reshape(dim, dim), cmap="jet", alpha=0.6)
plt.savefig("./image/ano2.png", bbox_inches="tight")

plt.figure()
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
plt.imshow(ano3.cpu().reshape(dim, dim), cmap="jet", alpha=0.6)
plt.savefig("./image/ano3.png", bbox_inches="tight")

plt.figure()
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
plt.imshow(ano4.cpu().reshape(dim, dim), cmap="jet", alpha=0.6)
plt.savefig("./image/ano4.png", bbox_inches="tight")

plt.figure()
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
plt.imshow(ano5.cpu().reshape(dim, dim), cmap="jet", alpha=0.6)
plt.savefig("./image/ano5.png", bbox_inches="tight")

plt.figure()
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
plt.imshow(ano_image.cpu().reshape(dim, dim), cmap="gray")
plt.imshow(ano6.cpu().reshape(dim, dim), cmap="jet", alpha=0.6)
plt.savefig("./image/ano6.png", bbox_inches="tight")
