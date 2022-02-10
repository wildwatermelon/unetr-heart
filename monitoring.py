import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNETR
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file = open("epoch_loss_values.txt", "r")
epoch_loss_values = list(map(float, file.read()[1:-1].split()))
file = open("metric_values.txt", "r")
metric_values = list(map(float, file.read()[1:-1].split()))
eval_num = 100

model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(48,48,48),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

# model.load_state_dict(torch.load("best_metric_model.pth"))
model.load_state_dict(torch.load("best_metric_model.pth", map_location=torch.device('cpu')),strict=False)
model.eval()

#monitoring
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
# plt.savefig('temp-model-monitoring.png')
plt.show()
