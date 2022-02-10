import torch,monai

output = torch.randn(4, 14, 48, 48, 48)
# randint is out of bounds (encoding?)
target = torch.randint(0, 15, (4, 1, 48, 48, 48))

criterion = monai.losses.DiceLoss(softmax=True, to_onehot_y=True)
loss = criterion(output, target)