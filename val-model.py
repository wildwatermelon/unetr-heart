import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch

train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(48, 48, 48),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

root_data_dir = r'D:\Capstone\dataset'
# root_data_dir = r'/workspace/unetr-heart/datasets'
data_dir = "/dataset-wholeheart/"
split_JSON = "dataset_0.json"
datasets = root_data_dir + data_dir + split_JSON

datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")

train_ds = CacheDataset(
    data=datalist,transform=train_transforms,cache_num=24,cache_rate=1.0,num_workers=8
)
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1,
    out_channels=8,
    img_size=(48, 48, 48),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

slice_map = {
    # val
    "ct_train_1017_image.nii.gz": 50,
    "ct_train_1018_image.nii.gz": 50,
    "ct_train_1019_image.nii.gz": 50,
    "ct_train_1020_image.nii.gz": 50,

    # train
    # "ct_train_1001_image.nii.gz": 50,
    # "ct_train_1002_image.nii.gz": 50,
    # "ct_train_1003_image.nii.gz": 50,
    # "ct_train_1004_image.nii.gz": 50,
}

case_num = 2
# model.load_state_dict(torch.load("best_metric_model.pth"))
model.load_state_dict(torch.load("best_metric_model.pth", map_location=torch.device('cpu')),strict=False)
model.eval()
with torch.no_grad():
    img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]

    # transform batch label to whs labels
    label = torch.where(label == 500.0, torch.tensor(1.0), label)
    label = torch.where(label == 600.0, torch.tensor(2.0), label)
    label = torch.where(label == 420.0, torch.tensor(3.0), label)
    label = torch.where(label == 550.0, torch.tensor(4.0), label)
    label = torch.where(label == 205.0, torch.tensor(5.0), label)
    label = torch.where(label == 820.0, torch.tensor(6.0), label)
    label = torch.where(label == 850.0, torch.tensor(7.0), label)

    # val_inputs = torch.unsqueeze(img, 1).cuda()
    # val_labels = torch.unsqueeze(label, 1).cuda()
    val_inputs = torch.unsqueeze(img, 1)
    val_labels = torch.unsqueeze(label, 1)

    val_outputs = sliding_window_inference(
        val_inputs, (48, 48, 48), 4, model, overlap=0.8
    )
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
    plt.subplot(1, 3, 3)
    plt.title("output")
    plt.imshow(
        torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]]
    )
    plt.savefig('temp-model-validation.png')
    plt.show()