import warnings

# +
warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import dataloader
from utils import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils.logger import MyWriter
import torch
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from Model import CNN
# from Dataset import CatsAndDogsDataset
from tqdm import tqdm
import pandas as pd
import numpy as np
from data_generator import Dataset_seq, ImageDataset, ToTensorTarget, NormalizeTarget, UnNormalize


# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
# %matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

import os
import numpy as np
import cv2
from glob import glob

import warnings

warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import dataloader
from utils import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils.logger import MyWriter
import torch
import argparse
import os
device = ("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# +
def main(hp, num_epochs, resume, name):

    checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = MyWriter("{}/{}".format(hp.log, name))
    # get model

#     if hp.RESNET_PLUS_PLUS:
#         model = ResUnetPlusPlus(3).cuda()
#     else:
#         model = ResUnet(3, 64).cuda()
    model = ResUnetPlusPlus().cuda()

    # set up binary cross entropy and dice loss
#     criterion = metrics.BCEDiceLoss()
    criterion = metrics.DiceLoss()
#     criterion = nn.BCELoss()
    
        ## Parameters
    image_size = 512
    batch_size = 8
    lr = 1e-3
    num_epochs = 1000
    learning_rate = 0.001
    train_CNN = True
#     batch_size = 32
    shuffle = True
    pin_memory = True
    num_workers = 2

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.05, nesterov=True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#     optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)

    # decay LR
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # starting params
    best_loss = 999
    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            start_epoch = checkpoint["epoch"]

            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            
    ## Path
    file_path = "files/"
    model_path = "files/resunetplusplus.h5"

    ## Create files folder
    try:
        os.mkdir("files")
    except:
        pass

    train_path_input = "/workspace/dals_tf2/dataset/patch_subset/Train/*_input.npy"
    train_path_mask = "/workspace/dals_tf2/dataset/patch_subset/Train/*_mask.npy"
    valid_path_input = "/workspace/dals_tf2/dataset/patch_subset/Valid/*_input.npy"
    valid_path_mask = "/workspace/dals_tf2/dataset/patch_subset/Valid/*_mask.npy"
    # valid_path = "/workspace/dals_tf2/dataset/INbreast_mass_data/Valid/"

    ## Training
    train_image_paths = glob(train_path_input)
#     print(train_image_paths)
    train_mask_paths = glob(train_path_mask)
    train_image_paths.sort()
    train_mask_paths.sort()
#     print(len(train_mask_paths), "train")

    # train_image_paths = train_image_paths[:2000]
    # train_mask_paths = train_mask_paths[:2000]

    ## Validation
    valid_image_paths = glob(valid_path_input)
    valid_mask_paths = glob(valid_path_mask)
    valid_image_paths.sort()
    valid_mask_paths.sort()
    print(len(valid_mask_paths), "valid")


#     mass_dataset_train = Dataset_seq(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
#     mass_dataset_val = Dataset_seq(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

#     mass_dataset_train = DataLoader(dataset=valid_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
#     mass_dataset_val = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
     # get data
    mass_dataset_train = ImageDataset(
        image_list=train_image_paths, mask_list=train_mask_paths, train=True, transform=None, image_size=image_size)


    mass_dataset_val = ImageDataset(
        image_list=valid_image_paths, mask_list=valid_mask_paths, train=False, transform=None, image_size=image_size)

    # creating loaders
    train_dataloader = DataLoader(
        mass_dataset_train, batch_size=batch_size, num_workers=2, shuffle=True
    )
    val_dataloader = DataLoader(
        mass_dataset_val, batch_size=batch_size, num_workers=2, shuffle=False
    )
    step = 0
    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        # logging accuracy and loss
        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()
        # iterate over data

        loader = tqdm(train_dataloader, desc="training")
#         print(enumerate(loader)[0])
        YU = 0
        for idx, data in enumerate(loader):
           
            print(idx, YU)
            YU = YU + 1
            # get the inputs and wrap in Variable
            inputs = data["image"].cuda()
            labels = data["mask"].cuda()
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
#             prob_map, s_pred = model(inputs) # last activation was a sigmoid
#             outputs = (prob_map > 0.8).float()
#             outputs = model(inputs)
#             outputs = torch.nn.functional.sigmoid(outputs)
            outputs, s_pred, c1, c2, c3, c4, d1, d2, d3, _, s2, s3, s4 = model(inputs)
#             outputs = outputs > t
#             outputs = outputs > 0.7
            for g in range(data["image"].numpy().shape[0]):
                fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                axs[0].imshow(data["image"].numpy()[g, 0, :, :])
    #             plt.show()
                axs[1].imshow(data["mask"].numpy()[g, 0, :, :])
    #             plt.show()
#                 axs[2].imshow(outputs.cpu().detach().numpy()[g, 0, :, :])
                axs[2].imshow((outputs.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                plt.show()
        
            for g in range(data["image"].numpy().shape[0]):
                fig, axs = plt.subplots(1, 13, figsize=(9, 3))
                axs[0].imshow(data["image"].numpy()[g, 0, :, :])
                axs[0].set_title('image')
                axs[1].imshow((c1.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[1].set_title('c1')
                axs[2].imshow((c2.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[2].set_title('c2')
                axs[3].imshow((c3.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[3].set_title('c3')
                axs[4].imshow((c4.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[4].set_title('c4')
                axs[5].imshow((d1.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[5].set_title('d1')
                axs[6].imshow((d2.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[6].set_title('d2')
#                 axs[2].imshow(outputs.cpu().detach().numpy()[g, 0, :, :])
                axs[7].imshow((d3.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[7].set_title('d3')
                axs[8].imshow((s2.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[8].set_title('s2')
                axs[9].imshow((s3.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[9].set_title('s3')
                axs[10].imshow((s4.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[10].set_title('s4')
                axs[11].imshow((outputs.cpu().detach().numpy() > 0.7)[g, 0, :, :])
                axs[11].set_title('outputs')
                axs[12].imshow(data["mask"].numpy()[g, 0, :, :])
                axs[12].set_title('masks')
                plt.show()

            loss = criterion(outputs, labels, s_pred)
#             loss = criterion(outputs, labels)
#             loss = dice(preds, target, average='micro')

            # backward
            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.data.item(), outputs.size(0))

            # tensorboard logging
#             if step % hp.logging_step == 0:
            writer.log_training(train_loss.avg, train_acc.avg, step)
            loader.set_description(
                "Training Loss: {:.4f} Acc: {:.4f}".format(
                    train_loss.avg, train_acc.avg
                )
            )

            # Validation
#             if step % hp.validation_interval == 0:
            valid_metrics = validation(
                val_dataloader, model, criterion, writer, step
            )
            save_path = os.path.join(
                checkpoint_dir, "%s_checkpoint_%04d.pt" % (name, step)
            )
            # store best loss and save a model checkpoint
            best_loss = min(valid_metrics["valid_loss"], best_loss)
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "arch": "ResUnet",
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            print("Saved checkpoint to: %s" % save_path)

            step += 1


def validation(valid_loader, model, criterion, logger, step):

    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    YU = 0
    for idx, data in enumerate(tqdm(valid_loader, desc="validation")):

        # get the inputs and wrap in Variable
        inputs = data["image"].cuda()
        labels = data["mask"].cuda()
        print(idx, YU)
        YU = YU + 1

        # forward
#         prob_map, s_pred = model(inputs) # last activation was a sigmoid
#         outputs = (prob_map > 0.8).float()
#         outputs = model(inputs)
#         outputs = torch.nn.functional.sigmoid(outputs)
        outputs, s_pred, c1, c2, c3, c4, d1, d2, d3, _, s1, s2, s3 = model(inputs)
#         outputs = outputs > 0.7
#         print(outputs.shape)
        for g in range(data["image"].numpy().shape[0]):
            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            axs[0].imshow(data["image"].numpy()[g, 0, :, :])
#             plt.show()
            axs[1].imshow(data["mask"].numpy()[g, 0, :, :])
#             plt.show()
#             axs[2].imshow(outputs.cpu().detach().numpy()[g, 0, :, :])
            axs[2].imshow((outputs.cpu().detach().numpy() > 0.7)[g, 0, :, :])
            plt.show()

        loss = criterion(outputs, labels, s_pred)
#         loss = criterion(outputs, labels)
        print(metrics.iou_pytorch(outputs, labels))

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data.item(), outputs.size(0))
#         if idx == 0:
#         logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)
    logger.log_validation(valid_loss.avg, valid_acc.avg, step)

    print("Validation Loss: {:.4f} Acc: {:.4f}".format(valid_loss.avg, valid_acc.avg))
    model.train()
    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}

config = "configs/default.yaml"
name = "default"
epochs = 1000
resume=""
hp = HParam(config)
with open(config, "r") as f:
    hp_str = "".join(f.readlines())

main(hp, num_epochs=epochs, resume=resume, name=name)

# -

# !pip install tensorboardX
# !pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# !pip install torchmetrics
# !pip install kornia




