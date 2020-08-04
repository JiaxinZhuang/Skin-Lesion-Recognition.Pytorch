"""Trainer

    Train all your model here.
"""

import torch
import os
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score


from utils.function import init_logging, init_environment, get_lr, \
    print_loss_sometime
from utils.metric import mean_class_recall
import config
import dataset
import model
from loss import class_balanced_loss


configs = config.Config()
configs_dict = configs.get_config()
# Load hyper parameter from config file
exp = configs_dict["experiment_index"]
cuda_ids = configs_dict["cudas"]
num_workers = configs_dict["num_workers"]
seed = configs_dict["seed"]
n_epochs = configs_dict["n_epochs"]
log_dir = configs_dict["log_dir"]
model_dir = configs_dict["model_dir"]
batch_size = configs_dict["batch_size"]
learning_rate = configs_dict["learning_rate"]
backbone = configs_dict["backbone"]
eval_frequency = configs_dict["eval_frequency"]
resume = configs_dict["resume"]
optimizer = configs_dict["optimizer"]
initialization = configs_dict["initialization"]
num_classes = configs_dict["num_classes"]
iter_fold = configs_dict["iter_fold"]
loss_fn = configs_dict["loss_fn"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_environment(seed=seed, cuda_id=cuda_ids)
_print = init_logging(log_dir, exp).info
configs.print_config(_print)
tf_log = os.path.join(log_dir, exp)
writer = SummaryWriter(log_dir=tf_log)


# Pre-peocessed input image
if backbone in ["resnet50", "resnet18"]:
    re_size = 300
    input_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
elif backbone in ["NASNetALarge", "PNASNet5Large"]:
    re_size = 441
    input_size = 331
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
else:
    _print("Need backbone")
    sys.exit(-1)

_print("=> Image resize to {} and crop to {}".format(re_size, input_size))

train_transform = transforms.Compose([
    transforms.Resize(re_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
    transforms.RandomRotation([-180, 180]),
    transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                            scale=[0.7, 1.3]),
    transforms.RandomCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

input_channel = 3
trainset = dataset.Skin7(root="./data/", iter_fold=iter_fold, train=True,
                         transform=train_transform)
valset = dataset.Skin7(root="./data/", iter_fold=iter_fold, train=False,
                       transform=val_transform)

net = model.Network(backbone=backbone, num_classes=num_classes,
                    input_channel=input_channel, pretrained=initialization)

_print("=> Using device ids: {}".format(cuda_ids))
device_ids = list(range(len(cuda_ids.split(","))))
train_sampler = val_sampler = None
if len(device_ids) == 1:
    _print("Model single cuda")
    net = net.to(device)
else:
    _print("Model parallel !!")
    # torch.distributed.init_process_group(backend="nccl")
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    # net = torch.nn.parallel.DistributedDataParallel(net)
    net = nn.DataParallel(net, device_ids=device_ids).to(device)

_print("=> iter_fold is {}".format(iter_fold))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, pin_memory=True,
                                          num_workers=num_workers,
                                          sampler=train_sampler)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=False, pin_memory=True,
                                        num_workers=num_workers,
                                        sampler=val_sampler)


# Loss
if loss_fn == "WCE":
    _print("Loss function is WCE")
    weights = [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
elif loss_fn == "CE":
    _print("Loss function is CE")
    criterion = nn.CrossEntropyLoss().to(device)
else:
    _print("Need loss function.")

# Optmizer
scheduler = None
if optimizer == "SGD":
    _print("=> Using optimizer SGD with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.1, patience=50, verbose=True,
                threshold=1e-4)
elif optimizer == "Adam":
    _print("=> Using optimizer Adam with lr:{:.4f}".format(learning_rate))
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
else:
    _print("Need optimizer")
    sys.exit(-1)


start_epoch = 0
if resume:
    _print("=> Resume from model at epoch {}".format(resume))
    resume_path = os.path.join(model_dir, str(exp), str(resume))
    ckpt = torch.load(resume_path)
    net.load_state_dict(ckpt)
    start_epoch = resume + 1
else:
    _print("Train from scrach!!")


desc = "Exp-{}-Train".format(exp)
sota = {}
sota["epoch"] = start_epoch
sota["mcr"] = -1.0


for epoch in range(start_epoch+1, n_epochs+1):
    net.train()
    losses = []
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        predict = net(data)
        opt.zero_grad()
        loss = criterion(predict, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # print to log
    dicts = {
        "epoch": epoch, "n_epochs": n_epochs, "loss": loss.item()
    }
    print_loss_sometime(dicts, _print=_print)

    train_avg_loss = np.mean(losses)
    if scheduler is not None:
        scheduler.step(train_avg_loss)

    writer.add_scalar("Lr", get_lr(opt), epoch)
    writer.add_scalar("Loss/train/", train_avg_loss, epoch)

    if epoch % eval_frequency == 0:
        net.eval()
        y_true = []
        y_pred = []
        for _, (data, target) in enumerate(trainloader):
            data = data.to(device)
            predict = torch.argmax(net(data), dim=1).cpu().data.numpy()
            y_pred.extend(predict)
            target = target.cpu().data.numpy()
            y_true.extend(target)

        acc = accuracy_score(y_true, y_pred)
        mcr = mean_class_recall(y_true, y_pred)
        _print("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))
        _print("=> Epoch:{} - train mcr: {:.4f}".format(epoch, mcr))
        writer.add_scalar("Acc/train/", acc, epoch)
        writer.add_scalar("Mcr/train/", mcr, epoch)

        y_true = []
        y_pred = []
        for _, (data, target) in enumerate(valloader):
            data = data.to(device)
            predict = torch.argmax(net(data), dim=1).cpu().data.numpy()
            y_pred.extend(predict)
            target = target.cpu().data.numpy()
            y_true.extend(target)

        acc = accuracy_score(y_true, y_pred)
        mcr = mean_class_recall(y_true, y_pred)
        _print("=> Epoch:{} - val acc: {:.4f}".format(epoch, acc))
        _print("=> Epoch:{} - val mcr: {:.4f}".format(epoch, mcr))
        writer.add_scalar("Acc/val/", acc, epoch)
        writer.add_scalar("Mcr/val/", mcr, epoch)

        # Val acc
        if mcr > sota["mcr"]:
            sota["mcr"] = mcr
            sota["epoch"] = epoch
            model_path = os.path.join(model_dir, str(exp), str(epoch))
            _print("=> Save model in {}".format(model_path))
            net_state_dict = net.state_dict()
            torch.save(net_state_dict, model_path)

_print("=> Finish Training")
_print("=> Best epoch {} with {} on Val: {:.4f}".format(sota["epoch"],
                                                        "sota",
                                                        sota["mcr"]))
