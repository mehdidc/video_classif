from clize import run
import random
import os
import glob
from subprocess import call

import numpy as np

import torch.nn as nn
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from folder import ImageFolder

import visdom

cudnn.benchmark = True

def compute_accuracy(ypred, y):
    _, ypred_argmax = ypred.max(1)
    return (ypred_argmax == y).float().mean()

def generate_frames(folder, *, pattern='*.mp4'):
    full_folder = os.path.join(folder, 'videos', 'full')
    for filename in glob.glob(os.path.join(full_folder, '**', pattern)):
        dest = filename.replace('videos', 'frames')
        os.makedirs(dest, exist_ok=True)
        cmd = 'ffmpeg -i {} {}/image_%05d.jpg'.format(filename, dest)
        call(cmd, shell=True)
 

def generate_splits(folder, *, pattern='*.mp4', ratio_train=0.7, ratio_valid=0.1, seed=42):
    random.seed(seed)
    os.makedirs(os.path.join(folder, 'frames', 'train'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'frames', 'valid'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'frames', 'test'), exist_ok=True)
    full_folder = os.path.join(folder, 'frames', 'full')
    for filename in glob.glob(os.path.join(full_folder, '**', pattern)):
        u = random.uniform(0, 1)
        if u <= ratio_train:
            sp = 'train'
        elif u <= ratio_train + ratio_valid:
            sp = 'valid'
        else:
            sp = 'test'
        dest = filename.replace('full', sp)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.symlink(os.path.abspath(filename), os.path.abspath(dest))


def train_2d(*, folder='data'):
    
    batch_size = 64
    test_batch_size = 32
    best_acc = 0.


    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
 
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(
        os.path.join(folder, 'frames', 'train'),
        transform=train_transform
    )
    valid_dataset = ImageFolder(
        os.path.join(folder, 'frames', 'valid'),
        transform=valid_transform
    )
    test_dataset = ImageFolder(
        os.path.join(folder, 'frames', 'test'),
        transform=valid_transform
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size)
    
    num_classes = len(train_dataset.classes)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, num_classes),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    model = model.cuda()
    crit = crit.cuda()

    num_epoch = 20
    avg_loss = 0.
    avg_acc = 0.
    gamma = 0.9
    nb_updates = 0
    
    viz = visdom.Visdom()
    loss_window = viz.line(X=np.array([0]), Y=np.array([0]), opts={'title': 'loss'})
    acc_window = viz.line(X=np.array([0]), Y=np.array([0]), opts={'title': 'acc'})

    valid_loss_window = viz.line(X=np.array([0]), Y=np.array([0]), opts={'title': 'valid_loss'})
    valid_acc_window = viz.line(X=np.array([0]), Y=np.array([0]), opts={'title': 'valid_acc'})
    
    model.transform = valid_transform
    for epoch in range(num_epoch):
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X = X.cuda()
            y = y.cuda()
            X = Variable(X)
            y = Variable(y)
            model.zero_grad()
            ypred = model(X)
            loss = crit(ypred, y)
            loss.backward()
            optimizer.step()
            acc = compute_accuracy(ypred, y)
            avg_loss = gamma * avg_loss + (1 - gamma) * loss.data[0]
            avg_acc = gamma * avg_acc + (1 - gamma) * acc.data[0]
            print('Batch {:05d}/{:05d} AvgTrainLoss : {:.4f} AvgTrainAcc : {:.4f}'.format(batch, len(train_loader), avg_loss, avg_acc))
            viz.line(X=np.array([nb_updates]), Y=np.array([loss.data[0]]), win=loss_window, update='append')
            viz.line(X=np.array([nb_updates]), Y=np.array([acc.data[0]]), win=acc_window, update='append')
            nb_updates += 1
        model.eval()
        valid_loss, valid_acc = validate(model, valid_loader)       
        if valid_acc > best_acc:
            print('New best valid acc : {:.4f}'.format(valid_acc))
            best_acc = valid_acc
            torch.save(model, 'model.th')

        print('Epoch {:03d}/{:03d} AvgTrainLoss : {:.4f} AvgTrainAcc : {:.4f} AvgValidLoss : {:.4f} AvgValidAcc : {:.4f}'.format(
            epoch + 1, num_epoch, avg_loss, avg_acc, valid_loss, valid_acc))
        viz.line(X=np.array([epoch]), Y=np.array([valid_loss]), win=valid_loss_window, update='append')
        viz.line(X=np.array([epoch]), Y=np.array([valid_acc]), win=valid_acc_window, update='append')
 

def validate(model, loader):
    losses = []
    accs = []
    crit = nn.CrossEntropyLoss()
    for X, y in loader:
        X = X.cuda()
        y = y.cuda()
        X = Variable(X)
        y = Variable(y)
        model.zero_grad()
        ypred = model(X)
        loss = crit(ypred, y)
        acc = compute_accuracy(ypred, y)
        losses.append(loss.data[0])
        accs.append(acc.data[0])
    return np.mean(losses), np.mean(accs)


if __name__ == '__main__':
    run([generate_frames, generate_splits, train_2d])
