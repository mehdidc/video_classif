import shutil
from skimage.io import imread, imsave
import cv2
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


def generate_frames(folder='data', pattern='*.mp4'):
    # generate frames from videos in `folder`
    # the structure of `folder` should be: folder/videos/full/<class_name>/<video_name.mp4>
    full_folder = os.path.join(folder, 'videos', 'full')
    for filename in glob.glob(os.path.join(full_folder, '**', pattern)):
        dest = filename.replace('videos', 'frames')
        os.makedirs(dest, exist_ok=True)
        _generate_frames_video(filename, dest)


def _generate_video_from_frames(pattern, dest):
    # creates a video from a set of images obeying the `pattern`
    # and name the video `dest`
    cmd = "ffmpeg -i {} -vf fps=25 {}".format(pattern, dest)
    call(cmd, shell=True)


def _generate_frames_video(filename, dest):
    # generate frames of a video
    cmd = 'ffmpeg -i {} {}/image_%05d.jpg'.format(filename, dest)
    call(cmd, shell=True)



def generate_splits(folder='data', pattern='*.mp4', ratio_train=0.7, ratio_valid=0.1, seed=42):
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


def train_2d(*, folder='data', resume=False, lr=0.001, use_visdom=False): 
    batch_size = 64
    test_batch_size = 32
    num_epoch = 20

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
    print('Number of training examples : {}'.format(len(train_dataset)))
    print('Number of valid examples : {}'.format(len(valid_dataset)))
    print('Number of test examples : {}'.format(len(test_dataset)))

    num_classes = len(train_dataset.classes)
    print('Number of classes : {}'.format(num_classes))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size)
    
    if resume:
        model = torch.load('model.th')
    else:
        model = models.resnet50(pretrained=True)
        model.idx_to_class = train_dataset.idx_to_class
        model.transform = valid_transform
        model.fc = nn.Sequential(
            nn.Linear(2048, num_classes),
        )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    
    model = model.cuda()
    crit = crit.cuda()
    if use_visdom:
        viz = visdom.Visdom()
        loss_window = viz.line(X=np.array([0]), Y=np.array([0]), opts={'title': 'loss'})
        acc_window = viz.line(X=np.array([0]), Y=np.array([0]), opts={'title': 'acc'})

        valid_loss_window = viz.line(X=np.array([0]), Y=np.array([0]), opts={'title': 'valid_loss'})
        valid_acc_window = viz.line(X=np.array([0]), Y=np.array([0]), opts={'title': 'valid_acc'})

    avg_loss = 0.
    avg_acc = 0.
    gamma = 0.9
    nb_updates = 0
    best_acc = 0.

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
            print('Epoch {:03d}/{:03d} Batch {:05d}/{:05d} AvgTrainLoss : {:.4f} AvgTrainAcc : {:.4f}'.format(epoch + 1, num_epoch, batch, len(train_loader), avg_loss, avg_acc))
            if use_visdom:
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
        if use_visdom:
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

def annotate_full(video_path, model='model.th'):
    # annotate the full video with a single label
    model = torch.load(model)
    idx_to_class = model.idx_to_class
    #classes = sorted(os.listdir('data/videos/full'))
    #idx_to_class = {i: cl for i, cl in enumerate(classes)}
    
    try:
        shutil.rmtree('.cache')
    except FileNotFoundError:
        pass
    os.makedirs('.cache/frames/imgs')
    _generate_frames_video(video_path, '.cache/frames/imgs')
    dataset = ImageFolder(
        '.cache/',
        transform=model.transform,
    )
    loader = DataLoader(dataset, batch_size=32)
    ypred_list = []
    for X, _ in loader:
        X = Variable(X)
        X = X.cuda()
        ypred = model(X)
        ypred_list.append(ypred.cpu().data)
    ypred = torch.cat(ypred_list, 0)
    ypred = ypred.mean(0)
    _, yind = ypred.max(0)
    yind = int(yind)
    print(idx_to_class[yind])

def annotate_frames(video_path, model='model.th', out='video.mp4'):
    # annotate all frames with text and generate a video containing
    # the annotations
    
    model = torch.load(model)
    
    idx_to_class = model.idx_to_class
    #classes = sorted(os.listdir('data/videos/full'))
    #idx_to_class = {i: cl for i, cl in enumerate(classes)}
    model.eval()
    try:
        shutil.rmtree('.cache')
    except FileNotFoundError:
        pass
    folder = '.cache/frames/imgs'
    os.makedirs(folder)
    print('Generating frames...')
    _generate_frames_video(video_path, folder)
    dataset = ImageFolder(
        '.cache/',
        transform=model.transform,
    )
    loader = DataLoader(dataset, batch_size=32)
    ypred_list = []
    print('Predicting frames...')
    for i, (X, _) in enumerate(loader):
        print('Batch {}/{}'.format(i + 1, len(loader)))
        X = Variable(X)
        X = X.cuda()
        ypred = model(X)
        ypred = nn.Softmax(dim=1)(ypred)
        ypred_list.append(ypred.cpu().data)
    ypred = torch.cat(ypred_list, 0)
    pred_bs = 1 
    for i in range(0, len(ypred), pred_bs):
        yb = ypred[i:i+pred_bs]
        m = yb.mean(0, keepdim=True)
        print(m)
        ypred[i:i+pred_bs] = m.repeat(yb.size(0), 1)
    _, yind = ypred.max(1)
    print('Putting text into images...')
    for i, name in enumerate(sorted(os.listdir(folder))):
        path = os.path.join(folder, name)
        name = idx_to_class[int(yind[i])]
        im = imread(path)
        text = name
        text_x, text_y = 50, 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_color = (255, 255, 255)
        im = cv2.putText(im, text, (text_x, text_y), font, font_scale, text_color, 2, cv2.LINE_AA)
        imsave(path, im)
    print('Generating the resulting video...')
    _generate_video_from_frames(os.path.join(folder, 'image_%05d.jpg'), out)


if __name__ == '__main__':
    run([generate_frames, generate_splits, train_2d, annotate_full, annotate_frames])
