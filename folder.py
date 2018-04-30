import torch.utils.data as data
from PIL import Image
from torchvision.datasets import folder
from collections import defaultdict
import numpy as np
import os
import os.path

class BalancedSample:
    
    def __init__(self, dataset, seed=None):
        self.rng = np.random.RandomState(seed)
        self.dataset = dataset
        self.class_indices = dataset.class_indices
        self.classes = dataset.classes
        self.idx_to_class = dataset.idx_to_class
        self.class_freq = dataset.class_freq

    def __getitem__(self, idx):
        cl = self.rng.randint(0, len(self.class_indices) - 1)
        idx = self.rng.choice(self.class_indices[cl])
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


class ImageFolder(folder.ImageFolder):
    # The same ImageFolder of torchvision but works correctly for the structure
    # of folders used in videos
    # The only thing we change is the function `make_dataset`, the rest is exactly
    # the same
    def __init__(self, root, transform=None, target_transform=None, loader=folder.default_loader): 
        
        classes, class_to_idx = folder.find_classes(root)
        idx_to_class = {i: cl for cl, i in class_to_idx.items()}
        extensions = folder.IMG_EXTENSIONS
        samples, class_indices, class_freq = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_freq = class_freq
        self.idx_to_class = idx_to_class
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples
        self.class_indices = class_indices
        self.num_inputs = 1
        self.num_targets = 1


def make_dataset(dir, class_to_idx, extensions):
    images = []
    class_indices = defaultdict(list)
    dir = os.path.expanduser(dir)
    class_freq = defaultdict(int)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, dirnames, fnames in sorted(os.walk(d)):
            for dname in sorted(dirnames):
                dpath = os.path.join(root, dname)
                for fname in os.listdir(dpath):
                    fpath = os.path.join(dpath, fname)
                    if folder.is_image_file(fpath):
                        item = (fpath, class_to_idx[target])
                        class_freq[class_to_idx[target]] += 1
                        class_indices[class_to_idx[target]].append(len(images))
                        images.append(item)
    return images, class_indices, class_freq


