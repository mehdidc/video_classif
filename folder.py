import torch.utils.data as data
from PIL import Image
from torchvision.datasets import folder

import os
import os.path

class ImageFolder(folder.ImageFolder):
    # The same ImageFolder of torchvision but works correctly for the structure
    # of folders used in videos
    # The only thing we change is the function `make_dataset`, the rest is exactly
    # the same
    def __init__(self, root, transform=None, target_transform=None, loader=folder.default_loader): 
        
        classes, class_to_idx = folder.find_classes(root)
        idx_to_class = {i: cl for cl, i in class_to_idx.items()}
        extensions = folder.IMG_EXTENSIONS
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

        self.imgs = self.samples
        
        self.num_inputs = 1
        self.num_targets = 1
def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
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
                        images.append(item)

    return images


