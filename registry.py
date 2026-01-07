from pyexpat import model
from torchvision import datasets, transforms as T
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
import os, sys
import engine.models as models
import engine.utils as utils
from functools import partial
NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'cifar10_224':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100_224': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
}


MODEL_DICT = {
    'resnet18_cifar10': models.cifar.resnet.resnet18,
    'resnet18_cifar100': models.cifar.resnet.resnet18_cifar100,
    'resnet56_cifar100': models.cifar.resnet.resnet56_cifar100,
    'resnet56_cifar10': models.cifar.resnet.resnet56_cifar10,
    'vgg19_bn_cifar100': models.cifar.vgg.vgg19_bn_cifar100,

}

IMAGENET_MODEL_DICT={
    "resnet50_imagenet": models.imagenet.resnet50, 
    "mobilenet_v2_imagenet": models.imagenet.mobilenet_v2,
}

def get_model(name: str, num_classes, pretrained=False, target_dataset='cifar', **kwargs):
    if target_dataset == "imagenet":
        model = IMAGENET_MODEL_DICT[name](pretrained=True)
    elif 'cifar' in target_dataset:
        model = MODEL_DICT[name](num_classes=num_classes)
    return model 


def get_dataset(name: str, data_root: str='data', return_transform=False):
    name = name.lower()
    data_root = os.path.expanduser( data_root )

    if name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        # data_root = os.path.join( data_root, 'torchdata' )
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
        input_size = (1, 3, 32, 32)
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        # data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
        input_size = (1, 3, 32, 32)
    elif name=='modelnet40':
        num_classes=40
        train_dst = utils.datasets.ModelNet40(data_root=data_root, partition='train', num_points=1024)
        val_dst = utils.datasets.ModelNet40(data_root=data_root, partition='test', num_points=1024)
        train_transform = val_transform = None
        input_size = (1, 3, 2048)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, input_size, train_transform, val_transform
    return num_classes, train_dst, val_dst, input_size

