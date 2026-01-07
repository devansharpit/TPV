import torch
import torch.nn as nn
import timm
# from main import get_dataloaders, evaluate, remove_batchnorm_layers

def load_model(model_name: str):
    """
    Load a pre-trained model from Hugging Face repository.
    Currently supports 'resnet18_cifar10'.
    """
    if model_name == "resnet18_cifar10":
        return load_resnet18_cifar10()
    elif model_name == "resnet50_cifar10":
        return load_resnet50_cifar10()
    elif model_name == "resnet18_cifar100":
        return load_resnet18_cifar100()
    elif model_name == "resnet56_cifar100":
        return load_resnet_cifar(model_name)
    elif model_name == "resnet56_cifar10":
        return load_resnet_cifar(model_name)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

def load_resnet_cifar(name):
    if name=='resnet56_cifar100':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
    elif name=='resnet56_cifar10':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

    model.eval()
    print("Model loaded successfully!")
    return model

def load_resnet18_cifar100():
    # 1. Create a standard ResNet-18 but with CIFAR-100 head
    model = timm.create_model("resnet18", num_classes=100, pretrained=False)

    # 2. Adapt the first layers to CIFAR-100 style (3x3 conv, no maxpool)
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    # 3. Load the HF checkpoint directly
    state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet18_cifar100/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet18_cifar100.pth",
    )

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model

def load_resnet50_cifar10():
    # 1. Create a standard ResNet-50 but with CIFAR-10 head
    model = timm.create_model("resnet50", num_classes=10, pretrained=False)

    # 2. Adapt the first layers to CIFAR-10 style (3x3 conv, no maxpool)
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    # 3. Load the HF checkpoint directly
    state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet50_cifar10/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet50_cifar10.pth",
    )

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model

def load_resnet18_cifar10():
    # 1. Create a standard ResNet-18 but with CIFAR-10 head
    model = timm.create_model("resnet18", num_classes=10, pretrained=False)

    # 2. Adapt the first layers to CIFAR-10 style (3x3 conv, no maxpool)
    model.conv1 = nn.Conv2d(
        3, 64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()

    # 3. Load the HF checkpoint directly
    state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
        map_location="cpu",
        file_name="resnet18_cifar10.pth",
    )

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model


def validate_model(model, val_loader):
    from tqdm import tqdm
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)

if __name__ == "__main__":
    # python engine/models/cifar/load_models.py 
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms

    # Test loading the model
    model = load_model("resnet56_cifar100")
    dataset = 'cifar100'
    data_root = '/home/darpit/Desktop/projects/datasets/cifar100/'
    
    batch_size = 256
    num_workers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    batch_size = 128
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    train_transform = transforms.Compose(
        [transforms.ToTensor(),  transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    
    if dataset=='cifar10':
        train_data = datasets.CIFAR10(data_root, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_root, train=False, transform=test_transform, download=True)        
    elif dataset=='cifar100':
        train_data = datasets.CIFAR100(data_root, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_root, train=False, transform=test_transform, download=True)         
    else:
        raise NotImplementedError

    # if args.label_noise>0:
    #     corrupt_labels(train_data, p=args.label_noise, seed=0)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=8, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=8, persistent_workers=True)    
    
    
    model = model.to(device)
    acc,_ = validate_model(model, val_loader)
    print(f"Test Accuracy: {100. * acc:.2f}%")
    