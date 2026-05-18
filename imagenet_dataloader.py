
def create_imagenet_dataloaders(train_batch_size=512, val_batch_size=512, use_augmentation=True):
    # use data augmentation on training data when use_augmentation = True; do not augment validation data 
    # should return train_loader, val_loader with given batch sizes
    raise NotImplementedError("Please implement data loading for ImageNet dataset.")