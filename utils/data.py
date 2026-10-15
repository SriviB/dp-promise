"""
Utility functions to load and transform datasets
"""
import torch
import torchvision.transforms as transforms

from src.utils import load_dataset, load_dataset_from_config


def load_data(config, dataset_size=0, split='train'):
    # DP-PROMISE transform: images scaled to [-1, 1] (train.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x * 2. - 1.,
    ])

    if split == 'train':
        dataset = load_dataset_from_config(config, transform, train=True)
    else:
        # load_dataset_from_config ignores its train argument, so call load_dataset directly
        dataset = load_dataset(config.data.name, transform=transform, train=False)

    out_dim = config.data.num_classes

    # load neighboring dataset D-
    dataset_size = len(dataset) if dataset_size is None or dataset_size <= 0 else dataset_size # load full dataset if dataset_size is None or 0
    shuffle = dataset_size != len(dataset) # only shuffle dataset if full dataset is not loaded
    tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset_size, shuffle=shuffle)

    X, y = next(iter(tmp_loader))

    return X, y, out_dim
