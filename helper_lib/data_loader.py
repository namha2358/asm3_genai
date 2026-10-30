# helper_lib/data_loader.py — builds the MNIST data loader used to train the GAN
import torch
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True):
    """Return a DataLoader over MNIST with pixels scaled to [-1, 1] to match the generator's Tanh output."""
    transform = transforms.Compose([
        transforms.ToTensor(),                    # PIL image -> tensor of shape (1, 28, 28), values in [0, 1]
        transforms.Normalize((0.5,), (0.5,)),     # (x - 0.5) / 0.5 -> values in [-1, 1]
    ])

    # Download MNIST into data_dir on first use; train=True gives the 60,000-image training split
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)

    # Shuffle only the training split, so each epoch sees the images in a new order
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader