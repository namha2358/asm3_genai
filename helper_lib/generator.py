# helper_lib/generator.py — sample new images from a trained GAN and plot them on a grid
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device='cpu', num_samples=10, show=True):
    """Generate num_samples images from random noise; plot the grid if show=True and return it."""
    gen = model.generator if hasattr(model, "generator") else model   # accept the GAN wrapper or a bare Generator
    gen.to(device)
    gen.eval()                                   # eval mode: BatchNorm uses its learned running statistics

    with torch.no_grad():                        # no gradients needed for sampling
        noise = torch.randn(num_samples, gen.z_dim, device=device)   # points in the latent space
        images = gen(noise).cpu()                # (num_samples, 1, 28, 28), values in [-1, 1]

    # Arrange the images in one grid; normalize maps [-1, 1] back to [0, 1] for display
    grid = make_grid(images, nrow=min(num_samples, 10), normalize=True, value_range=(-1, 1))

    if show:
        plt.imshow(grid.permute(1, 2, 0))        # imshow expects height x width x channels
        plt.axis("off")
        plt.title(f"{num_samples} generated digits")
        plt.show()
    return grid