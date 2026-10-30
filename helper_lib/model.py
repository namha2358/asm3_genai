# helper_lib/model.py — GAN architecture from the assignment spec, plus get_model
import torch.nn as nn

Z_DIM = 100                  # length of the noise vector fed to the generator


class Generator(nn.Module):
    """Generator matching the assignment spec: noise (B, 100) -> image (B, 1, 28, 28)."""

    def __init__(self, z_dim=Z_DIM):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        # Fully connected layer: 100 -> 7*7*128 = 6272 values per sample
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)

        # ConvTranspose2D 128 -> 64, kernel 4, stride 2, padding 1: upsamples 7x7 -> 14x14
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)        # normalizes each of the 64 channels across the batch
        self.act1 = nn.ReLU(True)            # ReLU; True = modify in place to save memory

        # ConvTranspose2D 64 -> 1, kernel 4, stride 2, padding 1: upsamples 14x14 -> 28x28
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()                # squashes pixels into [-1, 1] to match normalized MNIST

    def forward(self, x):
        x = self.fc(x)                       # (B, 100) -> (B, 6272)
        x = x.view(x.size(0), 128, 7, 7)     # reshape -> (B, 128, 7, 7)

        x = self.deconv1(x)                  # -> (B, 64, 14, 14)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.deconv2(x)                  # -> (B, 1, 28, 28)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    """Discriminator matching the assignment spec: image (B, 1, 28, 28) -> one score per image."""

    def __init__(self):
        super(Discriminator, self).__init__()

        # Conv2D 1 -> 64, kernel 4, stride 2, padding 1: downsamples 28x28 -> 14x14
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)   # LeakyReLU with slope 0.2 for negatives

        # Conv2D 64 -> 128, kernel 4, stride 2, padding 1: downsamples 14x14 -> 7x7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Flatten 128*7*7 = 6272 values, then one Linear output (the real/fake logit)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv1(x)                    # (B, 1, 28, 28) -> (B, 64, 14, 14)
        x = self.act1(x)

        x = self.conv2(x)                    # -> (B, 128, 7, 7)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.flatten(x)                  # -> (B, 6272)
        x = self.fc(x)                       # -> (B, 1); sigmoid(x) gives the real/fake probability
        return x


class GAN(nn.Module):
    """Wrapper holding both networks so the helper functions can take a single model argument."""

    def __init__(self, z_dim=Z_DIM):
        super(GAN, self).__init__()
        self.z_dim = z_dim
        self.generator = Generator(z_dim)    # maps noise -> fake image
        self.discriminator = Discriminator() # scores images as real (high) or fake (low)

    def forward(self, z):
        return self.generator(z)             # calling the GAN on noise generates images


def get_model(model_name):
    """Return the model for model_name; only "GAN" is implemented in this project."""
    if model_name == "GAN":
        model = GAN()
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Supported: GAN")
    return model