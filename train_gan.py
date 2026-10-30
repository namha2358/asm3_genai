# train_gan.py — trains the assignment GAN on MNIST through helper_lib and saves the generator weights
import torch
import torch.nn as nn
import torch.optim as optim
from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples

EPOCHS = 10                                  # full passes over the 60,000 MNIST training images
BATCH_SIZE = 128                             # images per training step, same as the class GAN notebook
WEIGHTS_PATH = "app/gan_mnist.pth"           # saved next to cnn_cifar10.pth so the API can load it

torch.manual_seed(42)                        # fixed seed so reruns start from the same random weights

# Use the Mac GPU (MPS) when available, otherwise CUDA, otherwise CPU, as in the class notebooks
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# Load MNIST with pixels scaled to [-1, 1] (already downloaded into data/ during the smoke test)
train_loader = get_data_loader("data", batch_size=BATCH_SIZE, train=True)

# Build the GAN wrapper (generator + discriminator) from the helper library
gan = get_model("GAN")

# Standard GAN loss: binary cross-entropy on the discriminator's logit (sigmoid applied inside the loss)
criterion = nn.BCEWithLogitsLoss()

# One Adam optimizer per network; lr 2e-4 and beta1 0.5 are the usual settings for a standard GAN
optimizer = (
    optim.Adam(gan.generator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
    optim.Adam(gan.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)),
)

# Train both networks
train_gan(gan, train_loader, criterion, optimizer, device=device, epochs=EPOCHS)

# Save only the generator: the API needs it to create digits, and the discriminator is only used in training
torch.save(gan.generator.state_dict(), WEIGHTS_PATH)
print(f"Saved generator weights to {WEIGHTS_PATH}")

# Plot 10 generated digits to check quality (close the plot window to end the script)
generate_samples(gan, device=device, num_samples=10)