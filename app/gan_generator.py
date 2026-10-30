# loads the trained GAN generator once and turns generated digits into a PNG
import io
from pathlib import Path
import torch
import torchvision.transforms.functional as TF

from helper_lib.model import Generator                 # same architecture used in training
from helper_lib.generator import generate_samples      # class-activity sampling function

# Locate the weights file relative to THIS file, not the current working directory
WEIGHTS_PATH = Path(__file__).parent / "gan_mnist.pth"

# Build the generator once, load the trained weights, switch to evaluation mode
# Runs once when the server starts, not on every request
generator = Generator()
generator.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
generator.eval()


def generate_digits_png(num_samples: int = 10, seed: int | None = None) -> bytes:
    """Generate num_samples handwritten digits from random noise and return them as one PNG grid."""
    # Optional seed: the same seed always returns the same digits
    if seed is not None:
        torch.manual_seed(seed)

    # Reuse the helper library: random noise -> generator -> grid of images in [0, 1]
    grid = generate_samples(generator, device="cpu", num_samples=num_samples, show=False)

    # Convert the grid tensor (3, H, W) to a PIL image, then to PNG bytes in memory
    image = TF.to_pil_image(grid)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()