# helper_lib/trainer.py — standard GAN training loop (binary cross-entropy loss)
import torch
from tqdm import tqdm


def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Train a GAN wrapper (model.generator + model.discriminator).
    criterion: nn.BCEWithLogitsLoss() — applies the sigmoid to the discriminator's logit inside the loss
    optimizer: pair (opt_gen, opt_disc), one optimizer per network
    """
    gen, disc = model.generator, model.discriminator
    opt_gen, opt_disc = optimizer              # unpack the two optimizers
    model.to(device)                           # move both networks to the chosen device
    model.train()                              # training mode: BatchNorm uses batch statistics

    for epoch in range(epochs):
        loop = tqdm(data_loader, ncols=100, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_number, (real, _) in enumerate(loop):   # MNIST labels are ignored, as in the class notebook
            real = real.to(device)
            batch_size = real.size(0)

            # Targets for the BCE loss: 1 = real, 0 = fake
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            ## === Train Discriminator === ##
            noise = torch.randn(batch_size, model.z_dim, device=device)   # fresh noise vectors
            fake = gen(noise)                                             # generate a batch of fake images
            loss_real = criterion(disc(real), real_labels)                # push scores on real images toward 1
            loss_fake = criterion(disc(fake.detach()), fake_labels)       # push scores on fakes toward 0; detach = don't update gen here
            loss_disc = loss_real + loss_fake

            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ## === Train Generator === ##
            loss_gen = criterion(disc(fake), real_labels)   # generator wants the discriminator to call its fakes real (1)

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Show the latest losses on the progress bar every 100 batches
            if batch_number % 100 == 0:
                loop.set_postfix({"D loss": f"{loss_disc.item():.4f}", "G loss": f"{loss_gen.item():.4f}"})

    return model