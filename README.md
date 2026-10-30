# sps_genai

FastAPI app with a bigram text generator, spaCy word embeddings, a CNN image classifier, and a GAN that generates handwritten digits.

## Run with Docker

```bash
docker build -t asm3-genai .
docker run --rm -p 8000:80 asm3-genai
```

Docs: http://127.0.0.1:8000/docs

## Endpoints

- `POST /generate`: `{"start_word": "bigram", "length": 5}`
- `POST /embedding`: `{"word": "apple"}` returns a 300-dimension vector
- `POST /similarity`: `{"word1": "apple", "word2": "orange"}`
- `POST /classify`: upload an image file, returns the predicted CIFAR10 class
- `POST /generate_digits`: `{"num_samples": 10, "seed": 42}` returns a PNG grid of GAN-generated digits (`num_samples` 1-64, `seed` optional)

## Examples

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'

curl -X POST http://127.0.0.1:8000/classify -F "file=@your_image.png"

curl -X POST http://127.0.0.1:8000/generate_digits \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 10, "seed": 42}' \
  -o digits.png
```

## Assignment 2: CNN image classifier

`POST /classify` accepts an image upload and returns the predicted CIFAR10 class (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) with probabilities.

Model: `app/cnn_model.py` (Conv16 -> Pool -> Conv32 -> Pool -> FC100 -> FC10, 64x64x3 input). Trained with `train_cnn.py` on CIFAR10 (resized to 64x64); weights saved in `app/cnn_cifar10.pth`.

## Assignment 3: GAN digit generator

`POST /generate_digits` returns a PNG grid of handwritten digits created by a GAN trained on MNIST. The same `seed` always returns the same digits.

Helper library (`helper_lib/`, Module 6 class activity):
- `model.py`: `Generator`, `Discriminator`, a `GAN` wrapper holding both, and `get_model("GAN")`
- `trainer.py`: `train_gan` (standard GAN training with binary cross-entropy)
- `generator.py`: `generate_samples` (samples noise, runs the generator, plots or returns the image grid)
- `data_loader.py`: `get_data_loader` (MNIST scaled to [-1, 1])

Architecture:
- Generator: noise (100) -> Linear to 7x7x128 -> ConvTranspose 128->64 (14x14) + BatchNorm + ReLU -> ConvTranspose 64->1 (28x28) + Tanh
- Discriminator: image (1x28x28) -> Conv 1->64 (14x14) + LeakyReLU(0.2) -> Conv 64->128 (7x7) + BatchNorm + LeakyReLU(0.2) -> Flatten -> Linear to 1 output

The discriminator's Linear layer outputs a logit; the sigmoid that turns it into the real/fake probability is applied inside `nn.BCEWithLogitsLoss` during training.

Trained with `train_gan.py` (10 epochs, Adam lr 2e-4, batch size 128); generator weights saved in `app/gan_mnist.pth`.