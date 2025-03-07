import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from data import dataset
from schedulers.LinearNoise import LinearNoise

def im_show(img):
    """Display a single CelebA image."""
    print(img)
    img = img * 0.5 + 0.5  # Denormalize
    #npimg = img.permute(1, 2, 0).numpy()  # Convert to (H, W, C) for matplotlib
    print(img.shape)
    npimg = np.transpose(img, (1, 2, 0))
    plt.imshow(npimg)
    plt.axis("off")
    plt.show()

def imshow_batch(images, num_images=8):
    """Display a batch of CelebA images."""
    images = images * 0.5 + 0.5  # Denormalize
    print(f"batch shape: {images.shape}")
    #np_images = images.permute(0, 2, 3, 1).numpy()  # Convert to (N, H, W, C)
    np_images = np.transpose(images, (1, 2, 0))
    print(f"np_batch shape: {np_images.shape}")
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    for i in range(num_images):
        axes[i].imshow(np_images[i])
        axes[i].axis("off")

    plt.show()

def get_img():
    """Fetch a single image from CelebA dataset."""
    loader = dataset.get_loader(64, "CelebA")
    print(f"loader: {loader}")
    data_iter = iter(dataset.get_loader(64,"CelebA"))
    images,labels = next(data_iter)  # CelebA may not have labels
    rand_id = random.randint(0, len(images) - 1)
    return images[rand_id]

def main():
    """Display CelebA image and diffusion noise process."""
    images = []
    image = get_img()
    im_show(image)  # Show original image

    ln = LinearNoise(0.0001, 0.02, 1000)
    for i in range(0, 1000, 100):
        noise = torch.randn_like(image)
        print(f"image shape: {image.shape}, noise: {noise.shape}")
        images.append(ln.forward_process(image.unsqueeze(0), noise.unsqueeze(0), i))

    images = torch.stack(images)
    imshow_batch(images)  # Show diffusion steps

if __name__ == "__main__":
    main()