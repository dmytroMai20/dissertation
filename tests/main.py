import matplotlib.pyplot as plt
import numpy as np
from data import dataset
from schedulers.LinearNoise import LinearNoise
import random
import torch

def im_show(img, label):
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    img = img * 0.5 + 0.5  # Denormalize
    npimg = img.numpy().squeeze()
    plt.imshow(npimg, cmap="gray")
    plt.title(classes[label])
    plt.axis("off")
    plt.show()

def imshow_batch(images, labels, num_images=8):
    images = images * 0.5 + 0.5  # Denormalize
    np_images = images.numpy()
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    
    for i in range(num_images):
        axes[i].imshow(np_images[i].squeeze(), cmap="gray")
        axes[i].set_title(classes[labels[i]])
        axes[i].axis("off")

    plt.show()
def get_img():
    # Get a batch of images
    data_iter = iter(dataset.get_loader())
    images, labels = next(data_iter)

    # Show an example image
    rand_id = random.randint(0,len(images))
    print(len(images))
    return images[rand_id],labels[rand_id].item()

def main():
    # Get random image from dataset, do some diffusion steps and plot results
    images = []
    image,label = get_img()
    im_show(image,label)
    ln = LinearNoise(0.0001,0.02, 1000)
    for i in range(0,1000,100):
        noise = torch.randn_like(image)
        images.append(ln.forward_process(image, noise, i))
    images = torch.stack(images)
    labels = [0 for x in images]
    imshow_batch(images, labels)
if __name__=="__main__":
    main()