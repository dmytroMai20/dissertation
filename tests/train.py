import torch
import torchvision
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from data import dataset
from torch.utils.data import DataLoader
from model.unet_base import Unet
from schedulers.LinearNoise import LinearNoise
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer(model, epoch, scheduler,  train_config, model_config, diffusion_config):
    """
        Used to infer and produce an image during training at checkpoints.
    """
    model.eval()
    with torch.no_grad():
        xt = torch.randn((train_config['num_samples_test'],
                      model_config['im_channels'],
                      model_config['im_size'],
                      model_config['im_size'])).to(device)
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
            
            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.reverse_process(xt, noise_pred, torch.as_tensor(i).to(device))
            
            # Save x0
            
        ims = torch.clamp(xt, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
            #imgs = []
            #if not os.path.exists(os.path.join(train_config['task_name'], 'training_samples')):
            #    os.mkdir(os.path.join(train_config['task_name'], 'training_samples'))
            #for j in range(ims.shape[0]):
            #    imgs.append(torchvision.transforms.ToPILImage()(ims[i]))
            #    imgs[j].save(os.path.join(train_config['task_name'], 'training_samples', 'x0_{}.png'.format(i)))
            #    imgs[i].close()
        if not os.path.exists(os.path.join(train_config['task_name'], 'training_samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'training_samples'))
        for i in range(ims.shape[0]):
            img = torchvision.transforms.ToPILImage()(ims[i])
            img.save(os.path.join(train_config['task_name'], 'training_samples', 'x0_{}_{}.png'.format(epoch, i)))
            img.close()
    model.train()

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = LinearNoise(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # Create the dataset
    #mnist = MnistDataset('train', im_path=dataset_config['im_path'])
    loader = dataset.get_loader(batch_s=train_config['batch_size'])
    
    # Instantiate the model
    model = Unet(model_config).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ckpt_name']), map_location=device))
    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    images_seen = 0
    loss_history = []

    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for im, labels in tqdm(loader): #im shape (batch_size, channel, h, w)
            optimizer.zero_grad()
            im = im.float().to(device)
            images_seen+=im.shape[0]
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.forward_process(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            avg_losses = np.mean(losses)
            loss_history.append(avg_losses)
            infer(model, epoch_idx, scheduler, train_config, model_config, diffusion_config)    # added infer step to see progression of images
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            avg_losses,
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))
        np.save(os.path.join(train_config['task_name'],train_config['ckpt_name'], 'loss_history.npy'), np.array(loss_history))
    print('Done Training ...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)