import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from cgan import Discriminator, Generator, initialize_weights
# from utils import gradient_penalty
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

torch.manual_seed(0)

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 0.0001
BATCH_SIZE = 50
IMG_SIZE = 129
CHANNELS_IMG = 1
NUM_CLASSES =4
GEN_EMBEDDING=100
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

class CustomDataset(Dataset):
    def __init__(self):
        # self.imgs_path = "/Users/salvatoreesposito/Documents/fourier_signals/"
        self.imgs_path = "/disk/scratch/datasets/fourier_signals/"
        file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for npy_path in glob.glob(class_path + "/*.npy"):
                self.data.append([npy_path, class_name])
        # print(self.data)
        self.class_map = {"0": 0, "1": 1, "2": 2, "3": 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        # img_tensor = torch.from_numpy(np.load(npy_path))
        img_tensor = torch.unsqueeze(torch.from_numpy(np.load(npy_path)), dim=0)
        # print(img_tensor.shape)
        # input('enter')
        # img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        # if self.transform:
        #     img_tensor = self.transform(img_tensor)
        # if self.transform:
        #     class_id = self.target_transform(class_id)
        return img_tensor, class_id


dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, IMG_SIZE, NUM_CLASSES).to(device).double()
criterion = nn.BCELoss()
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# for tensorboard plotting
fixed_noise = torch.randn(5, Z_DIM, 1, 1).to(device)
# writer_real=SummaryWriter(f"/Users/salvatoreesposito/Documents/Github/2DCGAN/logs/real")
# writer_fake=SummaryWriter(f"/Users/salvatoreesposito/Documents/Github/2DCGAN/logs/fake")
writer_real=SummaryWriter(f"/disk/scratch/cgan_logs/real")
writer_fake=SummaryWriter(f"/disk/scratch/cgan_loss/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, labels) in enumerate(data_loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)
        # img_grid_real = torchvision.utils.make_grid(real[:,0,:,:], normalize=True, nrow=12, padding=5)
        # writer_real.add_image("Real INITIAL", img_grid_real, global_step=step)
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1, 1).to(device)
        fake = gen(noise, labels)
        critic_real = critic(real, labels).reshape(-1)
        critic_fake = critic(fake, labels).reshape(-1)
        loss_real = criterion(critic_real, torch.ones_like(critic_real))
        loss_fake = criterion(critic_fake, torch.zeros_like(critic_fake))
        loss_disc = (loss_real + loss_fake) / 2
        # gp = gradient_penalty(critic, real, fake, device=device)
        # loss_critic = (
        #     -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
        # )
        critic.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = criterion(gen_fake, torch.ones_like(gen_fake))
        gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        opt_gen.step()

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(data_loader)} \
              Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
        )

        with torch.no_grad():
            to_visualise = 50
            noise = torch.randn(to_visualise, Z_DIM, 1, 1, 1).to(device)
            fake = gen(noise, labels)
            # print(fake.shape)
            # print(real.shape)
            # noise_t = torch.squeeze(fake, 1)
            # noise_f = torch.flatten(noise_t, start_dim=0, end_dim=1)
            # noise_u = torch.unsqueeze(noise_f, 1)
            # current_real = real[:to_visualise]
            # current_real_t = torch.squeeze(current_real, 1)
            # current_real_f = torch.flatten(current_real_t, start_dim=0, end_dim=1)
            # current_real_u = torch.unsqueeze(current_real_f, 1)
            # take out (up to) 32 examples
            current_real_u = torch.squeeze(real, 1)
            current_real_u = current_real_u.reshape(-1,1, 64,6)
            # print(current_real_u.shape)
            current_fake_f = torch.squeeze(fake[:real.shape[0]], 0).reshape(-1,1, 64,6)
            # print(current_fake_f.shape)
            img_grid_real = torchvision.utils.make_grid(current_real_u, normalize=True, nrow=12)
            img_grid_fake = torchvision.utils.make_grid(current_fake_f, normalize=True, nrow=12)

            writer_real.add_image("Real", img_grid_real, global_step=step)
            writer_fake.add_image("Fake", img_grid_fake, global_step=step)


        step += 1