from config import *
import torch.nn as nn
import torch
from model import weights_init
import model
from data_loader import get_minst_loader
import time
import torch.utils as utils

torch.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
generator = model.generator().to(device)
discriminator = model.discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

loss_function = nn.BCELoss()

real_label_num = 1
fake_label_num = 0

optimizer_G = torch.optim.Adam(generator.parameters(), lr_G, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr_D, betas=(0.5, 0.999))

label_one_hot = torch.zeros(10, 10)
for i in range(10):
    label_one_hot[i][i] = 1
label_one_hot = label_one_hot.view(10, 10, 1, 1).to(device)

label_fills = torch.zeros(10, 10, image_size, image_size)
ones = torch.ones(image_size, image_size)
for i in range(10):
    label_fills[i][i] = ones
label_fills = label_fills.to(device)

fixed_noise = torch.randn(100, dim_noise, 1, 1).to(device)
fixed_label = label_one_hot[torch.arange(10).repeat(10).sort().values]

img_list = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []
loss_tep = 10

epochs = num_epochs
dataset, dataloader = get_minst_loader()

print("Starting Training Loop...")

for epoch in range(num_epochs):
    beg_time = time.time()
    for i, data in enumerate(dataloader):
        # Trian Discriminator
        discriminator.zero_grad()
        real_image = data[0].to(device)
        b_size = real_image.size(0)

        real_label = torch.full((b_size, ), real_label_num).to(device)
        fake_label = torch.full((b_size, ), fake_label_num).to(device)

        G_label = label_one_hot[data[1]]
        D_label = label_fills[data[1]]

        output = discriminator(real_image, D_label).view(-1)
        errD_real = loss_function(output, real_label.float())
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, dim_noise, 1, 1).to(device)
        fake = generator(noise, G_label)
        output = discriminator(fake.detach(), D_label).view(-1)
        errD_fake = loss_function(output, fake_label.float())
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizer_D.step()

        # Train Generator
        generator.zero_grad()
        output = discriminator(fake, D_label).view(-1)
        errG = loss_function(output, real_label.float())
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizer_G.step()

        end_time = time.time() 
        run_time = round(end_time - beg_time)

        print(
            f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
            f'Step: [{i+1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
            f'Loss-D: {errD.item():.4f}',
            f'Loss-G: {errG.item():.4f}',
            f'D(x): {D_x:.4f}',
            f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
            f'Time: {run_time}s',
            end='\r'
        )

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        D_x_list.append(D_x)
        D_z_list.append(D_G_z1)

        if errG < loss_tep:
            torch.save(generator.state_dict(), f'./models/generator_{epoch}.pt')
            loss_tep = errG
    with torch.no_grad():
        fake = generator(fixed_noise, fixed_label).detach().cpu()
    img_list.append(utils.make_grid(fake, nrow=10))
    print()
