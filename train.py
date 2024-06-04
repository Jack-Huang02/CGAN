from config import *
import torch.nn as nn
import torch
from model import weights_init
import model
from data_loader import get_minst_loader
import time
from torchvision import utils

# 固定种子
torch.manual_seed(seed)

# 定义生成器和判别器，并分别对权重进行初始化
generator = model.generator().to(device)
discriminator = model.discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

# 定义损失函数
loss_function = nn.BCELoss()

# 定义俩个变量
real_label_num = 1
fake_label_num = 0

# 定义优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr_G, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr_D, betas=(0.5, 0.999))

img_list = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []
loss_tep = 10

epochs = num_epochs
# 取训练数据
dataset, dataloader = get_minst_loader()

print("Starting Training Loop...")

for epoch in range(num_epochs):
    # 记录开始时间
    beg_time = time.time()
    for i, data in enumerate(dataloader):
        # Trian Discriminator
        # 梯度清零
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
