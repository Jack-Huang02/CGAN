import torch
# Parameters
image_size = 32
batch_size = 100
workers = 0
dim_noise = 100
num_class = 10
lr_D = 2e-4
lr_G = 2e-4
num_epochs = 10
seed = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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