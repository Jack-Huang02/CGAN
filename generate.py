from model import generator,discriminator
from data_loader import get_minst_loader
import matplotlib.pyplot as plt
import torch
from torchvision import utils
from config import *

dataset, dataloader = get_minst_loader()
imgs = {}
for x, y in dataset:
    if y not in imgs:
        imgs[y] = []
    elif len(imgs[y])!=10:
        imgs[y].append(x)
    elif sum(len(imgs[key]) for key in imgs)==100:
        break
    else:
        continue
        
imgs = sorted(imgs.items(), key=lambda x:x[0])
imgs = [torch.stack(item[1], dim=0) for item in imgs]
imgs = torch.cat(imgs, dim=0)
imgs = utils.make_grid(imgs, nrow=10)
plt.figure(figsize=(20,10))

# Plot the real images
plt.subplot(1,2,1)
plt.axis('off')
plt.title("Real Images")
imgs = utils.make_grid(imgs, nrow=10)
plt.imshow(imgs.permute(1, 2, 0)*0.5+0.5)

generator = generator()
generator.load_state_dict(torch.load('./models/generator_0.pt', map_location=torch.device('cpu')))
generator.eval()

with torch.no_grad():
    fake = generator(fixed_noise.cpu(), fixed_label.cpu())
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
fake = utils.make_grid(fake, nrow=10)
plt.imshow(fake.permute(1, 2, 0)*0.5+0.5)

# Save the comparation result
plt.savefig('comparation.jpg', bbox_inches='tight')
