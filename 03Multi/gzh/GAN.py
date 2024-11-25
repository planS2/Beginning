import argparse
import os
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches") 
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # 随机噪声z的维度
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension") # 输入图像的尺寸
parser.add_argument("--channels", type=int, default=1, help="number of image channels") # 输入图像的channel数
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples") # 保存生成图像和模型的间隔
opt = parser.parse_known_args()[0]

img_shape = (opt.channels, opt.img_size, opt.img_size)

###数据集

# 数据加载，加载手写体数据集
# 加载数据格式为28*28的灰度图，和其对应的0-9的数字标签

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),

    batch_size=opt.batch_size,
    shuffle=True,
)

# show函数
from torch.autograd import Variable
import matplotlib.pyplot as plt

def show_img(img, trans=True):
    if trans:
        img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))  # 把channel维度放到最后
        plt.imshow(img[:, :, 0], cmap="gray")
    else:
        plt.imshow(img, cmap="gray")
    plt.show()
    
mnist = datasets.MNIST("../../data/mnist")


    
# 数据
trans_resize = transforms.Resize(opt.img_size)
trans_to_tensor = transforms.ToTensor()
trans_normalize = transforms.Normalize([0.5], [0.5]) # x_n = (x - 0.5) / 0.5



# 模型

# 生成器
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
    
generator = Generator()


# 判别器

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
    
discriminator = Discriminator()


# BCE损失函数
adversarial_loss = torch.nn.BCELoss()

cuda = True if torch.cuda.is_available() else False
print("cuda_is_available =", cuda)
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#Adam优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
print("learning_rate =", opt.lr)


# 创建输入
for i, (imgs, _) in list(enumerate(dataloader))[:1]:
    real_imgs = Variable(imgs.type(Tensor))
    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
    print("i =", i, '\n')
    print("shape of z =", z.shape, '\n')
    print("shape of real_imgs =", real_imgs.shape, '\n')
    print("z =", z, '\n')
    print("real_imgs =")
    for img in real_imgs[:3]:
        show_img(img)




# 计算loss反向传播

    # Adversarial ground truths
    valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False) # 为1时判定为真
    fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False) # 为0时判定为假
    
    
    #训练生成器
   
    
    optimizer_G.zero_grad()
    
    gen_imgs = generator(z) # 生成图像
    print("gen_imgs =")
    for img in gen_imgs[:3]:
        show_img(img)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)
    print("g_loss =", g_loss, '\n')

    g_loss.backward()
    optimizer_G.step()

    # 训练判别器

    optimizer_D.zero_grad()
    
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2
    print("real_loss =", real_loss, '\n')
    print("fake_loss =", fake_loss, '\n')
    print("d_loss =", d_loss, '\n')    
    
    d_loss.backward()
    optimizer_D.step()


# 保存

    from torchvision.utils import save_image

    epoch = 0 
    batches_done = epoch * len(dataloader) + i
    if batches_done % opt.sample_interval == 0:
        save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True) # 保存生成图像
        
        os.makedirs("model", exist_ok=True) # 保存模型
        torch.save(generator, 'model/generator.pkl') 
        torch.save(discriminator, 'model/discriminator.pkl')
        
        print("gen images saved!\n")
        print("model saved!")