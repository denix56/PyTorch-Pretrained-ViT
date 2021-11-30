from pytorch_pretrained_vit.model import ViT
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.datasets import STL10, ImageFolder
from torchvision.transforms import RandAugment, Compose, Resize, ToTensor, Normalize, RandomResizedCrop
from torchvision.utils import save_image
from torchvision import models
from tqdm import trange, tqdm
from torch import nn
from itertools import chain

from pytorch_pretrained_vit.gan_models import Generator, Discriminator


def train_teacher(loader, loader_test, device):
    regnet_y_16gf = models.regnet_y_16gf(pretrained=True)
    for param in regnet_y_16gf.parameters():
        param.requires_grad = False
    regnet_y_16gf.fc = nn.Linear(regnet_y_16gf.fc.in_features, 10)
    regnet_y_16gf = regnet_y_16gf.to(device)
    regnet_y_16gf.eval()

    opt = torch.optim.Adam(regnet_y_16gf.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 100

    for epoch in trange(n_epochs):
        loss_train = 0
        for batch_i, (X, y) in tqdm(enumerate(loader), total=len(loader)):
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            out = regnet_y_16gf(X)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            loss_train += loss.item()
        if epoch % 20 == 0:
            with torch.no_grad():
                loss_test = 0
                for batch_i, (X, y) in tqdm(enumerate(loader_test), total=len(loader_test), desc='Testing'):
                    X = X.to(device)
                    y = y.to(device)

                    out = regnet_y_16gf(X)
                    loss = criterion(out, y)

                    loss_test += loss.item()
                loss_train /= len(loader)
                loss_test /= len(loader_test)

                print('Epoch: {}, train loss: {}, test loss: {}'.format(epoch, loss_train, loss_test))

    torch.save({'model': regnet_y_16gf.state_dict(),
                'n_epochs': n_epochs,
                'opt': opt.state_dict()
                }, 'regnet_y.pth')


def train_mae(loader, loader_test, device):
    encoder = ViT('encoder', distilled=True, image_size=224).to(device)
    decoder = ViT('decoder', distilled=True, use_mask_token=True, num_layers=4, num_heads=6, ff_dim=1536,
                  image_size=224).to(device)

    opt = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    n_epochs = 200

    for epoch in trange(n_epochs):
        loss_train = 0
        encoder.train()
        decoder.train()
        for batch_i, (X, y) in tqdm(enumerate(loader), total=len(loader)):
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            out, mask = encoder(X, mask_rate=0.75)
            dec_out, _ = decoder(out, enc_mask=mask)
            loss = criterion(dec_out, X)
            loss.backward()
            opt.step()

            loss_train += loss.item()

        if epoch % 20 == 0:
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                loss_test = 0
                for batch_i, (X, y) in tqdm(enumerate(loader_test), total=len(loader_test), desc='Testing'):
                    X = X.to(device)
                    y = y.to(device)

                    out, mask = encoder(X, mask_rate=0.75)
                    dec_out, _ = decoder(out, enc_mask=mask)
                    loss = criterion(dec_out, X)

                    if batch_i == 0:
                        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])[None, -1, None, None],
                        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225])[None, -1, None, None]
                        X = TF.normalize(X, mean=mean, std=std)
                        dec_out = TF.normalize(dec_out, mean=mean, std=std)
                        save_image(X, 'gt_{}.png'.format(epoch))
                        save_image(dec_out, 'rec_{}.png'.format(epoch))

                    loss_test += loss.item()
                loss_train /= len(loader)
                loss_test /= len(loader_test)

                print('Epoch: {}, train loss: {}, test loss: {}'.format(epoch, loss_train, loss_test))

    torch.save({'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'n_epochs': n_epochs,
                'opt': opt.state_dict()
                }, 'mae.pth')


def train_gan(loader, loader_test, device):
    generator = Generator(2).to(device)
    discriminator = Discriminator().to(device)

    opt_g = torch.optim.Adam(generator.parameters(), betas=(0.0, 0.99), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), betas=(0.0, 0.99), lr=1e-4)
    criterion_mse = nn.MSELoss()
    #criterion_ns = lambda d_score: -torch.mean(torch.log(torch.sigmoid(d_score) + 1e-8))

    n_epochs = 1000
    latent_dim = 1024

    for epoch in trange(n_epochs):
        loss_g_train = 0
        loss_d_train = 0

        generator.train()
        discriminator.train()

        for batch_i, (X, y) in tqdm(enumerate(loader), total=len(loader)):
            X = X.to(device)
            y = y.to(device)

            valid = torch.ones(X.shape[0], 1, device=device)
            fake = torch.zeros(X.shape[0], 1, device=device)

            opt_d.zero_grad()
            z = torch.randn(X.shape[0], latent_dim, device=device)
            x_gen, _ = generator(z)
            d_score_valid = discriminator(X)
            d_score_fake = discriminator(x_gen)
            d_loss = 0.5*(criterion_mse(d_score_valid, valid) + criterion_mse(d_score_fake, fake))
            d_loss.backward()
            opt_d.step()

            loss_d_train += d_loss.item()

            opt_g.zero_grad()
            z = torch.randn(X.shape[0], latent_dim, device=device)
            x_gen, _ = generator(z)
            d_score_fake = discriminator(x_gen)
            g_loss = criterion_mse(d_score_fake, valid)
            g_loss.backward()
            opt_g.step()

            loss_g_train += g_loss.item()

        if epoch % 1 == 0:
            with torch.no_grad():
                generator.eval()
                discriminator.eval()

                z = torch.randn(16, latent_dim, device=device)

                x_gen, imgs = generator(z, ret_all_imgs=True)
                mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])[None, -1, None, None],
                std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225])[None, -1, None, None]
                x_gen = TF.normalize(x_gen, mean=mean, std=std)
                save_image(x_gen, 'generated_{}.png'.format(epoch))
                for i, img_i in enumerate(imgs):
                    img_i = TF.normalize(img_i, mean=mean, std=std)
                    save_image(img_i, 'generated_scale_{}_{}.png'.format(i, epoch))
                loss_g_train /= len(loader)
                loss_d_train /= len(loader)

                print('Epoch: {}, gen train loss: {}, disc train loss: {}'.format(epoch, loss_g_train, loss_d_train))

    torch.save({'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'n_epochs': n_epochs,
                'opt': opt.state_dict()
                }, 'vitgan.pth')


if __name__ == '__main__':
    device = 'cuda:0'

    mode = 'vitgan'

    if mode == 'teacher':
        transform = Compose([Resize(224),
                             RandAugment(),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                             ])
    elif mode == 'mae':
        transform = Compose([RandomResizedCrop(224),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                             ])
    elif mode == 'vitgan':
        transform = Compose([Resize(224),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                             ])

    transform_test = Compose([Resize(224),
                              ToTensor(),
                              Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                              ])

    batch_size = 16
    n_workers = 4

    if mode == 'vitgan':
        dataset = ImageFolder('C:/Users/denys/PyTorch-Pretrained-ViT/afhq/train', transform=transform)
        dataset_test = ImageFolder('C:/Users/denys/PyTorch-Pretrained-ViT/afhq/train', transform=transform_test)
    else:
        dataset = STL10('stl10', transform=transform, split='train', download=True)
        dataset_test = STL10('stl10', transform=transform_test, split='test', download=True)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, persistent_workers=(n_workers > 0), shuffle=True, pin_memory=True)

    loader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=n_workers, persistent_workers=False, shuffle=False,
                             pin_memory=True)

    if mode == 'teacher':
        train_teacher(loader, loader_test, device)
    elif mode == 'mae':
        train_mae(loader, loader_test, device)
    elif mode == 'vitgan':
        train_gan(loader, loader_test, device)



