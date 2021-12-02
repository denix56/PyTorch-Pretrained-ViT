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
import torch.nn.functional as F
from itertools import chain
import os
import shutil

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


def train_gan(loader_a, loader_b, loader_a_test, loader_b_test, device, blocks=6, mode='none', use_siren=False):
    generator_ab = Generator(image_size=224, blocks=blocks, mode=mode, use_siren=use_siren).to(device)
    discriminator_ab = Discriminator(image_size=224, blocks=blocks).to(device)
    generator_ba = Generator(image_size=224, blocks=blocks, mode=mode, use_siren=use_siren).to(device)
    discriminator_ba = Discriminator(image_size=224, blocks=blocks).to(device)

    opt_g_ab = torch.optim.Adam(generator_ab.parameters(), lr=1e-4)
    opt_d_ab = torch.optim.Adam(discriminator_ab.parameters(), lr=1e-4)
    opt_g_ba = torch.optim.Adam(generator_ba.parameters(), lr=1e-4)
    opt_d_ba = torch.optim.Adam(discriminator_ba.parameters(), lr=1e-4)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    n_epochs = 10000
    img_out_dir = '/home/dsenkin/Desktop/scratch/images_' + mode + ('_siren' if use_siren else '')

    shutil.rmtree(img_out_dir, ignore_errors=True)
    os.makedirs(img_out_dir, exist_ok=True)

    for epoch in trange(n_epochs):
        loss_g_ab_train = 0
        loss_d_ab_train = 0

        loss_g_ba_train = 0
        loss_d_ba_train = 0

        generator_ab.train()
        discriminator_ab.train()
        generator_ba.train()
        discriminator_ba.train()

        lmbda = 10

        for batch_i, ((X_a, _), (X_b, _)) in tqdm(enumerate(zip(loader_a, loader_b)), total=min(len(loader_a), len(loader_b))):
            X_a = X_a.to(device)
            X_b = X_b.to(device)

            batch_size = min(X_a.shape[0], X_b.shape[0])
            X_a = X_a[:batch_size]
            X_b = X_b[:batch_size]

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            opt_d_ab.zero_grad()
            x_b_gen, _ = generator_ab(X_a)
            d_score_valid = discriminator_ab(X_b)
            d_score_fake = discriminator_ab(x_b_gen)
            d_ab_loss = 0.5*(criterion_mse(d_score_valid, valid) + criterion_mse(d_score_fake, fake))
            d_ab_loss.backward()
            opt_d_ab.step()
            loss_d_ab_train += d_ab_loss.item()

            opt_d_ba.zero_grad()
            x_a_gen, _ = generator_ba(X_b)
            d_score_valid = discriminator_ba(X_a)
            d_score_fake = discriminator_ba(x_a_gen)
            d_ba_loss = 0.5 * (criterion_mse(d_score_valid, valid) + criterion_mse(d_score_fake, fake))
            d_ba_loss.backward()
            opt_d_ba.step()
            loss_d_ba_train += d_ba_loss.item()


            opt_g_ab.zero_grad()
            opt_g_ba.zero_grad()

            x_b_gen, x_b_flat = generator_ab(X_a)
            d_ab_score_fake = discriminator_ab(x_b_gen)

            x_a_gen, x_a_flat = generator_ba(X_b)
            d_ba_score_fake = discriminator_ba(x_a_gen)

            x_a_cyc, _ = generator_ba(x_b_gen, mem=x_b_flat)
            x_b_cyc, _ = generator_ab(x_a_gen, mem=x_a_flat)

            g_ab_loss = criterion_mse(d_ab_score_fake, valid) + lmbda*criterion_mae(x_a_cyc, X_a)
            g_ba_loss = criterion_mse(d_ba_score_fake, valid) + lmbda*criterion_mae(x_b_cyc, X_b)
            g_loss = g_ab_loss + g_ba_loss
            g_loss.backward()
            opt_g_ba.step()
            opt_g_ab.step()
            loss_g_ab_train += g_ab_loss.item()
            loss_g_ba_train += g_ba_loss.item()

        if epoch % 1 == 0:
            with torch.no_grad():
                generator_ab.eval()
                discriminator_ab.eval()
                generator_ba.eval()
                discriminator_ba.eval()

                X_a, _ = next(iter(loader_a_test))
                X_b, _ = next(iter(loader_b_test))

                X_a = X_a.to(device)
                X_b = X_b.to(device)

                x_b_gen, _ = generator_ab(X_a)
                x_a_gen, _ = generator_ba(X_b)

                mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])[None, -1, None, None],
                std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225])[None, -1, None, None]
                X_a = TF.normalize(X_a, mean=mean, std=std)
                x_a_gen = TF.normalize(x_a_gen, mean=mean, std=std)
                X_b = TF.normalize(X_b, mean=mean, std=std)
                x_b_gen = TF.normalize(x_b_gen, mean=mean, std=std)

                save_image(X_a, os.path.join(img_out_dir, 'gt_a_{}.png').format(epoch))
                save_image(x_a_gen, os.path.join(img_out_dir, 'generated_a_{}.png').format(epoch))
                save_image(X_b, os.path.join(img_out_dir, 'gt_b_{}.png').format(epoch))
                save_image(x_b_gen, os.path.join(img_out_dir, 'generated_b_{}.png').format(epoch))

                loss_g_ab_train /= len(loader_a)
                loss_g_ba_train /= len(loader_b)
                loss_d_ab_train /= len(loader_a)
                loss_d_ba_train /= len(loader_b)

                print('Epoch: {}, gen a train loss: {}, gen b train loss: {}, disc a train loss {}, disc b train loss: {}'.format(
                    epoch, loss_g_ab_train, loss_g_ba_train, loss_d_ab_train, loss_d_ba_train))

        if epoch % 5000 == 0:
            torch.save({'generator_ab': generator_ab.state_dict(),
                        'discriminator_ab': discriminator_ab.state_dict(),
                        'generator_ba': generator_ba.state_dict(),
                        'discriminator_ba': discriminator_ba.state_dict(),
                        'epoch': epoch,
                        'opt_g_ab': opt_g_ab.state_dict(),
                        'opt_g_ba': opt_g_ba.state_dict(),
                        'opt_d_ab': opt_d_ab.state_dict(),
                        'opt_d_ba': opt_d_ba.state_dict()
                        }, ('cycle_vitgan_{}_' + mode + ('_siren' if use_siren else '') + '.pth').format(epoch))


if __name__ == '__main__':
    device = 'cuda:0'

    mode = 'cycle'

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
                             RandAugment(),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                             ])
    elif mode == 'cycle':
        transform = Compose([Resize(224),
                             RandAugment(),
                             ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                             ])

    transform_test = Compose([Resize(224),
                              ToTensor(),
                              Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                              ])


    batch_size = 32
    n_workers = 4
    root = '/home/dsenkin/Desktop/scratch/monet2photo'

    if mode == 'vitgan':
        dataset = ImageFolder('C:/Users/denys/PyTorch-Pretrained-ViT/afhq/train', transform=transform)
        dataset_test = ImageFolder('C:/Users/denys/PyTorch-Pretrained-ViT/afhq/train', transform=transform_test)
    elif mode == 'cycle':
        dataset_a = ImageFolder(os.path.join(root, 'train/trainA'), transform=transform)
        dataset_b = ImageFolder(os.path.join(root, 'train/trainB'), transform=transform)
        dataset_a_test = ImageFolder(os.path.join(root, 'test/testA'),
                                     transform=transform_test)
        dataset_b_test = ImageFolder(os.path.join(root, 'test/testB'),
                                     transform=transform_test)
    else:
        dataset = STL10('stl10', transform=transform, split='train', download=True)
        dataset_test = STL10('stl10', transform=transform_test, split='test', download=True)

    if mode == 'cycle':
        loader_a = DataLoader(dataset_a, batch_size=batch_size, num_workers=n_workers, persistent_workers=(n_workers > 0),
                            shuffle=True, pin_memory=True)

        loader_a_test = DataLoader(dataset_a_test, batch_size=batch_size, num_workers=0, persistent_workers=False,
                                 shuffle=True, pin_memory=True)

        loader_b = DataLoader(dataset_b, batch_size=batch_size, num_workers=n_workers,
                              persistent_workers=(n_workers > 0),
                              shuffle=True, pin_memory=True)

        loader_b_test = DataLoader(dataset_b_test, batch_size=batch_size, num_workers=0, persistent_workers=False,
                                   shuffle=True, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, persistent_workers=(n_workers > 0), shuffle=True, pin_memory=True)

        loader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=n_workers, persistent_workers=False, shuffle=False,
                                 pin_memory=True)

    if mode == 'teacher':
        train_teacher(loader, loader_test, device)
    elif mode == 'mae':
        train_mae(loader, loader_test, device)
    elif mode == 'vitgan':
        train_gan(loader, loader_test, device)
    elif mode == 'cycle':
        train_gan(loader_a, loader_b, loader_a_test, loader_b_test, device, mode='none', use_siren=False)



