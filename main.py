from pytorch_pretrained_vit.model import ViT
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.datasets import STL10, ImageFolder
from torchvision.transforms import RandAugment, Compose, Resize, ToTensor, Normalize, RandomResizedCrop
from torchvision.utils import save_image, make_grid
from torchvision import models
from tqdm import trange, tqdm
from torch import nn
import torch.nn.functional as F
from itertools import chain
import os
import sys
import shutil
import uuid

from pytorch_pretrained_vit.gan_models import Generator, Discriminator

import pytorch_lightning as pl
from torchmetrics.image.lpip_similarity import LPIPS
from torchmetrics import PSNR, MeanSquaredError, MetricCollection, Accuracy, Precision, Recall, AUROC, F1, AveragePrecision


class MAE(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, mask_rate: float = 0.75, root: str = '', name: str = ''):
        super().__init__()
        self.lr = lr
        self.mask_rate = mask_rate
        self.root = root
        self.name = name

        self.mask_generator = ViT('mask_generator', distilled=False, num_layers=4, num_heads=6, ff_dim=1536,
                                  image_size=224, mask_gen=True)
        self.encoder = ViT('encoder', distilled=True, image_size=224)
        self.decoder = ViT('decoder', distilled=True, use_mask_token=True, num_layers=4, num_heads=6, ff_dim=1536,
                           image_size=224)

        self.criterion = nn.L1Loss()

        self.metrics_train = MetricCollection([
            MeanSquaredError(),
            PSNR(data_range=2)
        ], prefix='train/')

        self.metrics_val = MetricCollection([
            MeanSquaredError(compute_on_step=False),
            PSNR(data_range=2, compute_on_step=False),
        ], prefix='val/')

        self.lpips_metric = LPIPS(compute_on_step=False)

        self.save_hyperparameters(ignore=['root', 'name'])

        self.example_input_array = torch.rand(1, 3, 224, 224)

    def forward(self, x):
        mask_probs, _, _ = self.mask_generator(x)
        mask = torch.bernoulli(mask_probs)
        mask = mask_probs + (mask - mask_probs).detach()
        out, mask, lengths = self.encoder(x, mask=mask)
        dec_out, _, _ = self.decoder(out, mask=mask, lengths=lengths)

        return dec_out

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch

        if optimizer_idx == 0:
            mask_probs, _, _ = self.mask_generator(x)
            mask = torch.bernoulli(mask_probs)

            mask = mask_probs + (mask - mask_probs).detach()

            out, enc_mask, lengths = self.encoder(x, mask=mask)
            dec_out, _, _ = self.decoder(out, mask=enc_mask, lengths=lengths)

            mse_loss = -self.criterion(dec_out, x)
            const_loss = torch.mean((torch.mean(mask_probs, dim=(1, 2)) - 0.25) ** 2)
            loss = mse_loss + const_loss
            self.log('mask_loss', loss)
            self.log('mse_loss', mse_loss)
            self.log('const_loss', const_loss)

        elif optimizer_idx == 1:
            mask_probs, _, _ = self.mask_generator(x)
            mask = torch.bernoulli(mask_probs)

            out, mask, lengths = self.encoder(x, mask=mask)
            dec_out, _, _ = self.decoder(out, mask=mask, lengths=lengths)

            loss = self.criterion(dec_out, x)

            self.log('Loss', loss)
            metrics_out = self.metrics_train(dec_out, x)
            self.log_dict(metrics_out)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mask_probs = None
        mask_probs, _, _ = self.mask_generator(x)
        mask = torch.bernoulli(mask_probs)
        out, mask, lengths = self.encoder(x, mask=mask)
        dec_out, _, _ = self.decoder(out, mask=mask, lengths=lengths)

        self.metrics_val(dec_out, x)

        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])[None, -1, None, None],
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225])[None, -1, None, None]
        x = TF.normalize(x, mean=mean, std=std)
        x_min = torch.amin(x, dim=(2, 3), keepdim=True)
        x_max = torch.amax(x, dim=(2, 3), keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min) * 2 - 1
        dec_out = TF.normalize(dec_out, mean=mean, std=std)
        dec_out_min = torch.amin(dec_out, dim=(2, 3), keepdim=True)
        dec_out_max = torch.amax(dec_out, dim=(2, 3), keepdim=True)
        dec_out_norm = (dec_out - dec_out_min) / (dec_out_max - dec_out_min) * 2 - 1

        self.lpips_metric(dec_out_norm, x_norm)

        if batch_idx == 0:
            x = make_grid(x, normalize=True)
            dec_out = make_grid(dec_out, normalize=True)

            self.logger.experiment.add_image('GT', x, global_step=self.global_step)
            self.logger.experiment.add_image('Reconstructed', dec_out, global_step=self.global_step)

            save_image(x, os.path.join(self.root, self.name, 'imgs/gt_{}.jpg'.format(self.current_epoch)))
            save_image(dec_out, os.path.join(self.root, self.name, 'imgs/rec_{}.jpg'.format(self.current_epoch)))

            if mask_probs is not None:
                mask_probs = mask_probs.view(-1, 1, self.mask_generator.gh, self.mask_generator.gw)
                self.logger.experiment.add_images('Mask probs', mask_probs, global_step=self.global_step)
                save_image(mask_probs, os.path.join(self.root, self.name, 'imgs/mp_{}.jpg'.format(self.current_epoch)))

    def on_validation_epoch_end(self):
        metrics_out = self.metrics_val.compute()
        metrics_out['val/LPIPS'] = self.lpips_metric.compute()
        self.log_dict(metrics_out)

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.mask_generator.parameters(), lr=self.lr)
        opt2 = torch.optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=self.lr)
        return [opt1, opt2]


class MAEClassifier(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, num_classes:int = 10, freeze:bool = False, root: str = '', name: str = '', *args):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.freeze = freeze
        self.root = root
        self.name = name

        self.encoder = ViT('encoder', distilled=True, image_size=224, dim=768)
        self.classifier = nn.Linear(768, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        self.metrics_train = MetricCollection([
            Accuracy(num_classes=num_classes),
            Precision(num_classes=num_classes),
            Recall(num_classes=num_classes),
            AUROC(num_classes=num_classes),
            F1(num_classes=num_classes),
            AveragePrecision(num_classes=num_classes)
        ], prefix='train/')

        self.metrics_val = MetricCollection([
            Accuracy(num_classes=num_classes, compute_on_step=False, average='macro'),
            Precision(num_classes=num_classes, compute_on_step=False, average='macro'),
            Recall(num_classes=num_classes, compute_on_step=False, average='macro'),
            AUROC(num_classes=num_classes, compute_on_step=False, average='macro'),
            F1(num_classes=num_classes, compute_on_step=False, average='macro'),
            AveragePrecision(num_classes=num_classes, compute_on_step=False, average='macro')
        ], prefix='val/')

        self.save_hyperparameters('lr', 'num_classes', 'freeze')

        self.example_input_array = torch.rand(1, 3, 224, 224)

    def forward(self, x):
        out, _, _ = self.encoder(x)
        probs = F.softmax(self.classifier(out[:, 0]), dim=-1)

        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch

        out, _, _ = self.encoder(x)
        out = self.classifier(out[:, 0])

        loss = self.criterion(out, y)
        metrics_train = self.metrics_train(F.softmax(out, dim=-1), y)
        self.log('loss', loss)
        self.log_dict(metrics_train)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)

        self.metrics_val(probs, y)

    def on_validation_epoch_end(self):
        metrics_out = self.metrics_val.compute()
        self.log_dict(metrics_out)

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return opt1


class MAEFinetuning(pl.callbacks.BaseFinetuning):
    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.encoder)
        self.make_trainable([pl_module.encoder.transformer.blocks[-1], pl_module.encoder.norm])

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        pass



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


def train_mae(loader, loader_test, root, classify = False, pretrained_path:str = None):
    name = str(uuid.uuid4()) + '_0.25_l1'

    callbacks = []

    if classify:
        name += '_classifier'

        if pretrained_path is not None:
            model = MAEClassifier.load_from_checkpoint(pretrained_path, strict=False, root=root, name=name, freeze=True)
            callbacks.append(MAEFinetuning())
        else:
            model = MAEClassifier(root=root, name=name)
    else:
        model = MAE(root=root, name=name)

    os.makedirs(os.path.join(root, name, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(root, name, 'cpts'), exist_ok=True)

    print('Name: {}'.format(name))

    logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(root, 'tb_log'), log_graph=False, version=name)
    callbacks.append(pl.callbacks.ModelSummary(max_depth=7))
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=os.path.join(root, name, 'cpts'), save_last=True, every_n_epochs=10))

    trainer = pl.Trainer(logger=logger, gpus=1, callbacks=callbacks,
                         max_epochs=1000, log_every_n_steps=10,
                         fast_dev_run=False)
    trainer.fit(model,
                train_dataloaders=loader,
                val_dataloaders=loader_test)


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
    out_dir = '/home/dsenkin/Desktop/scratch/vit'
    img_out_dir = os.path.join(out_dir, 'images_' + mode + ('_siren' if use_siren else ''))

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

        for batch_i, ((X_a, _), (X_b, _)) in tqdm(enumerate(zip(loader_a, loader_b)),
                                                  total=min(len(loader_a), len(loader_b))):
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
            d_ab_loss = 0.5 * (criterion_mse(d_score_valid, valid) + criterion_mse(d_score_fake, fake))
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

            g_ab_loss = criterion_mse(d_ab_score_fake, valid) + lmbda * criterion_mae(x_a_cyc, X_a)
            g_ba_loss = criterion_mse(d_ba_score_fake, valid) + lmbda * criterion_mae(x_b_cyc, X_b)
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

                print(
                    'Epoch: {}, gen a train loss: {}, gen b train loss: {}, disc a train loss {}, disc b train loss: {}'.format(
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
                        }, os.path.join(out_dir,
                                        ('cycle_vitgan_{}_' + mode + ('_siren' if use_siren else '') + '.pth').format(
                                            epoch)))


if __name__ == '__main__':
    device = 'cuda:0'

    mode = 'mae'
    root = sys.argv[1]

    if mode == 'mae' and len(sys.argv) > 2:
        classify = sys.argv[2]
        if classify and len(sys.argv) == 4:
            pretrained_path = sys.argv[3]

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

    batch_size = 64
    n_workers = 4
    # root = '/home/dsenkin/Desktop/scratch/monet2photo'

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
        print('Loading dataset...')
        dataset = STL10(os.path.join(root, 'stl10'), transform=transform, split='train', download=False)
        dataset_test = STL10(os.path.join(root, 'stl10'), transform=transform_test, split='test', download=False)

    if mode == 'cycle':
        loader_a = DataLoader(dataset_a, batch_size=batch_size, num_workers=n_workers,
                              persistent_workers=(n_workers > 0),
                              shuffle=True, pin_memory=True)

        loader_a_test = DataLoader(dataset_a_test, batch_size=batch_size, num_workers=0, persistent_workers=False,
                                   shuffle=True, pin_memory=True)

        loader_b = DataLoader(dataset_b, batch_size=batch_size, num_workers=n_workers,
                              persistent_workers=(n_workers > 0),
                              shuffle=True, pin_memory=True)

        loader_b_test = DataLoader(dataset_b_test, batch_size=batch_size, num_workers=0, persistent_workers=False,
                                   shuffle=True, pin_memory=True)
    else:
        print('Creating loaders...')
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, persistent_workers=(n_workers > 0),
                            shuffle=True, pin_memory=True)

        loader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=n_workers, persistent_workers=False,
                                 shuffle=False,
                                 pin_memory=True)

    if mode == 'teacher':
        train_teacher(loader, loader_test, device)
    elif mode == 'mae':
        print('Starting training...')
        train_mae(loader, loader_test, root, classify=classify, pretrained_path=pretrained_path)
    elif mode == 'vitgan':
        train_gan(loader, loader_test, device)
    elif mode == 'cycle':
        train_gan(loader_a, loader_b, loader_a_test, loader_b_test, device, mode='none', use_siren=True)
