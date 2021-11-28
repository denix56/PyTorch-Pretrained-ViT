from pytorch_pretrained_vit.model import ViT
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.transforms import RandAugment, Compose, Resize, ToTensor, Normalize
from torchvision import models
from tqdm import trange, tqdm
from torch import nn

if __name__ == '__main__':
    device = 'cuda:0'

    transform = Compose([Resize(224),
                         RandAugment(),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
                         ])
    dataset = STL10('stl10', transform=transform, split='train', download=True)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, persistent_workers=True, shuffle=True, pin_memory=True)

    transform_test = Compose([Resize(224),
                         ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
                         ])

    dataset_test = STL10('stl10', transform=transform_test, split='test', download=True)
    loader_test = DataLoader(dataset_test, batch_size=64, num_workers=4, persistent_workers=False, shuffle=False,
                             pin_memory=True)

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


