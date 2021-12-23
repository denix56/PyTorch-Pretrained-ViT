"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer
from .utils import load_pretrained_weights, as_tuple
from .configs import PRETRAINED_MODELS


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x, mask = None):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        if mask is None:
            return x + self.pos_embedding
        else:
            return x + self.pos_embedding.expand(x.shape[0], -1, -1)[mask].view(*x.shape)


class MaskTransLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct the normalization for each patch
        """
        super(MaskTransLayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x[:, :].mean(-1, keepdim=True)
        s = (x[:, :] - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta



class ViT(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3,
        distilled: bool = False,
        use_mask_token: bool = False,
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        mask_gen = False
    ):
        super().__init__()

        # Configuration
        if name is None or not pretrained:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            # assert not resize_positional_embedding, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.image_size = image_size                

        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw
        self.gh = gh
        self.gw = gw
        self.distilled = distilled
        self.use_mask_token = use_mask_token
        self.mask_gen = mask_gen

        # Patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        # Class token
        if classifier == 'token' and not self.mask_gen:
            self.has_cls_token = True
            if not self.use_mask_token:
                self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        else:
            self.has_cls_token = False

        if distilled and not self.mask_gen:
            if not self.use_mask_token:
                self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1

        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.patch_norm = MaskTransLayerNorm(dim)
            self.unconv = nn.ConvTranspose2d(dim, in_channels, kernel_size=(fh, fw), stride=(fh, fw))
        if mask_gen:
            self.patch_norm = nn.LayerNorm(dim, eps=1e-6)
            self.unconv = nn.Linear(dim, 1)
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()

        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)

        if not mask_gen:
            # Representation layer
            if representation_size and load_repr_layer:
                self.pre_logits = nn.Linear(dim, representation_size)
                pre_logits_size = representation_size
            else:
                pre_logits_size = dim

            # Classifier head
            self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
            self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()
        
        # Load pretrained model
        if pretrained:
            pretrained_num_channels = 3
            pretrained_num_classes = PRETRAINED_MODELS[name]['num_classes']
            pretrained_image_size = PRETRAINED_MODELS[name]['image_size']
            load_pretrained_weights(
                self, name, 
                load_first_conv=(in_channels == pretrained_num_channels),
                load_fc=(num_classes == pretrained_num_classes),
                load_repr_layer=load_repr_layer,
                resize_positional_embedding=(image_size != pretrained_image_size),
            )

        self.mask_gen = mask_gen
        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        if hasattr(self, 'fc'):
            nn.init.constant_(self.fc.weight, 0)
            nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        if hasattr(self, 'class_token'):
            nn.init.constant_(self.class_token, 0)
        if hasattr(self, 'dist_token'):
            nn.init.constant_(self.dist_token, 0)
        if hasattr(self, 'mask_token'):
            nn.init.constant_(self.mask_token, 0)


    def forward(self, x, mask_rate:float = 0.0, mask=None, lengths=None):
        """Breaks image into patches, applies transformer, applies MLP head.

        Args:
            x (tensor): `b,c,fh,fw`
        """

        if not self.use_mask_token or self.mask_gen:
            b, c, fh, fw = x.shape
            x = self.patch_embedding(x)  # b,d,gh,gw
            x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
            if not self.mask_gen:
                if mask_rate > 0.0:
                    mask_indices = torch.multinomial(torch.full((1, x.shape[1]), 1/x.shape[1], device=x.device).expand(b, -1),
                                                     int((1-mask_rate)*x.shape[1]))
                    mask = torch.zeros(*x.shape[:2], dtype=bool, device=x.device)
                    mask.scatter_(1, mask_indices, 1)
                    x = x[mask].view(b, -1, x.shape[2])
                elif mask is not None:
                    x = x*mask
                    #mask_b = mask.squeeze(-1) > 0
                    #x = [x[i, mask_b[i]] for i in range(x.shape[0])]
                    #max_len = max(x, key=lambda x: x.shape[0])

                    #print(x.shape)
                    #x = x.view(b, -1, x.shape[-1])
                else:
                    raise NotImplementedError()

                add_patches = 0
                if hasattr(self, 'class_token'):
                    add_patches += 1
                    if not self.distilled:
                        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
                    else:
                        add_patches += 1
                        x = torch.cat((self.class_token.expand(b, -1, -1), self.dist_token.expand(b, -1, -1), x),
                                      dim=1)  # b,gh*gw+2,d
                if mask is not None and add_patches > 0:
                    mask = torch.cat((torch.ones(b, add_patches, 1, device=mask.device), mask), dim=1)
        else:
            assert mask is not None
            aux_patches = 0
            if self.has_cls_token:
                aux_patches += 1
            if self.distilled:
                aux_patches += 1

            x_new = self.mask_token.repeat(x.shape[0], self.gh*self.gw + aux_patches, 1)
            x_new = x_new*(1-mask)
            x = torch.cat([t[:length] for t, length in zip(x, lengths)], dim=0)
            mask_b = mask.squeeze(-1) > 0
            x_new[mask_b] = x
            x = x_new

        if hasattr(self, 'positional_embedding'):
            x = self.positional_embedding(x)  # b,gh*gw+1(2),d

        if mask is not None and not self.use_mask_token:
            mask_b = mask.squeeze(-1) > 0
            x = [x[i, mask_b[i]] for i in range(x.shape[0])]
            lengths = torch.tensor([t.shape[0] for t in x])
            x = nn.utils.rnn.pad_sequence(x, batch_first=True)

        x = self.transformer(x)  # b,gh*gw+1(2),d

        if self.use_mask_token:
            x = self.patch_norm(x[:, aux_patches:])
            x = x.view(-1, self.gh, self.gw, x.shape[-1]).permute(0, 3, 1, 2)
            x = self.unconv(x)
        elif self.mask_gen:
            x = self.patch_norm(x)
            x = x.view(-1, self.gh*self.gw, x.shape[-1])
            x = self.unconv(x)
            x = torch.sigmoid(x)

        if mask is None:
            if hasattr(self, 'pre_logits'):
                x = self.pre_logits(x)
                x = torch.tanh(x)
            if hasattr(self, 'fc'):
                x = self.norm(x)[:, 0]  # b,d
                x = self.fc(x)  # b,num_classes
        elif not self.use_mask_token:
            x = self.norm(x)
        return x, mask, lengths



