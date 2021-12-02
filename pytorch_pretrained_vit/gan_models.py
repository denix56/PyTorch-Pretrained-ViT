import numpy as np
from einops import rearrange, repeat
import opt_einsum as oe

import torch
import torch.nn as nn
import torch.nn.functional as F


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class SLN(nn.Module):
    """
    Self-modulated LayerNorm
    """
    def __init__(self, num_features):
        super(SLN, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        # self.gamma = nn.Parameter(torch.FloatTensor(1, 1, 1))
        # self.beta = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")
        self.beta = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")

    def forward(self, hl, w):
        return self.gamma * w * self.ln(hl) + self.beta * w


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".
    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """
    def __init__(self, dim, num_heads = 4, dim_head = None, discriminator = False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        self.weight_dim = self.num_heads * self.dim_head
        self.to_qkv = nn.Linear(dim, self.weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5
        self.discriminator = discriminator
        self.w_out = nn.Linear(self.weight_dim, dim, bias = True)

        if discriminator:
            s = torch.linalg.svdvals(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3

        if self.discriminator:
            s = torch.linalg.svdvals(self.to_qkv.weight)
            self.to_qkv.weight = torch.nn.Parameter(self.to_qkv.weight * self.init_spect_norm / torch.max(s))

        # Generate the q, k, v vectors
        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k = 3, h = self.num_heads))

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.discriminator:
            attn = torch.cdist(q, k, p = 2)
        else:
            attn = oe.contract("... i d, ... j d -> ... i j", q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim = -1)
        result = oe.contract("... i j, ... j d -> ... i d", scale_attn_score, v)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.w_out(result)


class DEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(DEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head, discriminator = True)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attn(x1))
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class GEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(GEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, x):
        x = self.dropout(self.attn(self.norm1(x))) + x
        x = self.mlp(self.norm2(x)) + x
        return x


class RGBBlock(nn.Module):
    def __init__(self, dim, in_patches, out_patches, out_channels):
        super().__init__()

        self.norm = SLN(dim)
        self.patch_scale = nn.Linear(in_patches, out_patches)

        self.w_out = nn.Sequential(
                SineLayer(dim, dim * 2, is_first=True, omega_0=30.),
                SineLayer(dim * 2, out_patches * out_channels, is_first=False, omega_0=30)
            )

    def forward(self, x, hl):
        x = self.norm(x, hl)
        x = self.patch_scale(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.w_out(x)
        return x


class GTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0,
    ):
        super(GTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

        # self.rgb_blocks = []
        # for i in range(blocks):
        #     self.rgb_blocks.append(RGBBlock(dim, initialize_size*8, 2**(i+1)*8, 3))
        # self.rgb_blocks = nn.ModuleList(self.rgb_blocks)



    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(GEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.ModuleList(layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(DTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)



    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(DEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.ModuleList(layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """
    def __init__(self, in_features, out_features, bias = True,is_first = False, omega_0 = 30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x, mask=None):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        if mask is None:
            return x + self.pos_embedding
        else:
            return x + self.pos_embedding.expand(x.shape[0], -1, -1)[mask].view(*x.shape)


class Generator(nn.Module):
    def __init__(self,
        image_size = 224,
        patches = 16,
        in_channels = 3,
        dim = 768,
        blocks = 6,
        num_heads = 6,
        dim_head = None,
        dropout = 0,
        out_channels = 3,
    ):
        super(Generator, self).__init__()
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw

        self.image_size = (h, w)
        self.n_patches = (gh, gw)

        self.dim = dim
        self.blocks = blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.out_channels = out_channels

        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(fh, fw), stride=(fh, fw))

        #self.pos_emb1D = nn.Parameter(torch.randn(self.initialize_size * 8, dim))
        self.pos_emb1D = PositionalEmbedding1D(seq_len, dim)
        #self.mlp = nn.Linear(1024, (self.initialize_size * 8) * self.dim)
        self.Transformer_Encoder = GTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)

        # Implicit Neural Representation
        # self.w_out = nn.Sequential(
        #     SineLayer(dim, dim * 2, is_first = True, omega_0 = 30.),
        #     SineLayer(dim * 2, fh * fw * self.out_channels, is_first = False, omega_0 = 30)
        # )
        self.w_out = nn.ConvTranspose2d(dim, self.out_channels, kernel_size=(fh, fw), stride=(fh, fw))
        self.sln_norm = nn.LayerNorm(self.dim, eps=1e-6)

    def forward(self, x):
        #x = self.mlp(noise).view(-1, self.initialize_size * 8, self.dim)

        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d

        x = self.pos_emb1D(x)
        x = self.Transformer_Encoder(x)
        # x = self.sln_norm(x)
        x = x.view(-1, *self.n_patches, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.w_out(x)  # Replace to siren
        result = x.view(x.shape[0], 3, *self.image_size)
        return result


class Discriminator(nn.Module):
    def __init__(self,
        image_size=224,
        patches=16,
        in_channels = 3,
        extend_size = 2,
        dim = 384,
        blocks = 6,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Discriminator, self).__init__()

        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw
        assert fh == fw
        patch_size = fh

        self.patch_size = patch_size + 2 * extend_size
        self.token_dim = in_channels * self.patch_size**2
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(self.token_dim + 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

        self.Transformer_Encoder = DTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)


    def forward(self, img):
        # Generate overlapping image patches
        stride_h = (img.shape[2] - self.patch_size) // 8 + 1
        stride_w = (img.shape[3] - self.patch_size) // 8 + 1
        img_patches = img.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        img_patches = img_patches.contiguous().view(
            img_patches.shape[0], img_patches.shape[2] * img_patches.shape[3], img_patches.shape[1] * img_patches.shape[4] * img_patches.shape[5]
        )
        img_patches = self.project_patches(img_patches)
        batch_size, tokens, _ = img_patches.shape

        # Prepend the classifier token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b = batch_size)
        img_patches = torch.cat((cls_token, img_patches), dim = 1)

        # Plus the positional embedding
        img_patches = img_patches + self.pos_emb1D[: tokens + 1, :]
        img_patches = self.emb_dropout(img_patches)

        result = self.Transformer_Encoder(img_patches)
        logits = self.mlp_head(result[:, 0, :])
        #logits = nn.Sigmoid()(logits)
        return logits


def test_both():
    B, dim = 10, 1024
    G = Generator(initialize_size = 8, dropout = 0.1)
    noise = torch.FloatTensor(np.random.normal(0, 1, (B, dim)))
    fake_img = G(noise)
    D = Discriminator(patch_size = 8, dropout = 0.1)
    D_logits = D(fake_img)
    print(D_logits)
    print(f"Max: {torch.max(D_logits)}, Min: {torch.min(D_logits)}")