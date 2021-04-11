'''
Modified from https://raw.githubusercontent.com/jeonsworld/ViT-pytorch/main/models/modeling.py

MIT License

Copyright (c) 2020 jeonsworld

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import sys
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional

cifar10_pretrained_weight_urls = {
    'vit_b16': '',
    'vit_b32': '',
    'vit_l16': '',
    'vit_l32': '',
    'vit_h14': '',
}

cifar100_pretrained_weight_urls = {
    'vit_b16': '',
    'vit_b32': '',
    'vit_l16': '',
    'vit_l32': '',
    'vit_h14': '',
}


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=10, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

        if zero_head:
            with torch.no_grad():
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        return logits


class TestConfig:
    patches = dict(size=(16, 16))
    hidden_size = 1
    transformer = dict(
        mlp_dim=1,
        num_heads=1,
        num_layers=1,
        attention_dropout_rate=0.0,
        dropout_rate=0.1
    )
    classifier = 'token'
    representation_size = None


class VitB16Config:
    patches = dict(size=(16, 16))
    hidden_size = 768
    transformer = dict(
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.1
    )
    classifier = 'token'
    representation_size = None


class VitB32Config(VitB16Config):
    patches = dict(size=(32, 32))


class VitL16Config:
    patches = dict(size=(16, 16))
    hidden_size = 1024
    transformer = dict(
        mlp_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1
    )
    classifier = 'token'
    representation_size = None


class VitL32Config(VitL16Config):
    patches = dict(size=(32, 32))


class VitH14Config:
    patches = dict(size=(14, 14))
    hidden_size = 1280
    transformer = dict(
        mlp_dim=5120,
        num_heads=16,
        num_layers=32,
        attention_dropout_rate=0.0,
        dropout_rate=0.1
    )
    classifier = 'token'
    representation_size = None


def _vit(
    arch: str,
    config: Any,
    model_urls: Dict[str, str],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> VisionTransformer:
    model = VisionTransformer(config=config, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def cifar10_vit_b16(*args, **kwargs) -> VisionTransformer: pass
def cifar10_vit_b32(*args, **kwargs) -> VisionTransformer: pass
def cifar10_vit_l16(*args, **kwargs) -> VisionTransformer: pass
def cifar10_vit_l32(*args, **kwargs) -> VisionTransformer: pass
def cifar10_vit_h14(*args, **kwargs) -> VisionTransformer: pass


def cifar100_vit_b16(*args, **kwargs) -> VisionTransformer: pass
def cifar100_vit_b32(*args, **kwargs) -> VisionTransformer: pass
def cifar100_vit_l16(*args, **kwargs) -> VisionTransformer: pass
def cifar100_vit_l32(*args, **kwargs) -> VisionTransformer: pass
def cifar100_vit_h14(*args, **kwargs) -> VisionTransformer: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for config, model_name in zip([VitB16Config, VitB32Config, VitL16Config, VitL32Config, VitH14Config],
                                  ["vit_b16", "vit_b32", "vit_l16", "vit_l32", "vit_h14"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_vit,
                    arch=model_name,
                    config=config,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )
