import sys
import os
import re
from requests.utils import urlparse

import torch
import torch.nn as nn
from torch.utils.model_zoo import _download_url_to_file as _download_url_to_file
from MobileNetV2 import MobileNetV2  # https://github.com/tonylins/pytorch-mobilenet-v2

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
mbnetv2_url = 'https://docs.google.com/uc?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR&export=download'
mbnetv2_hash = 'ecbe2b568c8602549fa9e1d5833c63848f490a48d92e5d224d1eb2063e152cf8'
mbnetv2_fname = 'mobilenet_v2.pth.tar'


def load_url(url, model_dir=None, map_location=None, progress=True, hash_prefix=None, filename=None):
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    parts = urlparse(url)
    if filename is None: filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        if hash_prefix is None: hash_prefix = HASH_REGEX.search(filename).group(1)
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return torch.load(cached_file, map_location=map_location)


def mbnetv2(pretrained=False, **kwargs):
    model = MobileNetV2(width_mult=1, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(mbnetv2_url, hash_prefix=mbnetv2_hash, filename=mbnetv2_fname))
    return model


def _mbnetv2_split(m: nn.Module): return (m[0][0][6], m[0][0][12], m[1])