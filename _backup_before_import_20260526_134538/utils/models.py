from pathlib import Path

import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    if len(size) != 4:
        raise ValueError('Expected feature tensor with shape [N, C, H, W]')

    batch_size, channels = size[:2]
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x):
        return self.net(x)


class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0, ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True),
        )

        vgg_path = Path(vgg_path)
        if not vgg_path.exists():
            raise FileNotFoundError(
                f'VGG checkpoint not found: {vgg_path}. '
                'Download vgg_normalised.pth or pass --vgg with the correct path.'
            )

        state_dict = torch.load(vgg_path, map_location='cpu')
        self.net.load_state_dict(state_dict, strict=False)

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        features = []
        for index, layer in enumerate(self.net):
            x = layer(x)
            if index in {3, 10, 17, 30}:
                features.append(x)
        return features
