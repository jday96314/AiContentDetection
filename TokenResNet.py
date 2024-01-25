# Modified from https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html

from typing import Any, Callable, List, Optional, Type, Union
from torchsummary import summary

import torch
import torch.nn as nn
from torch import Tensor
import tokenmonster
import numpy as np

def conv3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.dropout = nn.Dropout1d(p = 0.5)
        # self.dropout = nn.Dropout1d(p = 0.25)

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.dropout(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.dropout = nn.Dropout1d(p = 0.5)

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.dropout(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        token_embedding_dim_count: int = 64,
        base_width: int = 64,
        width_coefs: List[int] = [1, 2, 4, 8],
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        return_raw_features = False
    ) -> None:
        super().__init__()

        self.return_raw_features = return_raw_features

        self.embedding = nn.Embedding(vocab_size, token_embedding_dim_count)

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = base_width
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = base_width
        # self.conv1 = nn.Conv1d(token_embedding_dim_count, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv1d(token_embedding_dim_count, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base_width * width_coefs[0], layers[0])
        self.layer2 = self._make_layer(block, self.base_width * width_coefs[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.base_width * width_coefs[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, self.base_width * width_coefs[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # self.constrictor = nn.Sequential(
        #     # nn.AvgPool1d(kernel_size=16, stride=1),
        #     nn.Dropout1d(p = 0.1),
        #     nn.Conv1d(in_channels=self.base_width * width_coefs[3], out_channels=8, kernel_size=1, stride=1),
        #     nn.Tanh()
        # )
        
        self.avgpool = nn.AdaptiveAvgPool1d((1,))
        self.dropout = nn.Dropout(p = 0.1)
        # print('WARNING! Omitting output dropout in favor of constrictor. Highly experimental, possibly revert.')
        self.dropout = nn.Dropout(p = 0.0)
        self.fc = nn.Linear(self.base_width * width_coefs[3] * block.expansion, num_classes)
        # self.fc = nn.Linear(8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) # TODO: Maybe add this back in?

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.constrictor(x)

        if self.return_raw_features:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        # Transpose necessary because...
        #   - Raw embedding layer outputs have shape (batch size, sequence length, embedding dim count)
        #   - The conv layers expect input shape (batch size, embedding dim count, sequence length)
        token_embeddings = self.embedding(x).transpose(1, 2)

        return self._forward_impl(token_embeddings)

class Logits:
    def __init__(self, logits):
        self.logits = logits

class ResNet_HF(ResNet):
    def forward(self, input_ids, attention_mask):
        return Logits(super().forward(input_ids))

class BasicDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicDecoder, self).__init__()

        # Use ConvTranspose1d layers to increase length by a factor of 2^3 = 8
        self.deconv1 = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.LeakyReLU()
        
        self.deconv2 = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels // 2)
        self.relu2 = nn.LeakyReLU()
        
        self.deconv3 = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(in_channels // 2)
        self.relu3 = nn.LeakyReLU()
        
        self.classifier = nn.Conv1d(in_channels // 2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.classifier(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

if __name__ == '__main__':
    # ResNet-18: block=BasicBlock, layers=[2, 2, 2, 2]
    # ResNet-34: block=BasicBlock, layers=[3, 4, 6, 3]
    # ResNet-50: block=Bottleneck, layers=[3, 4, 6, 3]
    # ResNet-101: block=Bottleneck, layers=[3, 4, 23, 3]
    
    VOCAB_SIZE = 24000
    encoder = ResNet(
        # One added for padding.
        vocab_size=(VOCAB_SIZE + 1),
        block=BasicBlock, 
        layers=[2, 2, 2, 2],
        width_coefs=[1, 1, 2, 2],
        num_classes=2, 
        token_embedding_dim_count = 128,
        base_width = 128,
        return_raw_features=True)
    decoder = BasicDecoder(in_channels = 256, out_channels = (VOCAB_SIZE + 1))
    auto_encodder = AutoEncoder(encoder, decoder).cuda()

    tokenizer = tokenmonster.load('Tokenizers/tokenmonster/vocabs/english-24000-strict-v1.vocab')
    token_ids = np.array(tokenizer.tokenize([
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis sed facilisis urna, et dignissim magna. Nullam imperdiet placerat augue, eleifend aliquam elit pharetra ut. Aenean eu tortor sollicitudin, vehicula erat ac, blandit justo. Cras dapibus diam vitae nisi placerat mollis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque convallis arcu at nibh volutpat condimentum. Ut sed ante eu est finibus dapibus. Donec leo neque, tempus eu rutrum ut, vehicula non enim. Vestibulum eleifend venenatis auctor. Duis massa urna, interdum sit amet luctus ac, elementum et dui. Sed nec lacinia dui. Duis placerat rutrum massa, in sodales lectus facilisis non. Sed porta arcu at congue vehicula. Integer sit amet tempor massa. Etiam porta dui ac mattis tristique. Donec hendrerit quam nec commodo blandit. Quisque sed placerat eros. Nam non tellus ligula. Maecenas malesuada facilisis metus, id iaculis arcu efficitur id. Integer ac neque eros. Cras quis sollicitudin lectus. Sed viverra pharetra placerat. Nunc sit amet semper elit, non cursus massa. Quisque pretium suscipit ipsum, ac tincidunt dui faucibus quis. Integer quis dui fringilla, congue dui et, fermentum felis. Quisque ultricies eleifend gravida. Donec non est sed nisl consequat rutrum nec at est. Integer in laoreet risus. Quisque vitae mi iaculis, feugiat libero non, pellentesque velit. Etiam rutrum arcu a hendrerit gravida. Donec euismod mauris ipsum, at dictum dui varius ac. Nam in tincidunt dui, non sollicitudin nisi. Aenean quis nibh non ex consequat aliquet vel vel velit. Mauris vel nisi eget urna malesuada aliquet. In ultricies accumsan quam vitae posuere. Aliquam dapibus tortor sem, vitae ultricies lacus aliquam sed. Duis porttitor convallis nibh, fringilla elementum ligula eleifend ac. Aliquam ac feugiat tortor, sed dictum nulla. Nunc iaculis auctor nisi. Mauris neque augue, auctor sit amet egestas nec, aliquam ac sapien. Mauris convallis nisi nulla, nec porta massa aliquam at. Quisque quam sem, porttitor at tristique placerat, dignissim id lectus. Ut lobortis placerat erat, quis laoreet mi pellentesque nec. Fusce eget ultrices libero. Suspendisse aliquam condimentum magna, sit amet viverra mi elementum ac. Integer convallis erat dapibus, varius odio eget, gravida ante. .',
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis sed facilisis urna, et dignissim magna. Nullam imperdiet placerat augue, eleifend aliquam elit pharetra ut. Aenean eu tortor sollicitudin, vehicula erat ac, blandit justo. Cras dapibus diam vitae nisi placerat mollis. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque convallis arcu at nibh volutpat condimentum. Ut sed ante eu est finibus dapibus. Donec leo neque, tempus eu rutrum ut, vehicula non enim. Vestibulum eleifend venenatis auctor. Duis massa urna, interdum sit amet luctus ac, elementum et dui. Sed nec lacinia dui. Duis placerat rutrum massa, in sodales lectus facilisis non. Sed porta arcu at congue vehicula. Integer sit amet tempor massa. Etiam porta dui ac mattis tristique. Donec hendrerit quam nec commodo blandit. Quisque sed placerat eros. Nam non tellus ligula. Maecenas malesuada facilisis metus, id iaculis arcu efficitur id. Integer ac neque eros. Cras quis sollicitudin lectus. Sed viverra pharetra placerat. Nunc sit amet semper elit, non cursus massa. Quisque pretium suscipit ipsum, ac tincidunt dui faucibus quis. Integer quis dui fringilla, congue dui et, fermentum felis. Quisque ultricies eleifend gravida. Donec non est sed nisl consequat rutrum nec at est. Integer in laoreet risus. Quisque vitae mi iaculis, feugiat libero non, pellentesque velit. Etiam rutrum arcu a hendrerit gravida. Donec euismod mauris ipsum, at dictum dui varius ac. Nam in tincidunt dui, non sollicitudin nisi. Aenean quis nibh non ex consequat aliquet vel vel velit. Mauris vel nisi eget urna malesuada aliquet. In ultricies accumsan quam vitae posuere. Aliquam dapibus tortor sem, vitae ultricies lacus aliquam sed. Duis porttitor convallis nibh, fringilla elementum ligula eleifend ac. Aliquam ac feugiat tortor, sed dictum nulla. Nunc iaculis auctor nisi. Mauris neque augue, auctor sit amet egestas nec, aliquam ac sapien. Mauris convallis nisi nulla, nec porta massa aliquam at. Quisque quam sem, porttitor at tristique placerat, dignissim id lectus. Ut lobortis placerat erat, quis laoreet mi pellentesque nec. Fusce eget ultrices libero. Suspendisse aliquam condimentum magna, sit amet viverra mi elementum ac. Integer convallis erat dapibus, varius odio eget, gravida ante. .'
    ]))

    example_inputs = torch.tensor(np.array(token_ids[: , :512 + 256], dtype = np.int32)).cuda()

    decoded_tokens = auto_encodder(example_inputs)

    print(example_inputs.shape, decoded_tokens.shape)
    
