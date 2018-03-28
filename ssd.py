import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes, configs, network_name):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        box_configs = configs[network_name]['box_configs']
        self.priorbox = PriorBox(box_configs)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 1166  # don't think this is used
        self.config = box_configs

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        norm_output_conv4_3 = self.L2Norm(x)
        sources.append(norm_output_conv4_3)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        print('total sources', len(sources))
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # print('before confidence layer:', x.shape)
            # print('after confidence layer:', c(x).shape)
            # print('loc0 ----- ', type(loc[0]))

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # print('-----loc', loc.shape)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                    self.num_classes)),                         # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            # print('locations:')
            # print(loc.view(loc.size(0), -1, 4))
            # print('confs:')
            # print(conf.view(conf.size(0), -1, self.num_classes))
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),  # why no softmax here? -> goes into multibox loss
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, layers5_7, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    if layers5_7:
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes, vgg_source=[21, -2]):
    loc_layers = []
    conf_layers = []
    print('vgg source:', vgg_source)

    # Not sure why he uses 21 here when the actual source comes from 23 (end of 4_3 rather than mid)
    # but it should give the same size. -- It's because ReLU has no out channels
    # vgg_source = [21, -2]  # full base network
    # vgg_source = [-2]  # trunctated
    for k, v in enumerate(vgg_source):
        print('vgg source output channels', vgg[v].out_channels)
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1, stride=1)]  # changed from s1 to s2 when using half number of prior boxes
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1, stride=1)]  # changed from s1 to s2...
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1, stride=1)]  # changed from s1 to s2...
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1, stride=1)]  # changed from s1 to s2...

    print('number of extra layers (each is really 2):', len(extra_layers))
    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(phase, configs, network_name='300', num_classes=21, square_boxes=False):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return

    print('number of classes =', num_classes)

    base = configs[network_name]['base']
    extras = configs[network_name]['extras']
    mbox = configs[network_name]['mbox']
    size_last_base_layer = configs[network_name]['final_base_layer_dim']
    final_vgg_layers = configs[network_name]['layers5to7']
    if square_boxes:
        configs[network_name]['box_configs']['square_only'] = True

    if square_boxes:
        mbox = [2]*len(mbox)
        print('number of boxes per position for each source layer:', mbox)

    base_, extras_, head_ = multibox(vgg(base, 3, layers5_7=final_vgg_layers),
                                     add_extras(extras, size_last_base_layer),  # 3 and 1024 are input channels
                                     mbox, num_classes,
                                     configs[network_name]['vgg_source'])
    return SSD(phase, base_, extras_, head_, num_classes, configs, network_name)
