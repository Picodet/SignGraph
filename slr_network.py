import pdb
import copy
from collections import OrderedDict

import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

    def load_gfslt_resnet_pretrain(self, weight_path, strict=False):
        """Load converted GFSLT-VLP Stage1 ResNet weights into SignGraph conv2d.

        Expected checkpoint formats:
        1. {'state_dict': {'conv1.weight': ..., 'layer1.0.conv1.weight': ...}}
        2. {'state_dict': {'conv2d.conv1.weight': ..., 'conv2d.layer1.0.conv1.weight': ...}}
        3. A raw state dict in either of the above key formats.

        This function intentionally touches only self.conv2d. It does not load
        GFSLT TemporalConv, MBart, text encoder, visual trans_encoder, or any
        SignGraph classifier / graph module weights.
        """
        checkpoint = torch.load(weight_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        conv2d_ref = self.conv2d.state_dict()
        conv2d_state = OrderedDict()
        incompatible = []

        for key, value in state_dict.items():
            clean_key = key.replace('module.', '')
            if clean_key.startswith('conv2d.'):
                clean_key = clean_key[len('conv2d.'):]

            if clean_key not in conv2d_ref:
                continue

            if conv2d_ref[clean_key].shape != value.shape:
                incompatible.append((clean_key, tuple(value.shape), tuple(conv2d_ref[clean_key].shape)))
                continue

            conv2d_state[clean_key] = value

        ret = self.conv2d.load_state_dict(conv2d_state, strict=strict)

        print(f"[GFSLT-ResNet] Loaded {len(conv2d_state)} tensors into SignGraph conv2d from: {weight_path}")
        if incompatible:
            print('[GFSLT-ResNet] Skipped incompatible tensors:')
            for name, src_shape, dst_shape in incompatible[:20]:
                print(f"  {name}: checkpoint {src_shape} vs model {dst_shape}")
            if len(incompatible) > 20:
                print(f"  ... and {len(incompatible) - 20} more")
        print('[GFSLT-ResNet] Missing keys:', ret.missing_keys)
        print('[GFSLT-ResNet] Unexpected keys:', ret.unexpected_keys)
        return ret

    def freeze_gfslt_resnet_stages(self, stages):
        """Freeze selected SignGraph conv2d stages after loading GFSLT ResNet weights."""
        if stages is None:
            return

        stage_to_modules = {
            'stem': [self.conv2d.conv1, self.conv2d.bn1],
            'layer1': [self.conv2d.layer1],
            'layer2': [self.conv2d.layer2],
            'layer3': [self.conv2d.layer3],
            'layer4': [self.conv2d.layer4],
        }

        frozen = []
        for stage in stages:
            if stage not in stage_to_modules:
                print(f"[GFSLT-ResNet] Unknown freeze stage skipped: {stage}")
                continue
            for module in stage_to_modules[stage]:
                for param in module.parameters():
                    param.requires_grad = False
            frozen.append(stage)

        if frozen:
            print(f"[GFSLT-ResNet] Frozen conv2d stages: {frozen}")

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):

        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
        else:
            framewise = x
        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len'].cpu()
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        return {
            "framewise_features": framewise,
            "visual_features": x,
            "temproal_features": tm_outputs['predictions'],
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
