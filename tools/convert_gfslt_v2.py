"""Convert GFSLT-VLP Stage1 checkpoint to SignGraph conv2d weights (v2).

Tailored to the actual checkpoint produced by retraining GFSLT-VLP stage1 with
the official SignGraph backbone (this repo's modules/). Compared to the legacy
script in convert_gfslt_resnet_to_signgraph.py, this v2 is:

- minimal: no 2D->3D unsqueeze, no per-stage filter, no full-model-prefix flag
- strict about source format: expects ckpt['model'] with the
  model_images.model.conv_2d.resnet.* prefix
- BN running stats are preserved
- fc.* is dropped (SignGraph replaces it with Identity)

Usage:
    python tools/convert_gfslt_v2.py \
        --src pretrained/best_checkpoint.pth \
        --dst pretrained/gfslt_resnet18_full.pth
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


SRC_PREFIX = 'model_images.model.conv_2d.resnet.'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='GFSLT-VLP stage1 checkpoint path.')
    p.add_argument('--dst', required=True, help='Output path for SignGraph conv2d weights.')
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.src, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    converted = OrderedDict()
    skipped_fc = 0
    for k, v in state.items():
        if not k.startswith(SRC_PREFIX):
            continue
        clean = k[len(SRC_PREFIX):]
        if clean.startswith('fc.'):
            skipped_fc += 1
            continue
        converted[clean] = v

    if not converted:
        raise RuntimeError(f'No keys found under prefix {SRC_PREFIX!r}.')

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': converted}, dst)

    print(f'[convert_gfslt_v2] kept   : {len(converted)} tensors')
    print(f'[convert_gfslt_v2] dropped fc.*: {skipped_fc}')
    print(f'[convert_gfslt_v2] saved  : {dst}')


if __name__ == '__main__':
    main()
