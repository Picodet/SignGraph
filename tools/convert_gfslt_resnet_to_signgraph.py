"""Convert GFSLT-VLP Stage1 ResNet weights to SignGraph conv2d weights.

This script intentionally extracts only the ResNet part from GFSLT-VLP Stage1:

    model_images.model.conv_2d.resnet.*

It does not import or depend on GFSLT model code. The output can be loaded by
SLRModel.load_gfslt_resnet_pretrain().

Example:
    python tools/convert_gfslt_resnet_to_signgraph.py \
        --src /path/to/gfslt_stage1_checkpoint.pth \
        --dst pretrained/gfslt_resnet18_conv2d_only.pth \
        --stages stem layer1 layer2 layer3 layer4
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


GFSLT_RESNET_PREFIXES = (
    'model_images.model.conv_2d.resnet.',
    'module.model_images.model.conv_2d.resnet.',
)

VALID_STAGES = ('stem', 'layer1', 'layer2', 'layer3', 'layer4')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GFSLT-VLP Stage1 ResNet18 weights to SignGraph conv2d weights.'
    )
    parser.add_argument('--src', required=True, help='Path to GFSLT-VLP Stage1 checkpoint.')
    parser.add_argument('--dst', required=True, help='Path to save converted SignGraph weights.')
    parser.add_argument(
        '--stages',
        nargs='+',
        default=list(VALID_STAGES),
        choices=VALID_STAGES,
        help='Which ResNet stages to convert.'
    )
    parser.add_argument(
        '--full-model-prefix',
        action='store_true',
        help="Save keys with 'conv2d.' prefix for loading into a full SLRModel state dict."
    )
    return parser.parse_args()


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def strip_gfslt_resnet_prefix(key):
    for prefix in GFSLT_RESNET_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def stage_allowed(src_key, stages):
    if src_key.startswith('conv1.') or src_key.startswith('bn1.'):
        return 'stem' in stages

    for stage in ('layer1', 'layer2', 'layer3', 'layer4'):
        if src_key.startswith(stage + '.'):
            return stage in stages

    return False


def should_unsqueeze_to_3d(src_key, tensor):
    if tensor.ndim != 4:
        return False
    if src_key == 'conv1.weight':
        return True
    if '.conv' in src_key and src_key.endswith('.weight'):
        return True
    if 'downsample.0.weight' in src_key:
        return True
    return False


def main():
    args = parse_args()

    checkpoint = torch.load(args.src, map_location='cpu')
    state = get_state_dict(checkpoint)

    converted = OrderedDict()
    seen_stage_count = {stage: 0 for stage in VALID_STAGES}
    skipped = []

    for key, value in state.items():
        src_key = strip_gfslt_resnet_prefix(key)
        if src_key is None:
            continue

        if src_key.startswith('fc.'):
            continue

        if not stage_allowed(src_key, args.stages):
            skipped.append(src_key)
            continue

        if should_unsqueeze_to_3d(src_key, value):
            value = value.unsqueeze(2)

        dst_key = f'conv2d.{src_key}' if args.full_model_prefix else src_key
        converted[dst_key] = value

        if src_key.startswith('conv1.') or src_key.startswith('bn1.'):
            seen_stage_count['stem'] += 1
        else:
            for stage in ('layer1', 'layer2', 'layer3', 'layer4'):
                if src_key.startswith(stage + '.'):
                    seen_stage_count[stage] += 1
                    break

    if not converted:
        raise RuntimeError(
            'No GFSLT ResNet weights were converted. Check whether the checkpoint contains '
            'model_images.model.conv_2d.resnet.* keys.'
        )

    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'state_dict': converted,
        'meta': {
            'source': args.src,
            'stages': args.stages,
            'format': 'signgraph_conv2d',
            'full_model_prefix': args.full_model_prefix,
            'converted_tensors': len(converted),
            'stage_tensor_count': seen_stage_count,
            'note': 'GFSLT-VLP 2D ResNet weights converted to SignGraph Conv3d weights by unsqueezing temporal dim.',
        }
    }
    torch.save(output, dst_path)

    print(f'[OK] saved to: {dst_path}')
    print(f'[OK] converted tensors: {len(converted)}')
    print(f'[OK] stages: {args.stages}')
    print(f'[OK] full_model_prefix: {args.full_model_prefix}')
    print(f'[OK] stage tensor count: {seen_stage_count}')
    if skipped:
        print(f'[OK] skipped tensors by stage filter: {len(skipped)}')


if __name__ == '__main__':
    main()
