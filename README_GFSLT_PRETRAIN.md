# 分支说明：feature/gfslt-pretrain-no-bn-stats

## 背景

本分支记录了一次**初步尝试**：将 GFSLT-VLP（VLP-SLT）Stage1 预训练得到的 ResNet18 视觉编码器权重，
直接迁移到 SignGraph 的 2D backbone（`conv_2d`）中，希望借助 VLP 多模态预训练带来的视觉表征
来提升 CSLR 任务的精度。

**结论：效果不及预期。** 该分支作为消融与失败记录保留，不计划合并到 `main`。

## 做了什么

### 1. 权重转换工具
[tools/convert_gfslt_resnet_to_signgraph.py](tools/convert_gfslt_resnet_to_signgraph.py)

从 GFSLT-VLP Stage1 checkpoint 中**仅抽取** ResNet 部分的参数：

```
model_images.model.conv_2d.resnet.*  →  conv2d.resnet.*
```

支持按 stage（stem / layer1-4）选择性转换，并默认 `--skip-bn-stats=True`：
跳过 BN 的 `running_mean / running_var / num_batches_tracked`，只迁移 conv 权重和
BN 的 `weight / bias`。

### 2. 加载链路
- [utils/parameters.py](utils/parameters.py)：新增 CLI 参数 `--gfslt-resnet-path` / `--gfslt-resnet-strict` / `--gfslt-freeze-stages`
- [configs/baseline.yaml](configs/baseline.yaml)：暴露 `gfslt_resnet_path` / `gfslt_resnet_strict` / `gfslt_freeze_stages` 配置
- [slr_network.py](slr_network.py)：`SLRModel.load_gfslt_resnet_pretrain()`，在优化器初始化前完成加载
- [main.py](main.py)：把加载时机放在 optimizer 构建之前，确保冻结阶段的参数不会被注册到优化器

### 3. 关键设计取舍

**为什么不迁 BN running stats？**
BN 的 running statistics 是数据分布在训练过程中的快照，与任务高度相关。GFSLT 在视频-语言对齐
任务上学到的 BN 分布，与 CSLR 的 CTC 训练分布不一致，强行迁移会让 BN 在初始几个 step 给出
偏移很大的归一化结果，反而拖慢收敛。所以 `--skip-bn-stats=True` 是默认行为。

**只迁 ResNet，不迁其他模块**
GFSLT 的视觉编码后还有 Transformer、对比学习头等结构，与 SignGraph 的图模块（LSG/TSG）
完全不同。我们只取 conv2d 这部分公共子集。

## 为什么效果不行

将 VLP-SLT 视觉权重直接嵌入 SignGraph 后观察到：

1. **训练初期 loss 下降略快，但最终 WER 没有比 ImageNet 预训练更优**，甚至略差。
2. 即使跳过 BN running stats，前几个 epoch 仍可见明显的特征分布漂移。
3. GFSLT 的视觉表征是为「视频-文本对齐」优化的，更偏全局语义；而 CSLR + LSG/TSG 需要
   细粒度的局部时空动作特征，二者优化目标不重合。
4. 单纯替换 ResNet 初始化无法把 GFSLT 学到的多模态对齐能力带过来，那部分能力分布在
   它的 Transformer 与对齐头里，而这些与 SignGraph 架构不兼容。

## 提交历史

```
c658bef feat: skip BN running stats by default in GFSLT weight conversion
95933b8 fix: add backbone.conv_2d.resnet. prefix to GFSLT key matching
95194c7 Add GFSLT ResNet conversion tool for SignGraph
dce287e Load GFSLT ResNet weights before optimizer initialization
16e5f19 Expose GFSLT ResNet pretrain config
abdf9c0 Add GFSLT ResNet pretrain CLI options
40f415b Add GFSLT ResNet pretrain loader to SLRModel
```

> 注：本分支基于 `feature/corrnet-layer3-layer4` 演进，因此也包含 CorrNet 模块的提交
> （`f5ea8ac`、`cc8a80d`），那部分属于另一条实验线，参见 `feature/corrnet-layer3-layer4`。

## 使用方式（如需复现失败结果）

```bash
# 1. 转换 GFSLT 权重
python tools/convert_gfslt_resnet_to_signgraph.py \
    --src /path/to/gfslt_stage1.pth \
    --dst pretrained/gfslt_resnet18_conv2d_only.pth \
    --stages stem layer1 layer2 layer3 layer4

# 2. 在 config 中指定
# configs/baseline.yaml:
#   gfslt_resnet_path: pretrained/gfslt_resnet18_conv2d_only.pth
#   gfslt_resnet_strict: False
#   gfslt_freeze_stages: []   # 也可冻结 ['stem','layer1'] 等

# 3. 正常训练
python main.py --device 0
```

## 后续方向

- 不再追求直接迁权重，转而考虑**蒸馏**：用 GFSLT 视觉编码器作为 teacher，对齐特征空间。
- 或者只在**初始化阶段**用 GFSLT 权重，但配合较强的 LR warmup 与 BN 重统计。
- 当前主线（`main`）回退到 ImageNet 预训练 + LSG/TSG 图模块的方案。
