"""Shape test and backward test for SignGraph-Corr fusion model.
Verifies that the modified ResNet backbone outputs correct dimensions.
"""
import torch
import torch.nn as nn
from modules.resnet import resnet18

def main():
    model = resnet18()
    model.fc = nn.Identity()
    model.eval()

    # Test input: [B=2, C=3, T=16, H=224, W=224]
    x = torch.randn(2, 3, 16, 224, 224)
    
    with torch.no_grad():
        y = model(x)
    print(f"Forward output shape: {y.shape}")
    assert y.shape == (2 * 16, 512), f"Expected (32, 512), got {y.shape}"
    print("✅ Forward shape test PASSED")

    # Backward test
    model.train()
    x = torch.randn(2, 3, 16, 224, 224)
    y = model(x)
    loss = y.mean()
    loss.backward()
    print("✅ Backward test PASSED")

    # Check alpha values
    print(f"\nalpha_graph: {model.alpha_graph.data}")
    print(f"alpha_corr: {model.alpha_corr.data}")
    print(f"alpha_graph sum: {model.alpha_graph.data.sum().item():.4f}")
    print(f"alpha_corr sum: {model.alpha_corr.data.sum().item():.4f}")

    # Parameter count comparison
    total_params = sum(p.numel() for p in model.parameters())
    corr_params = sum(p.numel() for p in model.corr3.parameters()) + sum(p.numel() for p in model.corr4.parameters())
    graph_params = sum(p.numel() for p in model.localG.parameters()) + sum(p.numel() for p in model.localG2.parameters()) \
                 + sum(p.numel() for p in model.temporalG.parameters()) + sum(p.numel() for p in model.temporalG2.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"CorrBlock parameters (corr3+corr4): {corr_params:,}")
    print(f"Graph parameters (localG+localG2+temporalG+temporalG2): {graph_params:,}")
    print(f"New CorrBlock adds {corr_params/total_params*100:.1f}% parameters")

if __name__ == "__main__":
    main()
