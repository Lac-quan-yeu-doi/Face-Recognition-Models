from fvcore.nn import FlopCountAnalysis
import torch

def count_qaface_flops(model,
                       model_name,
                       num_classes,
                       feat_dim: int = 512,
                       batch_size: int = 1):

    model.eval()   # make sure forward doesn't enter training-only memory update

    # ----- dummy inputs -----
    feats   = torch.randn(batch_size, feat_dim)        # backbone feature
    minput  = torch.randn(batch_size, feat_dim)        # momentum feature
    labels  = torch.randint(0, num_classes, (batch_size,))

    # ----- FLOPs -----
    flops = FlopCountAnalysis(
        model,
        (feats, minput, labels)
    )

    total_flops = flops.total()

    print(f"QAFace FLOPs (batch={batch_size}): {total_flops:,}")
    return total_flops
