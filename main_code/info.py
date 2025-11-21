import os
import csv
import torch
from fvcore.nn import FlopCountAnalysis

from utils.criterion import *

model_names = {
    'SphereFace': SphereFaceNet,
    'CosFace': CosFaceNet,
    'ArcFace': ArcFaceNet,
    'MV_Softmax_cos': MV_SoftmaxCosNet,
    'MV_Softmax_arc': MV_SoftmaxArcNet,
    'CurricularFace': CurricularFaceNet,
    'VPLArcFace': VPLArcFaceNet,
    'MagFace': MagFaceNet,
    'AdaFace': AdaFaceNet,
    'ElasticCosFace': ElasticCosFaceNet,
    'ElasticArcFace': ElasticArcFaceNet,
    'SphereFace2': SphereFace2Net,
    'UniFace': UniFaceNet,
    'UniTSFace': UniTSFaceNet,
    'QAFace': QAFaceNet,
    # 'QMagFace': QMagFaceNet
}

num_classes = 10575
backbone_name = "iresnet100"
batch_size = 32


# ============================================================
# FLOP counter for QAFace and QMagFace
# ============================================================
def count_qaface_flops(model, feat_dim, num_classes, batch_size=1):

    model.eval()

    feats   = torch.randn(batch_size, feat_dim)
    minput  = torch.randn(batch_size, feat_dim)
    labels  = torch.randint(0, num_classes, (batch_size,))

    flops = FlopCountAnalysis(
        model,
        (feats, minput, labels)
    )

    return flops.total()


# ============================================================
# FLOP counter for normal margin heads (ArcFace, CosFace, etc.)
# ============================================================
def count_normal_flops(model, feat_dim, num_classes, batch_size=1):

    model.eval()

    feats  = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    flops = FlopCountAnalysis(
        model,
        (feats, labels)
    )

    return flops.total()


if __name__ == "__main__":
    os.makedirs("info_result", exist_ok=True)

    csv_path = "info_result/flops.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ModelName", "FLOPs"])

        for model_name, ModelClass in model_names.items():

            print(f"\n==============================")
            print(f"⚡ Processing model: {model_name}")
            print("==============================")

            # Instantiate model
            model = ModelClass(num_classes=num_classes, backbone=backbone_name)
            model.eval()

            # Obtain feat_dim from backbone output
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 112, 112)
                feat = model.backbone(dummy_input)
                feat_dim = feat.shape[1]

            # Count flops
            if model_name in ["QAFace"]:
                flops = count_qaface_flops(model.sphereface if hasattr(model, 'sphereface') else model,
                                        feat_dim,
                                        num_classes,
                                        batch_size=1)
            else:
                # For standard heads
                flops = count_normal_flops(model.sphereface if hasattr(model, 'sphereface') else model,
                                        feat_dim,
                                        num_classes,
                                        batch_size=1)

            print(f"FLOPs for {model_name}: {flops:,}")

            writer.writerow([model_name, flops])

    print("\n\n🎉 FLOPs saved to: info_result/flops.csv")
