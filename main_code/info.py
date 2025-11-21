import os
import csv
import time
import torch
from torch.profiler import profile, ProfilerActivity

from utils.criterion import *
from utils.config import FEATURE_DIM


model_names = {
    "SphereFace": SphereFace,
    "CosFace": CosFace,
    "ArcFace": ArcFace,
    "MV_Softmax_cos": MV_SoftmaxCos,
    "MV_Softmax_arc": MV_SoftmaxArc,
    "CurricularFace": CurricularFace,
    "VPLArcFace": VPLArcFace,
    "MagFace": MagFace,
    "AdaFace": AdaFace,
    "ElasticCosFace": ElasticCosFace,
    "ElasticArcFace": ElasticArcFace,
    "SphereFace2": SphereFace2,
    "UniFace": UniFace,
    "UniTSFace": UniTSFace,
    "QAFace": QAFace,
}

num_classes = 10575
batch_size = 1


# --------------------------------------------------
# FLOPs Wrapper
# --------------------------------------------------
class FLOPsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out


# --------------------------------------------------
# FLOPs Measurement
# --------------------------------------------------
def count_flops_torch(model, model_name, num_classes, batch_size=1, feat_dim=512):

    model.eval()
    wrapped = FLOPsWrapper(model)

    feats = torch.randn(batch_size, feat_dim)
    minput = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    inputs = (feats, minput, labels) if model_name == "QAFace" else (feats, labels)

    with profile(
        activities=[ProfilerActivity.CPU],
        with_flops=True,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        wrapped(*inputs)

    # Sum FLOPs from PyTorch profiler
    flops = sum([e.flops for e in prof.key_averages() if e.flops is not None])
    return flops


# --------------------------------------------------
# Inference Time Measurement
# --------------------------------------------------
def measure_inference_time(
    model, model_name, num_classes, batch_size=1, feat_dim=512, runs=100
):

    model.eval()

    feats = torch.randn(batch_size, feat_dim)
    minput = torch.randn(batch_size, feat_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    inputs = (feats, minput, labels) if model_name == "QAFace" else (feats, labels)

    use_cuda = (
        next(model.parameters()).is_cuda
        if any(p.requires_grad for p in model.parameters())
        else False
    )

    # GPU path
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(10):
            model(*inputs)
        torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            starter.record()
            model(*inputs)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # ms

        avg_ms = sum(times) / len(times)

    else:
        # CPU path
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(*inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # convert sec → ms

        avg_ms = sum(times) / len(times)

    return avg_ms  # ms per inference


# --------------------------------------------------
# Main Program
# --------------------------------------------------
if __name__ == "__main__":

    os.makedirs("info_result", exist_ok=True)
    csv_path = f"info_result/flops_{batch_size}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ModelName", "MFLOPs", "InferenceTime(ms)"])

        for model_name, ModelClass in model_names.items():

            print("\n==============================")
            print(f"⚡ Processing model: {model_name}")
            print("==============================")

            # Instantiate model
            model = ModelClass(num_classes=num_classes, feat_dim=FEATURE_DIM)
            model.eval()

            # ------------------------
            # FLOPs
            # ------------------------
            flops = count_flops_torch(
                model=model,
                model_name=model_name,
                num_classes=num_classes,
                batch_size=batch_size,
            )
            mflops = flops / 1e6
            print(f"MFLOPs for {model_name}: {mflops:,.6f}")

            # ------------------------
            # Inference Time
            # ------------------------
            infer_ms = measure_inference_time(
                model=model,
                model_name=model_name,
                num_classes=num_classes,
                batch_size=batch_size,
            )
            print(f"Inference time: {infer_ms:.4f} ms")

            # ------------------------
            # Write result
            # ------------------------
            writer.writerow([model_name, f"{mflops:.6f}", f"{infer_ms:.4f}"])

    print(f"\n\n🎉 Results saved to: info_result/flops_{batch_size}.csv")
