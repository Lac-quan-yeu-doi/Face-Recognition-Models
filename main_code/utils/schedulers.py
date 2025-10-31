import torch

class CustomStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, steps, ratio=0.1, last_epoch=-1):
        self.steps = set(steps)
        self.ratio = ratio
        super().__init__(optimizer, last_epoch)
        print(f"### Iinitialize CustomStepLR wiht steps = {self.steps}")

    def get_lr(self):
        if self.last_epoch in self.steps:
            print("## LEARNING RATE CHANGED ##")
            return [group['lr'] * self.ratio for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]

SCHEDULER_CONFIGS = {
    "step": {"step_size": 30, "gamma": 0.1},
    "multistep": {"milestones": [40, 80, 100, 150], "gamma": 0.1},
    # "customstep": {"steps": [10, 15, 20, 22, 24, 26, 28], "ratio": 0.5},
    "customstep": {"steps": [20, 40, 60], "ratio": 0.1},
    "cosine": {"eta_min": 0},
    # "exponential": {"gamma": 0.95},
    # "cosine_warm_restarts": {"T_0": 10, "T_mult": 2, "eta_min": 0},
    # "plateau": {"mode": "min", "factor": 0.1, "patience": 10, "threshold": 1e-4, "min_lr": 0},
    # "cyclic": {"base_lr": 1e-5, "max_lr": 1e-2, "mode": "triangular2"},
    # "onecycle": {"max_lr": 1e-2},
    # "warmup_cosine": {"warmup_epochs": 5},
    "none": {},
}

SCHEDULER_DICT = {i + 1: name for i, name in enumerate(SCHEDULER_CONFIGS.keys())}

def get_scheduler(optimizer, choice, num_epochs=None, steps_per_epoch=None, **overrides):
    if isinstance(choice, int):
        if choice not in SCHEDULER_DICT:
            raise ValueError(f"Invalid scheduler id: {choice}")
        scheduler_name = SCHEDULER_DICT[choice]
    else:
        scheduler_name = choice.lower()

    cfg = {**SCHEDULER_CONFIGS.get(scheduler_name, {}), **overrides}

    # Lazy imports
    if scheduler_name == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, **cfg)

    elif scheduler_name == "multistep":
        from torch.optim.lr_scheduler import MultiStepLR
        return MultiStepLR(optimizer, **cfg)

    elif scheduler_name == "customstep":
        return CustomStepLR(optimizer, **cfg)

    elif scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        if num_epochs is None:
            raise ValueError("num_epochs must be provided for cosine scheduler")
        return CosineAnnealingLR(optimizer, T_max=num_epochs, **cfg)

    elif scheduler_name == "exponential":
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, **cfg)

    elif scheduler_name == "cosine_warm_restarts":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        return CosineAnnealingWarmRestarts(optimizer, **cfg)

    elif scheduler_name == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, **cfg)

    elif scheduler_name == "cyclic":
        from torch.optim.lr_scheduler import CyclicLR
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided for CyclicLR")
        if "step_size_up" not in cfg:
            cfg["step_size_up"] = steps_per_epoch // 2
        return CyclicLR(optimizer, **cfg, cycle_momentum=False)

    elif scheduler_name == "onecycle":
        from torch.optim.lr_scheduler import OneCycleLR
        if num_epochs is None or steps_per_epoch is None:
            raise ValueError("num_epochs and steps_per_epoch must be provided for OneCycleLR")
        return OneCycleLR(optimizer, epochs=num_epochs, steps_per_epoch=steps_per_epoch, **cfg)

    elif scheduler_name == "warmup_cosine":
        from torch.optim.lr_scheduler import LambdaLR
        if num_epochs is None:
            raise ValueError("num_epochs must be provided for warmup_cosine scheduler")
        warmup_epochs = cfg.get("warmup_epochs", 5)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / float(warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif scheduler_name == "none":
        return None

    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")

if __name__ == "__main__":
    print("-" * 40)
    print("Available Schedulers:")
    print("-" * 40)
    for key, name in SCHEDULER_DICT.items():
        cfg = SCHEDULER_CONFIGS[name]
        if cfg:
            params = ", ".join(f"{k}={v}" for k, v in cfg.items())
            print(f"({key}): {name.capitalize()}({params})")
        else:
            print(f"({key}): {name.capitalize()}()")
