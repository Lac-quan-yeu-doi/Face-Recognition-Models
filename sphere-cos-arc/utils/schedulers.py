import torch

class CustomStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, steps, ratio=0.1, last_epoch=-1):
        self.steps = set(steps)
        self.ratio = ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.steps:
            print(f"## LEARNING RATE CHANGED at epoch {self.last_epoch} ##")
            return [group['lr'] * self.ratio for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


SCHEDULER_CONFIGS = {
    "step": {"step_size": 30, "gamma": 0.1},
    "multistep": {"milestones": [40, 80, 90], "gamma": 0.1},
    "customstep": {"steps": [40, 80, 90], "ratio": 0.5},
    "cosine": {"eta_min": 0},
    "none": {},
}

SCHEDULER_DICT = {i + 1: name for i, name in enumerate(SCHEDULER_CONFIGS.keys())}


def get_scheduler(optimizer, choice, num_epochs=None, steps_per_epoch=None, last_epoch=-1, **overrides):
    """
    Create and return the appropriate learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.
        choice (int or str): Scheduler name or numeric ID.
        num_epochs (int): Total training epochs (required for cosine or warmup).
        steps_per_epoch (int): Steps per epoch (for cyclic or onecycle schedulers).
        last_epoch (int): The last completed epoch, used for resuming training.
        **overrides: Override default scheduler config values.

    Returns:
        torch.optim.lr_scheduler._LRScheduler or None
    """
    if isinstance(choice, int):
        if choice not in SCHEDULER_DICT:
            raise ValueError(f"Invalid scheduler id: {choice}")
        scheduler_name = SCHEDULER_DICT[choice]
    else:
        scheduler_name = choice.lower()

    cfg = {**SCHEDULER_CONFIGS.get(scheduler_name, {}), **overrides}

    if scheduler_name == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, last_epoch=last_epoch, **cfg)

    elif scheduler_name == "multistep":
        from torch.optim.lr_scheduler import MultiStepLR
        return MultiStepLR(optimizer, last_epoch=last_epoch, **cfg)

    elif scheduler_name == "customstep":
        return CustomStepLR(optimizer, last_epoch=last_epoch, **cfg)

    elif scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        if num_epochs is None:
            raise ValueError("num_epochs must be provided for cosine scheduler")
        return CosineAnnealingLR(optimizer, T_max=num_epochs, last_epoch=last_epoch, **cfg)

    elif scheduler_name == "exponential":
        from torch.optim.lr_scheduler import ExponentialLR
        return ExponentialLR(optimizer, last_epoch=last_epoch, **cfg)

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
        return CyclicLR(optimizer, cycle_momentum=False, **cfg)

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

        return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

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
