import torch

OPTIMIZER_CONFIGS = {
    "sgd": {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "nesterov": False},
    "adam": {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0},
    "adamw": {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.01},
    "rmsprop": {"lr": 0.01, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0, "momentum": 0.9},
    "adagrad": {"lr": 0.01, "lr_decay": 0, "weight_decay": 0},
    # "nadam": {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0},
    # "adamax": {"lr": 0.002, "betas": (0.9, 0.999), "weight_decay": 0},
    # "lion": {"lr": 0.001, "betas": (0.9, 0.99), "weight_decay": 0.01},  # requires PyTorch ≥2.0
}

OPTIMIZER_DICT = {i + 1: name for i, name in enumerate(OPTIMIZER_CONFIGS.keys())}

def get_optimizer(model, choice, **overrides):
    """
    Create optimizer dynamically based on numeric ID or name.

    Args:
        model: torch.nn.Module
        choice: int or str — optimizer id or name
        **overrides: custom optimizer parameters (lr, weight_decay, etc.)
    """
    if isinstance(choice, int):
        if choice not in OPTIMIZER_DICT:
            raise ValueError(f"Invalid optimizer id: {choice}")
        opt_name = OPTIMIZER_DICT[choice]
    else:
        opt_name = choice.lower()

    cfg = {**OPTIMIZER_CONFIGS.get(opt_name, {}), **overrides}

    # Lazy imports for performance
    if opt_name == "sgd":
        from torch.optim import SGD
        return SGD(model.parameters(), **cfg)

    elif opt_name == "adam":
        from torch.optim import Adam
        return Adam(model.parameters(), **cfg)

    elif opt_name == "adamw":
        from torch.optim import AdamW
        return AdamW(model.parameters(), **cfg)

    elif opt_name == "rmsprop":
        from torch.optim import RMSprop
        return RMSprop(model.parameters(), **cfg)

    elif opt_name == "adagrad":
        from torch.optim import Adagrad
        return Adagrad(model.parameters(), **cfg)

    elif opt_name == "nadam":
        from torch.optim import NAdam
        return NAdam(model.parameters(), **cfg)

    elif opt_name == "adamax":
        from torch.optim import Adamax
        return Adamax(model.parameters(), **cfg)

    elif opt_name == "lion":
        try:
            from torch.optim import Lion  # available in PyTorch ≥ 2.0
            return Lion(model.parameters(), **cfg)
        except ImportError:
            raise ImportError("Lion optimizer not found. Requires PyTorch >= 2.0.")

    else:
        raise ValueError(f"Unknown optimizer name: {opt_name}")

if __name__ == "__main__":
    print("-" * 40)
    print("Available Schedulers:")
    print("-" * 40)
    for key, name in OPTIMIZER_DICT.items():
        cfg = OPTIMIZER_CONFIGS[name]
        params = ", ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"({key}): {name.upper()}({params})")
