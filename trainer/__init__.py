__all__ = [
    "AFSTrainer",
]


def __getattr__(name):
    if name == "AFSTrainer":
        from .afs_trainer import AFSTrainer
        return AFSTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
