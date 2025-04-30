# Minimal build.py for SOLIDER inference
from .solider import SOLIDER

def build_solider(cfg, num_classes=1):
    # cfg is expected to be a dict with model parameters
    model = SOLIDER(cfg, num_classes)
    return model
