"""Load the exact C3-v2 t225000 policy and keep it frozen."""

import hashlib

import torch
from torch import nn

from ..m2_3.model import TemporalDuelingDQN
from ..m2_4.c3_v2.model import C3ResidualAdapter
from .config import (
    ACTIONS,
    C1_MODEL_PATH,
    FEATURE_DIM,
    M2_3_MODEL_PATH,
    SOURCE_MODEL_PATH,
    SOURCE_SHA256,
    SOURCE_TRANSITION_COUNT,
)


def load_torch(path, map_location="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_files():
    for path in (M2_3_MODEL_PATH, C1_MODEL_PATH, SOURCE_MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError("M2-5 source file is missing: {}".format(path))
    if sha256(SOURCE_MODEL_PATH) != SOURCE_SHA256:
        raise ValueError("The frozen C3-v2 checkpoint checksum changed")

    source = load_torch(SOURCE_MODEL_PATH, "cpu")
    metadata = source.get("metadata", {})
    if metadata.get("variant_id") != "M2-4-C3-V2":
        raise ValueError("M2-5 source is not a C3-v2 checkpoint")
    if int(source.get("transition_count", -1)) != SOURCE_TRANSITION_COUNT:
        raise ValueError("M2-5 must inherit C3-v2 t0225000")
    if metadata.get("m2_3_source_sha256") != sha256(M2_3_MODEL_PATH):
        raise ValueError("The C3-v2 source uses a different M2-3 base")
    if metadata.get("c1_source_sha256") != sha256(C1_MODEL_PATH):
        raise ValueError("The C3-v2 source uses a different C1 model")
    return source


class FrozenC3Policy(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        source = verify_source_files()
        base_data = load_torch(M2_3_MODEL_PATH, self.device)
        c1_data = load_torch(C1_MODEL_PATH, self.device)

        self.base = TemporalDuelingDQN(FEATURE_DIM, len(ACTIONS)).to(self.device)
        self.c1 = C3ResidualAdapter(FEATURE_DIM, len(ACTIONS)).to(self.device)
        self.c3 = C3ResidualAdapter(FEATURE_DIM, len(ACTIONS)).to(self.device)
        self.base.load_state_dict(base_data["model_state_dict"], strict=True)
        self.c1.load_state_dict(c1_data["residual_state_dict"], strict=True)
        self.c3.load_state_dict(source["adapter_state_dict"], strict=True)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, states):
        with torch.no_grad():
            return self.base(states) + self.c1(states) + self.c3(states)

