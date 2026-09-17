"""Fail-closed verification for the M2 inference delivery."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile


M2_ROOT = Path(__file__).resolve().parent
AGENT_DIR = M2_ROOT / "m2_final_submission"
PACKAGE_DIR = M2_ROOT / "package"
EXPECTED_NUMPY_VERSION = "2.5.3"
EXPECTED_TORCH_VERSION = "2.11.0"
EXPECTED_POLICY_TENSORS = 19


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _torch_library_path() -> Path:
    specification = importlib.util.find_spec("torch")
    if specification is None or not specification.submodule_search_locations:
        raise RuntimeError("PyTorch is not installed in the active environment")
    package_root = Path(next(iter(specification.submodule_search_locations)))
    return package_root / "lib" / "libtorch_cpu.so"


def _verify_linux_stack_flag() -> str:
    if not sys.platform.startswith("linux"):
        return "not-applicable"

    library = _torch_library_path()
    if not library.is_file():
        raise RuntimeError(f"PyTorch CPU library is missing: {library}")
    readelf = shutil.which("readelf")
    if readelf is None:
        raise RuntimeError("readelf is required to verify the GNU_STACK flag")

    completed = subprocess.run(
        [readelf, "-W", "-l", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    stack_lines = [
        line.strip() for line in completed.stdout.splitlines() if "GNU_STACK" in line
    ]
    if len(stack_lines) != 1:
        raise RuntimeError(f"Expected one GNU_STACK header in {library}")
    flag_tokens = [
        token
        for token in stack_lines[0].split()
        if token and set(token).issubset({"R", "W", "E"})
    ]
    if not flag_tokens:
        raise RuntimeError(f"Could not parse GNU_STACK flags: {stack_lines[0]}")
    flags = flag_tokens[-1]
    if "E" in flags:
        raise RuntimeError(
            f"Unsafe executable-stack flag on {library}; GNU_STACK={flags}"
        )
    return flags


def _verify_archive() -> tuple[dict, Path]:
    manifest_path = PACKAGE_DIR / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not manifest.get("inference_only"):
        raise RuntimeError("Submission manifest is not an inference-only PASS")

    archive = PACKAGE_DIR / manifest["archive"]
    if _sha256(archive) != manifest["archive_sha256"]:
        raise RuntimeError("Submission archive SHA-256 does not match the manifest")

    expected_members = sorted(manifest["members"])
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Submission archive failed its CRC check")
        if sorted(bundle.namelist()) != expected_members:
            raise RuntimeError("Submission archive members do not match the manifest")
        for member in expected_members:
            name = Path(member).name
            digest = hashlib.sha256(bundle.read(member)).hexdigest().upper()
            if digest != manifest["member_sha256"][name]:
                raise RuntimeError(f"Archive member hash mismatch: {member}")

    for name, expected_hash in manifest["member_sha256"].items():
        if _sha256(AGENT_DIR / name) != expected_hash:
            raise RuntimeError(f"Source package hash mismatch: {name}")
    return manifest, archive


def main() -> None:
    stack_flags = _verify_linux_stack_flag()

    import numpy as np
    import torch

    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise RuntimeError(
            f"Expected NumPy {EXPECTED_NUMPY_VERSION}, got {np.__version__}"
        )
    if not torch.__version__.startswith(EXPECTED_TORCH_VERSION):
        raise RuntimeError(
            f"Expected PyTorch {EXPECTED_TORCH_VERSION}, got {torch.__version__}"
        )

    for path in AGENT_DIR.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    sys.path.insert(0, str(M2_ROOT))
    from m2_final_submission.config import (
        ACTIONS,
        AUX_FEATURES,
        BOARD_CHANNELS,
        BOARD_SIZE,
        configured_stage,
    )
    from m2_final_submission.model import (
        DuelingQNetwork,
        load_checkpoint,
        load_policy_state,
    )

    if configured_stage() != "task4":
        raise RuntimeError("M2 must default to Task 4 for delivery")

    manifest, archive = _verify_archive()
    payload = load_checkpoint(AGENT_DIR / "model.pt", torch.device("cpu"))
    if set(payload) != {"format_version", "policy_state"}:
        raise RuntimeError("Checkpoint contains non-inference state")
    if len(payload["policy_state"]) != EXPECTED_POLICY_TENSORS:
        raise RuntimeError("Unexpected number of policy tensors")

    model = DuelingQNetwork(len(ACTIONS))
    migrated = load_policy_state(model, payload["policy_state"])
    model.eval()
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("Checkpoint contains non-finite model values")

    board = torch.zeros((1, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    auxiliary = torch.zeros((1, AUX_FEATURES))
    with torch.inference_mode():
        q_values = model(board, auxiliary)
    if q_values.shape != (1, len(ACTIONS)) or not torch.isfinite(q_values).all():
        raise RuntimeError("M2 forward-pass smoke test failed")

    report = {
        "archive_sha256": _sha256(archive),
        "archive_members": len(manifest["members"]),
        "checkpoint_migrated": migrated,
        "default_stage": configured_stage(),
        "gnu_stack": stack_flags,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "policy_tensors": len(payload["policy_state"]),
        "python": platform.python_version(),
        "q_shape": list(q_values.shape),
        "status": "PASS",
        "torch": torch.__version__,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
