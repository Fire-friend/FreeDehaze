"""Lossless PCA sharding with explicit diffusion timestep metadata."""

import hashlib
import json
from pathlib import Path

DEFAULT_BASIS = Path(__file__).resolve().parent / "checkpoints" / "pca"


def load_basis(path, timesteps, components=64):
    import torch
    from safetensors.torch import load_file

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported PCA format")
    if components > manifest["components"]:
        raise ValueError(f"PCA contains only {manifest['components']} components")
    available = manifest["timesteps"]
    missing = sorted(set(int(t) for t in timesteps) - set(available))
    if missing:
        raise ValueError(
            f"PCA does not cover diffusion timesteps {missing}. Use a compatible "
            "step count (bundled basis: 10/20/25/50/100), or export matching bases."
        )
    flat = {}
    for shard in manifest["shards"]:
        file = path / shard["file"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError(f"PCA checksum mismatch: {file}")
        flat.update(load_file(str(file), device="cpu"))
    result = {}
    for i, t in enumerate(timesteps):
        prefix = str(available.index(int(t)))
        layers = {}
        for layer in manifest["layers"]:
            mean = flat[f"{prefix}.{layer}.mean"]
            basis = flat[f"{prefix}.{layer}.basis"][:components]
            if mean.ndim != 1 or basis.shape[1] != mean.shape[0]:
                raise ValueError(f"Invalid PCA dimensions at {t}: {layer}")
            if not torch.isfinite(mean).all() or not torch.isfinite(basis).all():
                raise ValueError(f"Nonfinite PCA values at {t}: {layer}")
            layers[layer] = {"mean": mean, "basis": basis}
        result[i] = {"attn_value": layers}
    return result, manifest


def export_legacy_basis(source, destination, steps):
    """Preserve every tensor; legacy attn_key naming does not prove its provenance."""
    import torch
    from safetensors.torch import save_file

    source, destination = Path(source), Path(destination)
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"PCA already exists: {destination}")
    if steps < 2 or 1000 % steps:
        raise ValueError("steps must divide 1000")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if set(raw) != set(range(steps)):
        raise ValueError("PCA step indices do not match --steps")
    kind = "attn_value" if "attn_value" in raw[0] else "attn_key"
    layers = sorted(raw[0][kind])
    destination.mkdir(parents=True, exist_ok=True)
    shards = []
    for start in range(0, steps, 10):
        tensors = {
            f"{i}.{layer}.{key}": raw[i][kind][layer][key].contiguous().clone()
            for i in range(start, min(start + 10, steps))
            for layer in layers
            for key in ("mean", "basis")
        }
        filename = f"basis-{start:03d}.safetensors"
        save_file(tensors, str(destination / filename))
        shards.append(
            {
                "file": filename,
                "sha256": hashlib.sha256(
                    (destination / filename).read_bytes()
                ).hexdigest(),
            }
        )
    manifest = {
        "format_version": 1,
        "model_family": "stable-diffusion-1.5",
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_feature_key": kind,
        "runtime_feature": "value",
        "provenance_note": "Legacy research basis; extraction feature provenance is not recorded in the source file.",
        "steps": steps,
        "timesteps": list(range(1, 1000, 1000 // steps))[::-1],
        "components": raw[0][kind][layers[0]]["basis"].shape[0],
        "layers": layers,
        "shards": shards,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
