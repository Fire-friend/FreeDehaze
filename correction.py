"""Single-image test-time correction and Otsu blending (paper Sec. IV-D)."""

import cv2
import numpy as np
import torch
from torch import nn
from correction_net import UNetSmall


def correction_mask(original, reconstruction):
    residual = np.abs(
        np.asarray(original, dtype=np.float32)
        - np.asarray(reconstruction, dtype=np.float32)
    ).mean(axis=2)
    residual[residual < max(float(residual.mean()), 10.0)] = 0
    span = float(np.ptp(residual))
    if span <= 1e-6:
        return np.zeros_like(residual)
    residual = ((residual - residual.min()) / span * 255).astype(np.uint8)
    mask = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    return cv2.GaussianBlur(mask, (51, 51), 0).astype(np.float32) / 255


def correct(
    original, reconstruction, dehazed, *, steps=100, lr=0.001, local=True, device="cuda"
):
    from PIL import Image

    mask = correction_mask(original, reconstruction)
    if steps == 0 or not np.any(mask):
        return dehazed.copy(), Image.fromarray((mask * 255).astype(np.uint8))

    def tensor(image):
        return (
            torch.from_numpy(np.array(image))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device, torch.float32)
            / 255
        )

    source, target, original_tensor = map(tensor, (reconstruction, dehazed, original))
    mask_tensor = torch.from_numpy(mask).to(device)[None, None]
    with torch.enable_grad():
        model = UNetSmall(3, 3).to(device).train()
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            error = (model(source) - target).abs()
            loss = (error * mask_tensor).mean() if local else error.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Correction loss became nonfinite")
            loss.backward()
            optimizer.step()
    # Batch statistics remain per-image, matching the original single-image script.
    with torch.no_grad():
        result = mask_tensor * model(original_tensor) + (1 - mask_tensor) * target
        array = result[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255).round().astype(np.uint8)), Image.fromarray(
        (mask * 255).round().astype(np.uint8)
    )
