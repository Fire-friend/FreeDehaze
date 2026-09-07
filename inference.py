"""FreeDehaze inference for a single image."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def make_comparison(result):
    """Arrange the five outputs horizontally at their original image size."""
    from PIL import Image, ImageDraw, ImageFont

    panels = (
        ("Input", result.input),
        ("Perception", result.perception),
        ("Inversion", result.reconstruction),
        ("Result", result.dehazed),
        ("Corrected", result.image),
    )
    width, height = result.input.size
    font_size = max(10, min(24, width // 12))
    label_height = font_size + 16
    font = ImageFont.load_default(size=font_size)
    comparison = Image.new("RGB", (width * len(panels), height + label_height), "white")
    draw = ImageDraw.Draw(comparison)
    for index, (label, image) in enumerate(panels):
        left = index * width
        comparison.paste(image, (left, label_height))
        draw.text(
            (left + width // 2, label_height // 2),
            label,
            fill="black",
            font=font,
            anchor="mm",
        )
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input image")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dehazed.png"),
        help="Output path for the five-panel comparison image",
    )
    parser.add_argument("--prompt", default="", help="Optional image description")
    parser.add_argument(
        "--model", default=str(ROOT / "checkpoints/sd15"), help="SD 1.5 model path"
    )
    parser.add_argument("--longclip", default=str(ROOT / "checkpoints/longclip-L.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2038)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error("--input must be a single image file")

    from pipeline import FreeDehazePipeline
    from config import InferenceConfig

    pipeline = FreeDehazePipeline.from_pretrained(
        args.model,
        longclip_path=args.longclip,
        config=InferenceConfig(steps=args.steps, seed=args.seed),
        local_files_only=True,
    )
    result = pipeline(args.input, prompt=args.prompt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_comparison(result).save(args.output)
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
