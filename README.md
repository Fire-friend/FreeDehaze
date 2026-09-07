# FreeDehaze

Single-image inference for **FreeDehaze: Towards Training-free Real-world Image Dehazing via Diffusion Degradation Prior**.

## Setup

Python 3.10/3.11 and an NVIDIA CUDA GPU are required. Run from this directory:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

PCA bases are included in `checkpoints/pca/`. Place [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) in Diffusers format (`model_index.json` and component folders) in `checkpoints/sd15/`, and [Long-CLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L/tree/main)'s `longclip-L.pt` in `checkpoints/`. These pretrained models are not included in Git; inference loads them locally.

## Inference

The prompt is empty by default:

```bash
python inference.py --input hazy.jpg --output dehazed.png
```

You can also provide a prompt:

```bash
python inference.py --input hazy.jpg --output dehazed.png --prompt "A street with buildings and trees."
```

Use `--model` and `--longclip` to specify other local model paths. Defaults: 512 × 512 internally, 100 diffusion steps, and 100 correction iterations. The final image is resized back to the input dimensions after correction. Algorithm settings are in `config.py`.

The default diffusion settings use the RTTS configuration; single-image correction runs afterward. Only one comparison image is saved, with five labeled panels from left to right: **Input | Perception | Inversion | Result | Corrected**. Each panel retains the input dimensions, with labels above. `Result` is the image before correction.
