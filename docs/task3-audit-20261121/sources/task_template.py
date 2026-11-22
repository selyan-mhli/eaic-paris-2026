"""Machine unlearning challenge - task template (ViT-B/16).

You are given
  model.pt            state dict of a 100-class image classifier (the "original" model)
  forget_images.npz   the forget set: key "images", uint8 array [N, 224, 224, 3] (N = 3500, RGB, channels LAST).
                      No labels are provided.

Goal: produce a model that behaves like the "gold" model - the same architecture trained with the same recipe on the
same data WITHOUT the forget images - while staying as close as possible to the original model everywhere else.

IMAGE FORMAT
  The forget images are raw uint8 pixels in [N, H, W, 3] layout (they are NOT preprocessed and NOT channel-first).
  The model expects float32 [B, 3, 224, 224] normalised to [-1, 1]: x = (pixels / 255 - 0.5) / 0.5.
  Use preprocess() below (it does exactly what the server does) - do not resize or normalise a second time.

Scoring (computed on the server)
  The forget score is measured on the forget images AND on hidden altered copies of them (modified versions of them that you
  do not have); the two groups count equally. For each group
    mean/std score    d = |mean(L) - gold_mean| + |std(L) - gold_std|, where L is the per-sample cross-entropy loss of
                      your model and gold_mean / gold_std are those of retrained gold models;  1 - clip(d / d_original, 0, 1)
    prediction score  1 - clip(mean KL(p_gold || p_yours) / mean KL(p_gold || p_original), 0, 1), where p_gold is the
                      average predicted class distribution (softmax at temperature 2) of several independently
                      retrained gold models for that image
    forget score      mean/std score * prediction score   (each averaged over the two groups first; both must be good)
  retain score  clip(acc(control) / acc_original(control), 0, 1) * 1 / (1 + mean KL(p_original || p_yours)) on a hidden
                control set (KL on softmax probabilities at temperature 2)
  final score   forget * retain
  utility gate  below 70% accuracy on a hidden validation set, the submission scores 0
The public leaderboard uses a random 30% of the forget images, the final ranking uses the other 70%.
Matching only the average loss is not enough: for every forget image (and unseen variations of it) your model has to
predict what a model that never saw the image would predict, including WHICH wrong class it would choose.

Submission: a state dict (torch.save(model.state_dict(), "submission.pt")) that loads with strict=True into the
architecture below. float32 weights (other dtypes are rejected), no NaN/Inf, at most 450 MB. The server runs the forward pass in eval mode under float16 autocast; this does not change the weights you submit.

Running this file loads the provided model and forget images, runs checks, and saves the unchanged original model as a
baseline submission (it scores about 0). Put your method in unlearn().
"""
import sys
from pathlib import Path

import numpy as np
import timm
import torch

# CONFIG
MODEL_FILE = Path("model.pt")
FORGET_IMAGES_FILE = Path("forget_images.npz")
OUTPUT_FILE = Path("submission.pt")

NUM_CLASSES = 100
IMAGE_SIZE = 224
EXPECTED_FORGET_IMAGES = 3500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> torch.nn.Module:
    """ViT-B/16 at 224x224 with LayerScale (init 1e-4), exactly as used for the original and gold models."""
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=NUM_CLASSES,
        init_values=1e-4,
    )


def load_model(path: Path = MODEL_FILE) -> torch.nn.Module:
    model = build_model()
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(DEVICE).eval()


def load_forget_images(path: Path = FORGET_IMAGES_FILE) -> torch.Tensor:
    """Returns the forget images as uint8 [N, 224, 224, 3] (raw pixels, channels last)."""
    with np.load(path, allow_pickle=False) as data:
        if "images" not in data.files:
            raise KeyError(f"{path} does not contain an 'images' array. Available arrays: {data.files}")
        images = torch.from_numpy(data["images"].copy())

    if images.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 pixels, got {images.dtype}")
    if images.ndim != 4:
        raise ValueError(f"Expected 4 dimensions [N, H, W, 3], got shape {tuple(images.shape)}")
    if images.shape[-1] != 3 and images.shape[1] == 3:        # tolerate channels-first input
        images = images.permute(0, 2, 3, 1).contiguous()
    if tuple(images.shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected images of shape [N, {IMAGE_SIZE}, {IMAGE_SIZE}, 3], got {tuple(images.shape)}")
    return images


def preprocess(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 224, 224, 3] -> float32 [B, 3, 224, 224] in [-1, 1] (mean = std = 0.5 per channel).
    This is exactly the preprocessing the evaluation server applies."""
    x = images_uint8.to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


@torch.no_grad()
def predict(model: torch.nn.Module, images_uint8: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Logits [N, 100] for uint8 images [N, 224, 224, 3], computed like the server (eval mode, fp16 autocast on GPU)."""
    model.eval()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(preprocess(images_uint8[i:i + batch_size])).float().cpu())
    return torch.cat(out)


def unlearn(model: torch.nn.Module, forget_images: torch.Tensor) -> torch.nn.Module:
    """TODO: your unlearning method. Must return a model with the same architecture (same parameters and buffers)."""
    return model


def main() -> None:
    print("Loading provided model checkpoint...")
    if not MODEL_FILE.exists():
        print(f"Error: {MODEL_FILE} not found.", file=sys.stderr)
        raise SystemExit(1)
    model = load_model()
    print("Model loaded successfully.")

    print("Loading forget set images...")
    if not FORGET_IMAGES_FILE.exists():
        print(f"Error: {FORGET_IMAGES_FILE} not found.", file=sys.stderr)
        raise SystemExit(1)
    forget_images = load_forget_images()
    print(f"Loaded {len(forget_images)} forget set images with shape {tuple(forget_images.shape)} ({forget_images.dtype}).")
    if len(forget_images) != EXPECTED_FORGET_IMAGES:
        print(f"[WARNING] Expected {EXPECTED_FORGET_IMAGES} forget images, got {len(forget_images)}.")

    print("original predictions on the first forget images:", predict(model, forget_images[:8]).argmax(1).tolist())

    # Baseline: submit the provided model unchanged (replace unlearn() with your method).
    # The final object must remain a plain state_dict that loads strictly into build_model().
    model = unlearn(model, forget_images)
    state_dict = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    torch.save(state_dict, OUTPUT_FILE)
    print(f"Saved baseline submission to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
