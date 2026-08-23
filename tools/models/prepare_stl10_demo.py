#!/usr/bin/env python3
"""Prepare the CNN STL-10 image-classifier real-inference demo.

Trains a CNN once in PyTorch on STL-10's 10 object classes, then exports
the trained weights and several real held-out test images:

  demos/artifacts/cnn/
      classes.json            -> the 10 class names (for the runner)
      eval/
          x.npy convW1..4.npy convB1..4.npy fcW.npy fcB.npy
                              -> a self-contained --data-dir (EVAL_N images)
          expected.npy        -> the PyTorch reference logits, one row per image
          image_<i>.png       -> a displayed image (opened in a GUI viewer)
          meta.json           -> true classes + which images are displayed

`eval/` is directly usable as `tensora --run examples/cnn_stl10.tns
--data-dir demos/artifacts/cnn/eval`. Run once, on a machine with PyTorch + torchvision
(see tools/models/requirements.txt):
    python3 tools/models/prepare_stl10_demo.py

STL-10 images are natively 96x96 -- unlike CIFAR-10/100 (32x32), no
upscaling is needed, so the displayed picture is genuinely that
resolution, not an interpolated enlargement (see ARCH-017). The STL-10
download (~2.5 GB) is larger than CIFAR's (~170 MB); that is the
tradeoff for native resolution.
"""
import json
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as T
except ImportError:
    raise SystemExit("error: PyTorch + torchvision are required. "
                     "Install with: pip install -r tools/models/requirements.txt")

SEED = 0
EPOCHS = 40  # STL-10's labeled train split is only 5000 images (vs CIFAR's 50000)
SIZE = 96    # STL-10's native resolution -- no resize/upscale needed
EVAL_N = 40   # held-out images classified in one run; matches examples/cnn_stl10.tns
DISPLAY = 5   # how many of them to show with their pictures (distinct classes)
# STL-10 per-channel normalization (standard values).
MEAN = (0.4467, 0.4398, 0.4066)
STD = (0.2603, 0.2566, 0.2713)
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "stl10-data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "cnn")
# Cache the trained model so re-running prep (e.g. to change the demo
# display) does not retrain from scratch. Delete this file to retrain.
MODEL_PT = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "cnn-model.pt")


class CNN(nn.Module):
    # Four conv/pool blocks: 96 -> 48 -> 24 -> 12 -> 6 spatially, then a
    # linear head over 64*6*6 = 2304 features. Mirrors
    # examples/cnn_stl10.tns exactly.
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 6 * 6, num_classes)

    def forward(self, x):
        x = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv3(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv4(x)), 2)
        return self.fc(x.flatten(1))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # STL-10's labeled train split is small (5000 images), so the model
    # sees each one many times over EPOCHS; random flip/crop augmentation
    # is the standard fix -- it synthesizes new views instead of the model
    # memorizing exact pixels, and is the single biggest accuracy lever
    # available without more labeled data. Only the TRAIN transform is
    # augmented: eval (held-out accuracy, and the batch fed to Tensora)
    # must see the real, unmodified image, or Tensora-vs-PyTorch agreement
    # would not mean what it claims.
    train_tf = T.Compose([
        T.RandomCrop(SIZE, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    print("Downloading / loading STL-10 (~2.5 GB, native 96x96) ...")
    train = torchvision.datasets.STL10(DATA_ROOT, split="train", download=True, transform=train_tf)
    test = torchvision.datasets.STL10(DATA_ROOT, split="test", download=True, transform=eval_tf)
    loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    classes = list(train.classes)  # ['airplane', 'bird', 'car', ..., 'truck']

    model = CNN(len(classes))
    cached_ok = False
    if os.path.exists(MODEL_PT):
        state = torch.load(MODEL_PT)
        # A cache from a different architecture or class count (e.g. the
        # old 20-superclass CIFAR-100 model, before ARCH-017) has a
        # different fc shape; loading it would crash with a shape
        # mismatch. Detect that and retrain instead of failing.
        if state.get("fc.weight") is not None and state["fc.weight"].shape[0] == len(classes):
            model.load_state_dict(state)
            cached_ok = True
            print(f"Loaded cached model from {os.path.normpath(MODEL_PT)} (delete it to retrain)")
        else:
            print(f"Cached model at {os.path.normpath(MODEL_PT)} does not match "
                  f"the current {len(classes)}-class model (stale from a previous "
                  f"dataset/architecture) -- retraining.")
    if not cached_ok:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
            print(f"  epoch {epoch + 1}/{EPOCHS} done")
        torch.save(model.state_dict(), MODEL_PT)
    model.eval()

    # Held-out accuracy over the 10 classes.
    correct = total = 0
    with torch.no_grad():
        for xb, yb in torch.utils.data.DataLoader(test, batch_size=256):
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.size(0)
    print(f"CNN ({len(classes)} classes): held-out accuracy = {correct / total:.3f}")

    with torch.no_grad():
        weights = {
            "convW1": model.conv1.weight.numpy().astype(np.float32),
            "convB1": model.conv1.bias.numpy().astype(np.float32),
            "convW2": model.conv2.weight.numpy().astype(np.float32),
            "convB2": model.conv2.bias.numpy().astype(np.float32),
            "convW3": model.conv3.weight.numpy().astype(np.float32),
            "convB3": model.conv3.bias.numpy().astype(np.float32),
            "convW4": model.conv4.weight.numpy().astype(np.float32),
            "convB4": model.conv4.bias.numpy().astype(np.float32),
            # linear: y = x @ W + b, so W is [in, out] (PyTorch is [out, in]).
            "fcW": model.fc.weight.t().contiguous().numpy().astype(np.float32),
            "fcB": model.fc.bias.numpy().astype(np.float32),
        }

    raw = torchvision.datasets.STL10(DATA_ROOT, split="test", download=False)  # unnormalized, for display

    # Evaluate a batch of EVAL_N random held-out images in one Tensora run
    # (batch axis n in examples/cnn_stl10.tns), so Tensora-vs-PyTorch
    # agreement is statistically meaningful. A random draw keeps the
    # accuracy honest (a real mix of right and wrong).
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(test), size=EVAL_N, replace=False).tolist()
    xb = torch.stack([test[i][0] for i in idxs])  # [EVAL_N, 3, SIZE, SIZE]
    true_classes = [int(test[i][1]) for i in idxs]
    with torch.no_grad():
        logits = model(xb).numpy().astype(np.float32)  # [EVAL_N, len(classes)]

    # Choose DISPLAY images to show with pictures: evenly spaced across
    # the batch, so they are spread out (not adjacent) and land on a
    # variety of classes (the batch is a random draw).
    display = sorted({round(k * (EVAL_N - 1) / (DISPLAY - 1)) for k in range(DISPLAY)})

    os.makedirs(OUT_DIR, exist_ok=True)
    d = os.path.join(OUT_DIR, "eval")
    os.makedirs(d, exist_ok=True)
    json.dump(classes, open(os.path.join(OUT_DIR, "classes.json"), "w"), indent=2)
    np.save(os.path.join(d, "x.npy"), xb.numpy().astype(np.float32))
    for wname, arr in weights.items():
        np.save(os.path.join(d, f"{wname}.npy"), arr)
    np.save(os.path.join(d, "expected.npy"), logits)  # [EVAL_N, len(classes)]

    for b in display:
        # Native 96x96 -- saved as-is, no upscaling.
        raw[idxs[b]][0].save(os.path.join(d, f"image_{b}.png"))

    meta = {
        "count": EVAL_N,
        "true_classes": [classes[c] for c in true_classes],
        "display": [{"index": b, "true_class": classes[true_classes[b]]} for b in display],
    }
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    pt_acc = float((logits.argmax(1) == np.array(true_classes)).mean())
    print(f"Wrote {EVAL_N}-image eval batch to {os.path.normpath(OUT_DIR)} "
          f"(PyTorch accuracy on this batch = {pt_acc:.3f}); showing {len(display)} images")


if __name__ == "__main__":
    main()
