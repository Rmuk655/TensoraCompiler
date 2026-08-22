#!/usr/bin/env python3
"""Prepare the CNN CIFAR-100 image-classifier real-inference demo.

Trains a CNN once in PyTorch on CIFAR-100's 20 coarse "superclasses"
(images upscaled to 96x96 so they are large and clearly viewable), then
exports the trained weights and several real held-out test images:

  demos/artifacts/cnn/
      classes.json            -> the 20 superclass names (for the runner)
      sample{0..N}/
          x.npy convW1..4.npy convB1..4.npy fcW.npy fcB.npy
                              -> a self-contained --data-dir
          expected.npy        -> the PyTorch reference logits
          image.png           -> the image, for GUI display
          thumb.npy           -> an RGB thumbnail, for a terminal preview
          meta.json           -> true class + PyTorch prediction

Each sample dir is directly usable as `tensora --run examples/cnn_cifar.tns
--data-dir <dir>`. Run once, on a machine with PyTorch + torchvision
(see tools/models/requirements.txt):
    python3 tools/models/prepare_cifar_demo.py
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
EPOCHS = 12
# CIFAR-100 images are natively 32x32; we upscale to 96x96 so the demo
# image is large and clearly viewable. The model is trained at 96x96
# too, so PyTorch and Tensora run the identical computation.
SIZE = 96
EVAL_N = 40   # held-out images classified in one run; matches examples/cnn_cifar.tns
DISPLAY = 5   # how many of them to show with their pictures (distinct superclasses)
# CIFAR-100 per-channel normalization (standard values).
MEAN = (0.5071, 0.4866, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "cifar-data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "cnn")
# Cache the trained model so re-running prep (e.g. to change the demo
# display) does not retrain from scratch. Delete this file to retrain.
MODEL_PT = os.path.join(os.path.dirname(__file__), "..", "..", "demos", "artifacts", "cnn-model.pt")

# The 20 CIFAR-100 coarse superclasses, and the standard fine(0-99) ->
# coarse(0-19) mapping.
SUPERCLASSES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium mammals", "non-insect invertebrates",
    "people", "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2",
]
FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


class CNN(nn.Module):
    # Four conv/pool blocks: 96 -> 48 -> 24 -> 12 -> 6 spatially, then a
    # linear head over 64*6*6 = 2304 features. Mirrors
    # examples/cnn_cifar.tns exactly.
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 6 * 6, 20)

    def forward(self, x):
        x = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv3(x)), 2)
        x = F.max_pool2d(torch.relu(self.conv4(x)), 2)
        return self.fc(x.flatten(1))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    coarse = torch.tensor(FINE_TO_COARSE)
    tf = T.Compose([T.Resize(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])
    print("Downloading / loading CIFAR-100 ...")
    train = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)

    model = CNN()
    if os.path.exists(MODEL_PT):
        model.load_state_dict(torch.load(MODEL_PT))
        print(f"Loaded cached model from {os.path.normpath(MODEL_PT)} (delete it to retrain)")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), coarse[yb]).backward()
                opt.step()
            print(f"  epoch {epoch + 1}/{EPOCHS} done")
        torch.save(model.state_dict(), MODEL_PT)
    model.eval()

    # Held-out accuracy over the 20 superclasses.
    correct = total = 0
    with torch.no_grad():
        for xb, yb in torch.utils.data.DataLoader(test, batch_size=256):
            correct += (model(xb).argmax(1) == coarse[yb]).sum().item()
            total += yb.size(0)
    print(f"CNN (20 superclasses): held-out accuracy = {correct / total:.3f}")

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

    raw = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=False)  # unnormalized, for display

    # Evaluate a batch of EVAL_N random held-out images in one Tensora run
    # (batch axis n in examples/cnn_cifar.tns), so Tensora-vs-PyTorch
    # agreement is statistically meaningful. A random draw keeps the
    # accuracy honest (a real mix of right and wrong).
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(test), size=EVAL_N, replace=False).tolist()
    xb = torch.stack([test[i][0] for i in idxs])  # [EVAL_N, 3, SIZE, SIZE]
    true_coarse = [int(coarse[test[i][1]]) for i in idxs]
    with torch.no_grad():
        logits = model(xb).numpy().astype(np.float32)  # [EVAL_N, 20]

    # Choose DISPLAY images to show with pictures: evenly spaced across
    # the batch, so they are spread out (not adjacent) and land on a
    # variety of superclasses (the batch is a random draw).
    display = sorted({round(k * (EVAL_N - 1) / (DISPLAY - 1)) for k in range(DISPLAY)})

    os.makedirs(OUT_DIR, exist_ok=True)
    d = os.path.join(OUT_DIR, "eval")
    os.makedirs(d, exist_ok=True)
    json.dump(SUPERCLASSES, open(os.path.join(OUT_DIR, "classes.json"), "w"), indent=2)
    np.save(os.path.join(d, "x.npy"), xb.numpy().astype(np.float32))
    for wname, arr in weights.items():
        np.save(os.path.join(d, f"{wname}.npy"), arr)
    np.save(os.path.join(d, "expected.npy"), logits)  # [EVAL_N, 20]

    for b in display:
        # Upscale the native 32x32 to SIZE so the shown image is viewable.
        pil_img = raw[idxs[b]][0].resize((SIZE, SIZE))
        pil_img.save(os.path.join(d, f"image_{b}.png"))
        np.save(os.path.join(d, f"thumb_{b}.npy"), np.asarray(pil_img, dtype=np.uint8))

    meta = {
        "count": EVAL_N,
        "true_classes": [SUPERCLASSES[c] for c in true_coarse],
        "display": [{"index": b, "true_class": SUPERCLASSES[true_coarse[b]]} for b in display],
    }
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    pt_acc = float((logits.argmax(1) == np.array(true_coarse)).mean())
    print(f"Wrote {EVAL_N}-image eval batch to {os.path.normpath(OUT_DIR)} "
          f"(PyTorch accuracy on this batch = {pt_acc:.3f}); showing {len(display)} images")


if __name__ == "__main__":
    main()
