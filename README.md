# Tensora

Tensora is a small, standalone declarative language — a teaching
compiler's DSL, not a production framework — for describing
machine-learning computation graphs over a fixed, closed set of
built-in tensor operators (`matmul`, `linear`, `relu`, `softmax`,
`conv2d`, etc.). It is not a PyTorch front-end: it does not read
PyTorch or any other existing framework's code, has no loops or
user-defined functions, and cannot express anything outside that fixed
operator set. What it does do for a `.tns` program written directly in
this DSL: statically check the computation, keep verified semantic
information in a typed graph IR, optimize that graph, and lower valid
programs through MLIR and LLVM for native execution.

**This is not an isolated toy — it interops with PyTorch today, at the
data boundary, to run genuine end-to-end programs.** Train a model in
PyTorch as usual, export its weights to `.npy` (one file per
input/weight), and load them into a compiled Tensora program with
`--data-dir` (see [docs/development/BUILD.md](docs/development/BUILD.md)).
Tensora then statically verifies, optimizes, and runs that exact
computation through MLIR/LLVM, and its result matches the PyTorch
reference within floating-point tolerance. This is not a synthetic
claim: ten demos under `demos/` are genuine, end-to-end, PyTorch-trained
programs running through Tensora this way — a tabular MLP, a CNN image
classifier, a fixed-grid CNN object detector (on synthetic shapes, then
on real annotated photographs), a single-head self-attention text
classifier, a 4-head self-attention associative-retrieval model, and
(going one step further) several demos built on one real self-attention
block extracted from a genuine, third-party pretrained Transformer
(`google/vit-base-patch16-224[-in21k]`) — checked against that model's
own output on a real photograph, verified independently at all 12
encoder layers, and, strongest of all, spliced into a **live** copy of
that model mid-inference to confirm its real-world prediction survives
unchanged. See
[Real-World Inference Demos](#real-world-inference-demos) below for the
full list, in order of difficulty. A
deeper, compile-time integration (for example registering Tensora as a
`torch.compile` backend, or converging with PyTorch's own MLIR path) is
design work, not yet built — see
[docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md](docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md)
§9.

The expected high-level pipeline is:

``` text
Tensora Source
      ↓
Lexer / Parser
      ↓
AST
      ↓
Semantic Analysis
      ↓
Typed Computation Graph
      ↓
Graph Optimization
      ↓
MLIR
      ↓
LLVM
      ↓
Native Execution
```

The implementation targets forward computation for inference.
Training, automatic differentiation, backpropagation, optimizers,
distributed training, and production serving possible future extensions.

------------------------------------------------------------------------

## 0. Current Repository State


**Identity.** Source files use the `.tns` extension. The compiler binary
is `tensora` (built at `build/tensora`).

**Repository architecture.** `include/tensora/` and `src/` are organized
by compiler phase, with cross-phase representations in one `shared/`
location. See [§7.6 Repository](docs/development/07_DEVELOPMENT_ENVIRONMENT.md#76-repository)
for the actual current tree.

**Compiler pipeline (as implemented):**

``` text
source (.tns)
  -> ANTLR lexer (Tensora.g4)      -> token stream
  -> ANTLR parser (Tensora.g4)     -> parse tree
  -> AstBuilder (src/parser/)      -> AST
  -> SemanticAnalyzer              -> Typed Computation Graph (TCG)
  -> Level-1 graph passes          -> optimized TCG
  -> MlirLowerer                   -> MLIR (func/linalg/tensor/arith/scf)
  -> Level-2 matmul tiling pass    -> tiled MLIR
  -> standard MLIR/LLVM pipeline   -> LLVM IR
  -> JIT execution or native object file
```

**One-time setup after cloning.** Install the system packages first
(`setup-dev.sh` deliberately does not `sudo`-install them), then run it
to verify the toolchain against
[docs/development/DEPENDENCIES.md](docs/development/DEPENDENCIES.md) and
print the next commands:

``` bash
sudo apt install -y build-essential cmake default-jre llvm-21-dev libmlir-21-dev mlir-21-tools
./tools/setup-dev.sh          # add --demos to also set up the PyTorch demo venv (large)
```

**Canonical build and test:**

``` bash
cmake -S . -B build                 # Configure project and generate build files
cmake --build build -j "$(nproc)"   # Compile and link, one job per core (default is 1 job -- serial)
ctest --test-dir build --output-on-failure # Run tests and show output for failed tests
```

(or `make build` / `make test`). First configure needs internet access:
CMake fetches the ANTLR jar and C++ runtime via `FetchContent`. See
[docs/development/BUILD.md](docs/development/BUILD.md) for the full
build/test commands, including building or testing one compiler phase
at a time (`make test-lexer`, `make test-parser`, etc.), and
[docs/development/07_DEVELOPMENT_ENVIRONMENT.md](docs/development/07_DEVELOPMENT_ENVIRONMENT.md)
for the toolchain policy (platform, compiler, LLVM/MLIR version,
warnings, debugging, git workflow).

# Tensora Phase Demos

There are six scripts: one per compiler phase, plus one that runs the
whole pipeline together. Each is a thin wrapper around the real,
tested artifacts — the `tensora` CLI driver
([`src/driver/main.cpp`](src/driver/main.cpp)) and the `test_passes`
CTest binary
([`tests/optimization/test_passes.cpp`](tests/optimization/test_passes.cpp))
— not a separate copy of any compiler logic.

**Demo commands** (one per phase; the table below has the full map):

``` bash
demos/01_lexer_demo.sh          # --emit-tokens
demos/02_parser_demo.sh         # --emit-parse-tree, --emit-ast, --emit-ast-dot
demos/03_semantic_demo.sh       # --check, --emit-symbols, --emit-graph, --emit-graph-dot, --explain
demos/04_optimization_demo.sh   # --emit-optimized-graph, the 6 Level-1 passes
demos/05_codegen_demo.sh        # --emit-mlir, --emit-llvm
demos/06_final_compiler_demo.sh # --run and -o across MLP / tiled matmul / CNN / attention
```

Run from the repository root after building (`cmake --build build`).
`make example` (alias `make examples`) runs all six in order; each is
also its own target (`make lexer-example` … `make codegen-example`,
`make final-example`). These six, plus the visualization/independent-
pipeline/benchmark demos just below, together make up
`make compiler-demos` — every demo that exercises the **compiler
itself** rather than a real-world trained model (that's
`make tensora-demos`, see [Real-World Inference
Demos](#real-world-inference-demos) below).

Each script accepts an optional `.tns` file as its first argument
(defaults to a sensible example under `examples/`) and an optional
build-directory path as its second argument (defaults to `build`).

| Script | Phase | What it runs |
|---|---|---|
| `01_lexer_demo.sh` | Lexer | `tensora --emit-tokens` |
| `02_parser_demo.sh` | Parser | `tensora --emit-parse-tree`, `--emit-ast`, `--emit-ast-dot` |
| `03_semantic_demo.sh` | Semantic analysis | `tensora --check`, `--emit-symbols`, `--emit-graph`, `--emit-graph-dot`, `--explain` |
| `04_optimization_demo.sh` | Level-1 graph optimization | `tensora --emit-optimized-graph`; `test_passes` on named pass cases (positive + near-miss) |
| `05_codegen_demo.sh` | Lowering + Level-2 tiling | `tensora --emit-mlir`, `--emit-llvm` |
| `06_final_compiler_demo.sh` | Full pipeline | `tensora --run` and `-o` across MLP / tiled matmul / CNN / attention |

You can inspect every intermediate form the compiler produces
directly, one flag per stage: tokens (`--emit-tokens`), the raw ANTLR
parse tree (`--emit-parse-tree`), the AST (`--emit-ast`/`--emit-ast-dot`),
the symbol table (`--emit-symbols`, RM 3.6 — global declarations, each
graph's local scope, and its name-resolution log), the Typed
Computation Graph before and after optimization
(`--emit-graph`/`--emit-graph-dot`, `--emit-optimized-graph`), MLIR
(`--emit-mlir`), and LLVM IR (`--emit-llvm`). This is useful for
debugging one phase, or for a demo, without running the whole
pipeline.

The `--polyhedral` flag is an optional lowering path: it runs `linalg`
through the MLIR affine dialect and one real polyhedral pass
(`affine-loop-tile`) instead of the default loop lowering. It gives the
same result on every runnable example (see
[docs/architecture/DECISIONS.md](docs/architecture/DECISIONS.md), ARCH-018);
it exists to prove Tensora's output can drive polyhedral optimization,
not to change what any program computes.

# Visualization, Independent Pipeline, and Benchmarks

Three more demos sit alongside the per-phase ones, still part of
`make compiler-demos` (none of them run a real-world trained model).
Each is its own `make` target too, and skips itself if its tools are
absent:

``` bash
make viz              # demos/08 -- render AST / TCG (before+after opt) / MLIR DAG to PNGs (needs Graphviz `dot`)
make demo-independent # demos/07 -- feed Tensora's emitted MLIR/LLVM IR to upstream mlir-opt/llc/lli, same math
make benchmark        # measure Level-2 matmul-tiling speedup (not a CTest; see 07_DEVELOPMENT_ENVIRONMENT.md)
```

`make viz` gives the clearest picture of optimization: the Typed
Computation Graph *before* and *after* the Level-1 passes, side by side.
`make demo-independent` is separate from the tests on purpose — the
CTest suite *asserts* correctness (e.g. `test_passes`, and the gated
`INT-IR-*-STD` cases that check emitted IR parses under external tools),
while these demos *visualize* it; both read the same
`examples/optimizations/*.tns`, and demo 04 ends by running the
authoritative pass tests rather than re-checking anything itself.

# Real-World Inference Demos

Ten demos (in order of difficulty, below) take recognizable real
inputs, use genuine PyTorch-trained parameters, run the whole
computation through Tensora (source → TCG → optimization → MLIR →
LLVM → native execution), and check the result against a PyTorch
reference. **Each needs its weights and inputs prepared once first**,
on a machine with PyTorch
([`tools/models/requirements.txt`](tools/models/requirements.txt)); the
demo runners themselves need only numpy. `make setup-demos` runs every
preparation script below in one command (needs internet access and
~350MB of disk, one time, for the ViT checkpoint):

``` bash
make setup-demos
```

which is equivalent to running these one time, each of which trains
(or, for the ViT demos, downloads) a model once and exports its
weights plus held-out inputs as `.npy` files under `demos/artifacts/`:

``` bash
python3 tools/models/prepare_mlp_demo.py            # tabular MLP
python3 tools/models/prepare_stl10_demo.py          # STL-10 CNN
python3 tools/models/prepare_grid_detector_demo.py  # fixed-grid CNN detector, synthetic shapes (no download)
python3 tools/models/prepare_grid_detector_real_demo.py # same detector family, real VOC photographs (needs internet)
python3 tools/models/prepare_text_attn_demo.py      # single-head attention text classifier
python3 tools/models/prepare_multihead_attn_demo.py # 4-head attention associative-retrieval model
python3 tools/models/prepare_vit_attention_demo.py  # downloads real pretrained ViT-Base (google/vit-base-patch16-224-in21k),
                                                     # extracts one real attention block's weights + hidden state
                                                     # (needs `pip install transformers safetensors pillow` in addition
                                                     # to tools/models/requirements.txt, and internet access once)
```

`demo-vit-attention-substitution` prepares (and downloads) its own
separate checkpoint on first run —
`google/vit-base-patch16-224` (a real ImageNet-1k *classifier*, unlike
the headless `-in21k` encoder above) — via
`tools/models/prepare_vit_substitution_demo.py`; it is not part of
`setup-demos` because, unlike every other demo here, it also needs a
built `tensora` binary at prep time (it shells out to it directly to
recompute one block), not only numpy.

`prepare_vit_attention_demo.py` also accepts two overrides, forwarded
through `make prep-vit-attention IMAGE=... LAYER=...` or set directly
as environment variables `TENSORA_VIT_IMAGE` (a local file path or an
http(s) URL) and `TENSORA_VIT_LAYER` (`0`-`11`), so trying a different
photo or encoder layer doesn't need editing the script.

## The demos, in order of difficulty

| # | `make` target | Data | Demonstrates |
|---|---|---|---|
| 1 | `demo-mlp-placement` | tabular, trained from scratch | simplest case: `linear`→`relu`→`linear` binary classifier (student placement eligibility) |
| 2 | `demo-cnn-image-classifier` | real photos (STL-10), trained from scratch | whole-image classification, 10 classes — "**what** is in this image" |
| 3 | `demo-cnn-grid-detector` | synthetic shapes, trained from scratch | fixed-grid object *localization* — "**what and where**" — over a toy 32x32 image; the minimal example this task family needs |
| 4 | `demo-cnn-grid-detector-real` | real photos (Pascal VOC 2012), trained from scratch | the same fixed-grid localization task, but on real, human-annotated photographs at finer (16x16) resolution — harder data, and honestly weaker precision (see below) |
| 5 | `demo-attn-text-classifier` | synthetic sentences, trained from scratch | single-head self-attention, topic classification |
| 6 | `demo-attn-retrieval` | synthetic sequences, trained from scratch | 4-head self-attention, rank-4 batched attention, associative key-value retrieval — a harder attention task than classification |
| 7 | `demo-vit-attention-one-layer` | real photo, real pretrained weights | one self-attention block (12-head, 768-dim) from a genuine pretrained ViT-Base checkpoint, checked against that model's own output |
| 8 | `demo-vit-attention-all-layers` | same as #7 | the same check, independently repeated at all 12 encoder layers (`PASS 12/12`) |
| 9 | `demo-vit-attention-viz` (optional) | same as #7 | renders the same block's attention weights as PNGs (needs `matplotlib`) |
| 10 | `demo-vit-attention-substitution` | 5 real photos, real pretrained weights | the strongest claim: splices Tensora's own recomputation of one block into a **live** real ImageNet-1k classifier mid-inference and shows the real-world prediction survives, on 5 visibly different photographs |

``` bash
make demo-mlp-placement
make demo-cnn-image-classifier
make demo-cnn-grid-detector
make demo-cnn-grid-detector-real
make demo-attn-text-classifier
make demo-attn-retrieval
make demo-vit-attention-one-layer
make demo-vit-attention-all-layers
make demo-vit-attention-viz
make demo-vit-attention-substitution
```

`make tensora-demos` runs all ten in this order; it skips (with a
message) any demo whose artifacts have not been prepared (or, for the
substitution demo, whose `transformers` dependency is not installed),
so it still succeeds on a fresh clone. `make demo` (alias `make demos`)
runs `compiler-demos` and `tensora-demos` together — every demo in the
repository, one command.

Real values reach the compiled graph through `tensora --run --data-dir`
(one float32 `.npy` per input/weight); see
[docs/development/BUILD.md](docs/development/BUILD.md).

### 1-2: tabular MLP, whole-image CNN classification

Straightforward feed-forward and convolutional classifiers, trained on
real (STL-10) or realistic tabular data — the baseline "does Tensora
run a real trained model correctly" case the harder demos build on.

### 3-4: fixed-grid object localization ("what AND where")

`demo-cnn-grid-detector` classifies every cell of a fixed 4x4 grid over
a 32x32 image containing 1-3 colored shapes (circle/square/triangle),
then draws a box + label at each non-background cell — the "what AND
where" complement to `demo-cnn-image-classifier`'s "what" (a single
whole-image label). It predicts a class per cell, not a continuous
`[x, y, w, h]` box (v1's `return` names exactly one tensor, so a
second, box-regression output would need a feature not yet in the
language — see `examples/cnn_grid_detector.tns`'s header comment), so
detected boxes sit at that cell's own fixed pixel boundary, not a
tightly-fit box around the shape. Because training data guarantees at
most one object per grid cell, no non-max suppression is needed either.

`demo-cnn-grid-detector-real` is the same architecture family and the
same design constraints, applied to real, human-annotated Pascal VOC
2012 photographs (person/car/dog/bicycle/bus) at a finer 16x16 grid
instead of drawn shapes at 4x4. Tensora still matches its own PyTorch
reference exactly (max prob error ~4e-7 over 6144 cells) — the
compiler is not in question — but the small, from-scratch CNN itself,
trained on only ~1000-1300 real photographs, has honestly weak
precision (recall ~49%, many false-positive cells on real, cluttered
backgrounds — often dozens of same-class regions per photograph). It
is included as-is, not polished away: the printed per-photo region
count and the recall figure below reflect every raw cell prediction,
unfiltered. The annotated PNG itself draws only the single
highest-confidence region per photograph (dozens of overlapping boxes
would be illegible), so a demo suite that only ever shows clean
results would be less useful than one that also shows what a small
model trained on little real data actually looks like — the honest
numbers are in the printed stats, not hidden by the picture.

### 5-6: self-attention

`demo-attn-text-classifier` is single-head self-attention over short
synthetic sentences (topic classification: weather/food/sports).
`demo-attn-retrieval` is a harder task on the same underlying
mechanism: 4-head, rank-4 batched attention doing associative
key-value retrieval across 6 shuffled slots — a deliberately different
task from the single-head demo (a reshuffled binding every example, so
it cannot be memorized), exercising batched matmul/transpose/sum at
rank 4 and a one-hot-selector read-out pattern in place of indexing.

### 7-10: a real pretrained Transformer (ViT-Base)

Unlike the demos above (which train their own small PyTorch model from
scratch to get realistic-but-synthetic parameters), demos 7-10 extract
weights and hidden states from a genuine, third-party, open-source
pretrained checkpoint rather than a small model trained for the demo.

`demo-vit-attention-one-layer` checks Tensora's compiled output for one
self-attention block against that same real pretrained model's own
output for the same block, on a real photograph.
`demo-vit-attention-all-layers` (`demos/vit_attention_all_layers_demo.sh`)
repeats that check independently at **all 12 encoder layers**: for
each layer it feeds that layer's own real input hidden state and real
pretrained weights through the real `tensora` CLI and compares against
that layer's own real output, printing a per-layer max-error table and
`PASS 12/12` — materially stronger than checking a single layer.

`make demo-vit-attention-viz` (optional, needs `matplotlib` in addition
to the packages above) goes one step further: it runs
[`examples/vit_attention_weights.tns`](examples/vit_attention_weights.tns)
— the same block, stopped right after `softmax` — through the real
`tensora` CLI, checks Tensora's own attention weights against the real
model's, and renders six PNGs under `demos/artifacts/vit_attn/`. Every
panel is bordered to show its provenance: a solid green
"TENSORA-VERIFIED" frame means the data came from Tensora's own
compiled/checked output; a dashed gray "reference only" frame means it
is real-model PyTorch data that Tensora did not compute.

- `attention_heatmap.png` — one panel per head, showing which patches
  of the real input photograph the [CLS] token attends to most **per
  head** — the concrete, visual reason multi-head attention exists,
  since different heads visibly attend to different parts of the same
  image — using Tensora's own compiled computation rather than a hand
  re-derivation.
- `attention_rollout.png` — a head-averaged view of the
  Tensora-verified layer next to a full **attention-rollout** trace
  across all 12 encoder layers (Abnar & Zuidema, 2020; computed
  directly from the real model's own per-layer attention, since Tensora
  compiles one block, not the whole chain — hence its dashed
  "reference only" frame). A single early layer's attention is
  genuinely noisy (it captures local texture, not semantics); the
  rollout is the standard technique for a legible summary of where the
  model's attention concentrates, and cleanly highlights the
  photographed subject.
- `attention_microscope.png` — the [CLS] row is only one slice of the
  full, Tensora-verified `[197, 197]` attention matrix. This figure
  inspects three patches at once — one on the dog's face, one on its
  body, one on the background, picked ahead of time from the image's
  own patch-grid colors, not cherry-picked after seeing the results —
  and for each patch shows the mean-over-heads "attends to" map next
  to three **individual heads'** "attends to" maps (the head with the
  lowest, middle, and highest [CLS]-attention entropy), plus the
  mean "attended back" column. The per-head columns matter: the
  mean-over-heads maps for all three patches look similar (a couple of
  globally-salient background patches dominate every average), but at
  least one individual head attends very differently depending on the
  query patch — a structure the mean alone hides. Pass a `"row,col"`
  argument (each `0`-`13`, since the image is a 14×14 grid of patches)
  to inspect one specific patch instead, e.g.
  `python3 tools/models/visualize_vit_attention.py build 6,7`.
- `attention_asymmetry.png` — attention is directional:
  `Attention(A → B) ≠ Attention(B → A)` in general. This figure finds
  the top-3 most asymmetric patch pairs in the Tensora-verified layer
  and draws each as a distinctly colored arrow with both directions'
  weights labeled directly on the image (thick = the stronger
  direction, thin = the weaker), rather than crowding the photo with
  every candidate pair.
- `attention_diagnostics.png` — two quantitative checks that don't
  depend on a human judging a picture: (a) per-head attention entropy,
  labeled `head 0`-`head 11` with each bar's numeric value (a focused
  head has low entropy, a diffuse one has high entropy — both computed
  straight from Tensora's own output), and (b) a
  real-model-vs-Tensora attention error map at a dramatically enlarged
  scale, so the "these match" claim is a picture (near-black, i.e. the
  two are numerically indistinguishable) rather than only the printed
  max-error number.
- `attention_topk_boxes.png` — the same "attends to" row from Figures 1
  and 3, but drawn without a heatmap: a colormap overlay with bilinear
  interpolation between 14x14 cells adds visual information the model
  didn't produce. This figure instead draws a box around exactly the
  top-5 attended patches, each labeled with its printed attention
  weight and nothing else — a literal picture of what the
  Tensora-verified matrix contains, not an interpolated impression of
  it. Pass a `"row,col"` argument for a different query patch, same as
  Figure 3.

`make demo-vit-attention-substitution`
(`demos/vit_attention_substitution_demo.sh`,
`tools/models/prepare_vit_substitution_demo.py`) is the strongest
practical-usefulness demo of the set. Every ViT demo above proves a
*numerical* claim in isolation ("Tensora's output tensor for this block
matches PyTorch's own output tensor for this block"). This one goes
further: it loads `google/vit-base-patch16-224` — a real, pretrained
**1000-way ImageNet classifier**, not the headless `-in21k` encoder the
other ViT demos use — runs each of **five visibly different real
photographs** (dog, car, elephant, airplane, cat) through it once
unmodified to get a baseline prediction, then **splices Tensora's own
recomputation of the last encoder layer's attention block into the
live model** (via a forward-hook monkey-patch) and runs the *same*
photograph through the *same* model again. Everything except that one
block — the patch-embedding convolution, the other 11 attention
blocks, every MLP block and layernorm, and the classifier head — is
the real, unmodified HuggingFace model. Result: on all 5 photographs,
the predicted label and confidence are unchanged (final-logit max abs
error ~1e-6, 11 encoder layers plus a classifier head downstream of
the substitution point). That is the difference between "Tensora's
numbers are numerically close" and "Tensora's compiled output is a
functional drop-in replacement inside a real production inference
pipeline." Also renders a visual dashboard (input photo, top-5
probability bars baseline-vs-substituted, a layer-substitution
schematic, and a numeric-error meter) per photograph plus a one-row
summary gallery, under `demos/artifacts/vit_substitution/*.png`, if
`matplotlib` is installed. This demo prepares its own checkpoint on
first run (separate from `setup-demos`, since it also needs a *built*
`tensora` binary at prep time, not only numpy) — see the script's
header for the one-time setup.

## Known limitations

- `--emit-mlir`/`--emit-llvm`/`--run`/`-o` need **concrete**
  (non-symbolic) shapes (see `docs/testing/TEST_MATRIX.md` LOW-005).
  Use `examples/mlp_concrete.tns`, not `examples/mlp.tns`, for those
  stages.
- The Transformer demos are a self-attention **classifier** and a
  self-attention **associative-retrieval** model, both built from
  Tensora's existing operators — a full pretrained BERT/DistilBERT QA
  model needs operators Tensora does not yet have (embedding/gather,
  layernorm, GELU, integer tensors), tracked in
  [docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md](docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md)
  §8.9a/§8.9.
- The only way to get data into a compiled Tensora program today is
  `.npy` files (`--data-dir`) — there is no in-memory way to hand
  Tensora a live tensor from another process. Making Tensora a
  first-class interop target (for example, a PyTorch custom op that
  passes tensors directly, no file round-trip) is needed for Tensora to
  be useful across wider use cases; see
  [docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md](docs/ProjectMgmt/11_OPEN_SOURCE_AND_RESEARCH_ROADMAP.md)
  §9.
