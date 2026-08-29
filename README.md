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
several demos built on one real self-attention block extracted from a
genuine, third-party pretrained Transformer
(`google/vit-base-patch16-224[-in21k]`) — checked against that model's
own output on a real photograph, verified independently at all 12
encoder layers, and spliced into a **live** copy of that model
mid-inference to confirm its real-world prediction survives unchanged.
See
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

# Compiler Demos

These demos exercise the **compiler itself** — no real-world trained
model involved (for that, see [Real-World Inference
Demos](#real-world-inference-demos) below). Together they make up
`make compiler-demos`. Each script is a thin wrapper around the real,
tested artifacts — the `tensora` CLI driver
([`src/driver/main.cpp`](src/driver/main.cpp)) and the `test_passes`
CTest binary
([`tests/optimization/test_passes.cpp`](tests/optimization/test_passes.cpp))
— not a separate copy of any compiler logic. Run from the repository
root after building (`cmake --build build`); each script accepts an
optional `.tns` file as its first argument (defaults to a sensible
example under `examples/`) and an optional build-directory path as its
second argument (defaults to `build`).

## One demo per phase

`make example` (alias `make examples`) runs all six below in order;
each is also its own target (`make lexer-example` … `make
codegen-example`, `make final-example`).

| Script | Phase | What it runs |
|---|---|---|
| `demos/01_lexer_demo.sh` | Lexer | `tensora --emit-tokens` |
| `demos/02_parser_demo.sh` | Parser | `tensora --emit-parse-tree`, `--emit-ast`, `--emit-ast-dot` |
| `demos/03_semantic_demo.sh` | Semantic analysis | `tensora --check`, `--emit-symbols`, `--emit-graph`, `--emit-graph-dot`, `--explain` |
| `demos/04_optimization_demo.sh` | Level-1 graph optimization | `tensora --emit-optimized-graph`; `test_passes` on named pass cases (positive + near-miss) |
| `demos/05_codegen_demo.sh` | Lowering + Level-2 tiling | `tensora --emit-mlir`, `--emit-llvm` |
| `demos/06_final_compiler_demo.sh` | Full pipeline | `tensora --run` and `-o` across MLP / tiled matmul / CNN / attention |

Each `--emit-*` flag above dumps one stage on its own (tokens, parse
tree, AST, symbol table, TCG before/after optimization, MLIR, LLVM IR)
— useful for debugging or demoing a single phase without running the
whole pipeline.

The `--polyhedral` flag is an optional lowering path: it runs `linalg`
through the MLIR affine dialect and one real polyhedral pass
(`affine-loop-tile`) instead of the default loop lowering. It gives the
same result on every runnable example (see
[docs/architecture/DECISIONS.md](docs/architecture/DECISIONS.md), ARCH-018);
it exists to prove Tensora's output can drive polyhedral optimization,
not to change what any program computes.

## Visualization, Independent Pipeline, and Benchmarks

Three more demos, still part of `make compiler-demos`. Each is its own
`make` target too, and skips itself if its tools are absent:

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
demo runners themselves need only numpy. `make setup-tensora-demos`
(alias `make setup-demos`) runs every preparation script below in one
command (needs internet access and ~350MB of disk, one time, for the
ViT checkpoint). It's safe to re-run: each script is skipped if its
`demos/artifacts/<name>/eval/` already exists, so re-running never
redoes a download/training step that already finished; delete a
specific demo's `demos/artifacts/<name>/` directory to force it to
redo that one demo from scratch:

``` bash
make setup-tensora-demos
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
`setup-tensora-demos` because, unlike every other demo here, it also
needs a built `tensora` binary at prep time (it shells out to it
directly to recompute one block), not only numpy.

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
| 10 | `demo-vit-attention-substitution` | 5 real photos, real pretrained weights | splices Tensora's own recomputation of one block into a **live** real ImageNet-1k classifier mid-inference and shows the real-world prediction survives, on 5 visibly different photographs |

``` bash
make demo-mlp-placement               # tabular MLP: linear->relu->linear binary classifier
make demo-cnn-image-classifier        # whole-image "what is it" classification (STL-10 photos)
make demo-cnn-grid-detector           # toy "what AND where": fixed-grid localization on synthetic shapes
make demo-cnn-grid-detector-real      # same grid localization, on real annotated VOC photographs
make demo-attn-text-classifier        # single-head self-attention topic classifier
make demo-attn-retrieval              # 4-head self-attention associative key-value retrieval
make demo-vit-attention-one-layer     # one real ViT-Base attention block vs. the real model's own output
make demo-vit-attention-all-layers    # the same check at all 12 encoder layers (PASS 12/12)
make demo-vit-attention-viz           # renders that block's attention weights as PNGs (needs matplotlib)
make demo-vit-attention-substitution  # splices Tensora's block into a live pretrained classifier and checks the prediction survives
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

### 3-4: object detection — "what AND where"

Plain classification (demos 1-2) only answers "what is in this image."
`demo-cnn-grid-detector` answers a harder question: "what, **and
where**." It splits a small 32x32 image into a 4x4 grid, predicts one
object class per cell (circle/square/triangle), and draws a labeled
box around every cell that isn't empty — a simplified version of how
early object-detector networks like YOLO work. Because each cell can
only hold one object by construction, there's no need for the
duplicate-box cleanup ("non-max suppression") real detectors need.

`demo-cnn-grid-detector-real` is the same idea on real photographs
instead of drawn shapes — real Pascal VOC 2012 street photos
(person/car/dog/bicycle/bus), at a finer 16x16 grid. Tensora's output
still matches PyTorch exactly, but the small model itself, trained on
only ~1,000 real photos, is honestly not very accurate (it finds a
real object only about half the time, with plenty of false alarms on
cluttered backgrounds). That's shown, not hidden: the printed stats
report the raw, unfiltered numbers, and the picture shows only the
model's single best guess per photo so it stays readable.

### 5-6: self-attention — the mechanism behind Transformers

`demo-attn-text-classifier` answers "what topic is this sentence
about?" (weather / food / sports) using single-head self-attention —
the same core building block inside every Transformer model (GPT,
BERT, ViT), just at toy scale.

`demo-attn-retrieval` is a harder task on that same mechanism: given a
shuffled list of key-value pairs, find the value for one requested key
— a lookup the model has to actually perform, not memorize, since the
shuffle changes every example. It uses 4 attention heads instead of 1,
closer to how real Transformers scale attention up.

### 7-10: a real pretrained Transformer (ViT-Base)

Demos 1-6 train their own small model from scratch. Demos 7-10 instead
borrow real, pretrained weights from a genuine, publicly available
Transformer — Google's ViT-Base image model — and check that Tensora
computes exactly what that real model computes.

`demo-vit-attention-one-layer` recomputes one real self-attention block
from ViT-Base and checks it against that model's own output, on a real
photograph. `demo-vit-attention-all-layers` repeats that check
independently for **all 12 layers** of the model (`PASS 12/12`), not
just one cherry-picked layer.

`make demo-vit-attention-viz` (optional, needs `matplotlib`) renders
six pictures of what that attention block is actually looking at in
the photo, each labeled to show whether Tensora computed it or it's
reference-only PyTorch data.

`make demo-vit-attention-substitution` puts the ViT demos together: it
takes a full, live, pretrained ImageNet classifier, swaps one of its
internal attention blocks for Tensora's own computation of that block
mid-inference, and runs 5 real photographs (dog, car, elephant,
airplane, cat) through it. On all 5, the prediction is unchanged
(final-logit error ~1e-6) — Tensora's output works as a drop-in
replacement for that block inside a real model, not only a numerical
match in isolation. This demo prepares its own checkpoint on first run
(separate from `setup-tensora-demos`, since it also needs a *built*
`tensora` binary at prep time, not only numpy) — see the script's
header for the one-time setup.

For a deeper explanation of each demo's output and design choices
(e.g. why the grid detectors don't do box regression or
non-max-suppression, or the exact provenance and meaning of each
`vit_attn/*.png` figure), see ["Reading the real-world demo output in
depth"](docs/development/BUILD.md#reading-the-real-world-demo-output-in-depth)
in `docs/development/BUILD.md`.

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
