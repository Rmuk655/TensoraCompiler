# Building Tensora

This is the canonical reference for the actual build/test/run commands.
[docs/development/07_DEVELOPMENT_ENVIRONMENT.md](07_DEVELOPMENT_ENVIRONMENT.md)
covers environment policy and process (why WSL2, why C++20, warnings
policy, git workflow, debugging, benchmarking) and links back here
instead of repeating these commands.

## Canonical platform: WSL2 + Ubuntu (INTEG-001)

Build and run on **WSL2 running Ubuntu** (or native Linux). Two steps:

```bash
# 1. System packages. This needs sudo; setup-dev.sh deliberately does not
#    sudo-install these (see DEPENDENCIES.md, "check-deps.sh vs setup-dev.sh").
sudo apt update && sudo apt install -y build-essential cmake default-jre \
    llvm-21-dev libmlir-21-dev mlir-21-tools

# 2. Verify the toolchain and print the next commands to run.
./tools/setup-dev.sh          # add --demos to also install the PyTorch demo tooling (large)
```

[`setup-dev.sh`](../../tools/setup-dev.sh) runs
[`check-deps.sh`](../../tools/check-deps.sh) (run that on its own any time
to audit a machine) and then prints the build/test commands. See
[DEPENDENCIES.md](DEPENDENCIES.md) for the full pinned-version reference
(including why LLVM/MLIR is pinned to 21) and what each dependency is for.

**Important:** work inside the WSL native filesystem. Copy the repo
to, for example, `~/tensora` (ext4), and build there. Do **not** build
from `/mnt/c/.../OneDrive/...`. `/mnt/c` is slow, and OneDrive fights
the generated `build/` directory. This is why `build/` is git-ignored.

macOS may work (`brew install cmake openjdk`, plus an LLVM/MLIR 21
install), but it is **not tested** — WSL2 + Ubuntu is the only supported
platform (INTEG-001). Windows-native is not supported either.

## Build everything

```bash
cmake -S . -B build                     # Configure project and generate build files
cmake --build build -j "$(nproc)"       # Compile and link, one job per core (default is 1 job -- serial)
```

`cmake --build` never parallelizes on its own with the Makefiles
generator (CMake leaves that decision to you, since full-core
parallelism compiling MLIR/LLVM headers can pressure memory on a small
machine) -- so every raw `cmake --build` command needs its own `-j`, or
export a default once per shell so you never have to remember it:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"   # add to ~/.bashrc to make it permanent
```

With that set, a bare `cmake --build build` (or `--target <phase>`,
below) is parallel automatically. `make`'s targets already pass
`-j$(nproc)` themselves, so this variable does not change them.

Or use the convenience wrapper (same thing, less typing):

```bash
make build          # configure + build everything
make test           # build + run the full ctest suite
make example        # build + run every per-phase demo in order (demos/01..06)
make compiler-demos # example, plus viz/independent/benchmark -- every demo about the compiler itself
make tensora-demos  # every real-world inference demo (genuine PyTorch-trained models through Tensora)
make demo           # compiler-demos + tensora-demos together, one command
make clean
```

`make example` is the per-phase walkthrough; `make compiler-demos`
widens that to every demo exercising the compiler itself (adding
visualization/independent-pipeline/benchmark); `make tensora-demos` is
the real-world set; `make demo` (alias `make demos`) runs both groups.
Every individual target is listed in "Make target reference" at the
end of this file.

## Build one phase at a time

`CMakeLists.txt` defines one CMake target per compiler phase, each
depending only on what that phase needs — building `lexer`, for
example, does not build the MLIR/LLVM backend.

These need an already-configured build tree, so run `cmake -S . -B build`
once first (in particular right after `make clean`, which deletes
`build/` — otherwise `cmake --build build` reports "build is not a
directory"). The `make` wrappers below configure automatically, so they
work straight after a clean.

These also need `-j` for parallel building (or `export
CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"` once, per "Build everything"
above, so you never have to pass it) — `cmake --build` compiles
**serially** by default otherwise, which is noticeably slow. (The flag
is `-j` or `--parallel`, e.g. `-j "$(nproc)"`, `-j4`, or bare `-j` for
all cores; `--j4` is not valid. The `make` wrappers below already pass
`-j$(nproc)`, so they do not need this.)

```bash
cmake -S . -B build                         # once (or after `make clean`)
cmake --build build --target lexer          # lexer only
cmake --build build --target parser         # parser only
cmake --build build --target semantic       # semantic analysis only
cmake --build build --target optimization   # the 6 Level-1 passes + liveness
cmake --build build --target backend        # MLIR/LLVM codegen + tensora CLI
cmake --build build --target full           # everything (same set `make build` builds)
```

("backend", not "codegen": CMake reserves the target name `codegen`
for its own `CODEGEN` custom-command feature, so the MLIR/LLVM backend
phase target is named `backend` instead.)

Or with the `make` wrapper, which also configures first:

```bash
make lexer         # configure + build test_lexer only
make parser        # configure + build test_parser only
make semantic      # configure + build test_semantic + test_graph only
make optimization  # configure + build test_passes only
make codegen       # configure + build the MLIR/LLVM backend + tensora CLI (the `backend` CMake target)
make full          # configure + build everything (same set `make build` builds)
```

Every CTest case also carries a phase label matching this same
breakdown (see "Run the tests" below), so a phase can be tested in
isolation the same way it is built in isolation.

## Run the tests

```bash
ctest --test-dir build --output-on-failure # Run tests and show output for failed tests
```

Filter to one phase with `-L <label>` (`lexer`, `parser`, `semantic`,
`optimization`, `codegen`), or to one test-kind with `-L <label>`
(`unit`, `boundary`, `smoke`, `integration`):

```bash
ctest --test-dir build --output-on-failure -L semantic
```

Or with the `make` wrapper, which builds only that phase first:

```bash
make test-lexer         # build + ctest -L lexer
make test-parser        # build + ctest -L parser
make test-semantic      # build + ctest -L semantic
make test-optimization  # build + ctest -L optimization
make test-codegen       # build + ctest -L codegen
```

## Documentation link checks

This project has no CI configured — nothing here runs automatically on
push or pull request; you run these yourself. Every `[label](target)`
cross-reference in `docs/*.md` is checked two different ways:

- **Does the link resolve at all** (file exists, `#anchor` exists)?
  Run by hand with `make docs` (below); it needs Node/npm (already on
  most dev machines) but no permanent global install, since it runs
  the tool (`markdown-link-check`) through `npx`.
- **Does the link land on the section its own label claims** (for
  example, does `[RM 3.6](...)` actually land on RM section 3.6, and
  not on 3.7 by a copy-paste mistake)? Checked automatically as the
  `INT-DOCLINKS-SYNC` test that `ctest`/`make test` already runs
  (`tools/check_doc_link_targets.py`). The only requirement is
  `python3` on `PATH` -- already the same optional dependency
  `./tools/check-deps.sh` reports for the real-world demos. If
  `python3` is missing, CMake skips just this one test instead of
  failing the whole suite.

Run both checks together, without a full compiler build first, with:

```bash
make docs
```

## Run the lexer on an example

```bash
./build/tensora --emit-tokens examples/mlp.tns
```

Each printed line is `line:col  kind` (it also shows the text for
identifiers and literals). The compiler reports an illegal character
on stderr and sets a non-zero exit code.
[docs/language/02_LANGUAGE_TUTORIAL.md](../language/02_LANGUAGE_TUTORIAL.md)
§2.12 lists every other `--emit-*` flag, one per phase.

`make example` runs every per-phase demo (01…06) in order. For a single
phase's demo (source in, phase output out — the same `demos/*.sh`
scripts README.md's Compiler Demos section describes), use the matching
`-example` target instead:

```bash
make lexer-example         # demos/01_lexer_demo.sh
make parser-example        # demos/02_parser_demo.sh
make semantic-example      # demos/03_semantic_demo.sh
make optimization-example  # demos/04_optimization_demo.sh
make codegen-example       # demos/05_codegen_demo.sh
make final-example         # demos/06_final_compiler_demo.sh (full-pipeline capstone)
```

## Run inference with real data

By default `--run` fills inputs and weights with synthetic all-ones data
(Tensora source declares shapes, not values). To run with real trained
parameters and real inputs, put one float32 `.npy` file per declared
input/weight, named `<name>.npy`, in a directory and pass `--data-dir`;
`--output` writes the full result tensor to a `.npy` file:

```bash
./build/tensora --run examples/mlp_placement.tns \
    --data-dir path/to/data \      # loads data/x.npy, data/W1.npy, ...
    --output   result.npy          # writes the full result tensor
```

Each file's shape and dtype are checked against the program's declared
tensor; a mismatch or missing file is a clear error. This is how the
real-world demos below feed genuine weights and inputs.

## Optional: the `--polyhedral` lowering path

By default, `--run`/`-o`/`--emit-llvm` lower `linalg` straight to `scf`
loops. Pass `--polyhedral` to lower `linalg` through the MLIR **affine**
dialect instead, and run one real polyhedral pass (`affine-loop-tile`) on
it before continuing to LLVM (ARCH-018):

```bash
./build/tensora --polyhedral --run examples/mlp_concrete.tns
```

This flag changes only how the program is compiled, not what it
computes: on every runnable example, `--polyhedral --run` gives the exact
same result as the plain `--run` (checked by the `INT-POLY-*` tests,
`tests/cli/cli_tests.txt`). It exists to prove Tensora's output can drive
a real polyhedral optimization, and to record one polyhedral pass
(`affine-loop-fusion`) that gave a wrong result on a larger example — a
bug in upstream MLIR, filed against LLVM/MLIR, not a Tensora defect. See
`research/polyhedral_spike/` on the `v0.3-research` branch for the full
finding.

## Run the real-world inference demos

Eleven demos take recognizable real inputs through the full pipeline and
check the result against a PyTorch reference; **README.md's
[Real-World Inference Demos](../../README.md#real-world-inference-demos)
section is the single canonical listing** of all eleven (in order of
difficulty, with what each one demonstrates) -- not repeated here.
**Each needs its artifacts prepared once first, using PyTorch** --
running `make demo-*` before this step fails with "demo artifacts not
found," telling you to do this. Modern Debian/Ubuntu block `pip
install` against the system Python (PEP 668), so the demo dependencies
go in a project-local virtual environment, not the system Python and
never `pip --break-system-packages`. `./tools/setup-dev.sh --demos`
creates `.venv/` and installs `tools/models/requirements.txt` into it:

```bash
./tools/setup-dev.sh --demos     # one time: creates .venv, installs torch etc.
source .venv/bin/activate        # 'which python' -> .venv/bin/python
make setup-tensora-demos          # runs every tools/models/prepare_*.py script once (alias: setup-demos)
deactivate                       # when finished
```

Safe to re-run: each prepare script is skipped (no re-download,
re-training, or other network/compute work) if its
`demos/artifacts/<name>/eval/` already exists -- both when run via
`make setup-tensora-demos` and when run directly (e.g.
`python3 tools/models/prepare_stl10_demo.py`). Delete one demo's
`demos/artifacts/<name>/` directory, or pass `--force` to that
script directly, to redo just that one demo's preparation from
scratch.

Artifacts land in `demos/artifacts/`
(git-ignored, reproduced on demand). The demo runners themselves need
only numpy, so `make demo-*` works without the venv active once the
artifacts exist. Each demo classifies a whole held-out batch in one run
and reports Tensora-vs-PyTorch agreement over the whole batch, showing
a few examples in detail. The CNN prep caches its trained model at
`demos/artifacts/cnn-model.pt` (the slow step); delete that file to
retrain from scratch.

## Reading the real-world demo output in depth

README.md's [Real-World Inference
Demos](../../README.md#real-world-inference-demos) section is the
concise, canonical overview -- what problem each demo solves and what
technique it uses. This section is the deeper reference for readers
who want to interpret one demo's *output* in more detail once it's
running: design choices, exact provenance of every figure, and what
each printed/rendered artifact means.

**`demo-cnn-grid-detector` / `demo-cnn-grid-detector-real`.** Both
predict one object class per grid cell (no continuous box regression,
and no duplicate-box cleanup since each cell holds at most one
object), so detected boxes sit at the cell's fixed pixel boundary, not
a tightly-fit box. `demo-cnn-grid-detector-real`'s printed stats report
every raw per-cell prediction, unfiltered (recall ~49% on cluttered
real backgrounds); the annotated PNG draws only the single
highest-confidence region per photo so it stays readable -- the honest
numbers are always in the printed stats, never hidden by the picture.

**`demo-attn-relational`.** A two-hop relational-reasoning task, not a
single lookup, over real natural-language sentences, not a symbol
table: given up to 3 completed-course sentences and, for each of 6
candidate courses (drawn per example from a 24-course pool, with
randomized student names, phrasing, and distractor facts), a sentence
stating its single prerequisite, the model must find the one course
the student is now eligible for -- a course that is never named as
"eligible" anywhere in the input. Every sentence is turned into a
float vector by a from-scratch word-embedding + mean-pool encoder,
shared across all sentence roles, trained entirely in PyTorch (v1
Tensora has no embedding/gather/tokenizer op, so it never sees a
token, only the resulting vectors). That takes two chained steps: hop
1 is ordinary single-head softmax attention (Q@Kt/softmax/A@V, with a
fixed *learned* query rather than a per-example one) reading the
completed-course sentences into one "profile" vector; hop 2 has no
attention op at all -- it projects that profile two ways and matmuls
each projection against every candidate course's prerequisite-sentence
encoding and its own identity-sentence encoding, producing one "is the
prerequisite covered?" and one "is this course already done?" score
per course, combined with two learned scalars (plain elementwise
`*`/`+`, since v1 Tensora has no stack/concat op to feed a combining
MLP instead). The printed output reports two different numbers, and
they mean different things: "Tensora vs PyTorch" is the usual
exact-agreement check (this demo's whole point, same as every other
demo here); "Eligibility accuracy" is a *separate* number -- how often
the trained PyTorch model itself gets the right answer, honestly
around 55-60% (well above the 1/6 ~ 17% chance rate, so real two-hop
fact-chaining over real sentences is happening, but clearly short of a
clean answer, since this is a genuinely harder task than a single
lookup, and mean-pooled sentence encodings dilute the signal further).
A less-than-perfect eligibility accuracy is not a bug in Tensora, or in
this demo -- it is what running an imperfect model correctly looks
like; run `tools/models/prepare_relational_demo.py` to see the exact
current figure.

**`demo-vit-attention`.** One combined demo, three parts, all against
one real pretrained ViT-Base checkpoint. Part 1 checks Tensora's
compiled output for one self-attention block against that same real
model's own output for the same block, on a real photograph. Part 2
repeats that check independently at all 12 encoder layers: for each
layer it feeds that layer's own real input hidden state and real
pretrained weights through the real `tensora` CLI and compares against
that layer's own real output, printing a per-layer max-error table and
`PASS 12/12`. Part 3 (optional, needs `matplotlib`) runs
`examples/vit_attention_weights.tns` -- the same block, stopped right
after `softmax` -- through the real `tensora` CLI, checks Tensora's own
attention weights against the real model's, and renders six PNGs under
`demos/artifacts/vit_attn/`. Every panel is bordered to show its
provenance: a solid green "TENSORA-VERIFIED" frame means the data came
from Tensora's own compiled/checked output; a dashed gray "reference
only" frame means it is real-model PyTorch data that Tensora did not
compute.

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

**`demo-vit-attention-substitution`**
(`demos/vit_attention_substitution_demo.sh`,
`tools/models/prepare_vit_substitution_demo.py`) loads
`google/vit-base-patch16-224` — a real, pretrained 1000-way ImageNet
classifier, not the headless `-in21k` encoder the other ViT demos use
— runs each of five visibly different real photographs (dog, car,
elephant, airplane, cat) through it once unmodified to get a baseline
prediction, then splices Tensora's own recomputation of the last
encoder layer's attention block into the live model (via a
forward-hook monkey-patch) and runs the same photograph through the
same model again. Everything except that one block — the
patch-embedding convolution, the other 11 attention blocks, every MLP
block and layernorm, and the classifier head — is the real, unmodified
HuggingFace model. Also renders a visual dashboard (input photo, top-5
probability bars baseline-vs-substituted, a layer-substitution
schematic, and a numeric-error meter) per photograph plus a one-row
summary gallery, under `demos/artifacts/vit_substitution/*.png`, if
`matplotlib` is installed.

## Visualization, independent pipeline, and benchmark

```bash
make viz              # render AST / TCG (before + after optimization) / MLIR DAG to PNGs
make demo-independent # feed Tensora's emitted MLIR/LLVM IR to upstream mlir-opt/llc/lli
make benchmark        # measure Level-2 matmul-tiling speedup
```

`make viz` needs Graphviz (`sudo apt install graphviz`) and is the
clearest view of what optimization does — the Typed Computation Graph
before and after the Level-1 passes, for **one** program at a time
(default `examples/optimizations/dead_code.tns`; `FILE=<path> make viz`
picks a different one, including any other file under
`examples/optimizations/`). `make benchmark` is a measurement, not a
CTest (see
[07_DEVELOPMENT_ENVIRONMENT.md](07_DEVELOPMENT_ENVIRONMENT.md) §7.10) —
it does not read a `.tns` file at all; it runs a hardcoded 1024x1024x1024
matmul (the same workload `LOOP-002`/`test_tiling_e2e.cpp` use) untiled
and at three tile sizes, timing each, to show Level-2 tiling is actually
faster, not just structurally different.

`make viz` and `make benchmark` **show**; the CTest suite **asserts** —
but they are not both checking the same artifact `make viz` renders.
`test_passes` (the Level-1 pass CTest) uses small graphs built directly
in C++ (`GraphBuilder`, see `tests/optimization/test_passes.cpp`), not
the `.tns` files under `examples/optimizations/` — it exists to assert
the *same six passes* `make viz` lets you *see* fire on a real program,
not to check that specific file. The gated `INT-IR-MLIR-STD`/
`INT-IR-LLVM-STD` cases run on `examples/mlp_concrete.tns`, a different
file again, to confirm Tensora's emitted MLIR/LLVM IR is standard enough
for the real `mlir-opt`/`llvm-as` tools to accept — a check about IR
validity, not about optimization correctness.

In plain terms, each one answers a different question:

- **`test_passes`**: "Does each optimization pass do the right thing?"
  For every pass it builds a small graph *by hand* (not from a `.tns`
  file) where the pass *should* fire, checks the result, then builds a
  second "near-miss" graph where it should *not* fire, and checks the
  pass left it alone. Both halves matter — a pass that fires too rarely
  is useless, and one that fires too eagerly is a correctness bug.
- **`INT-IR-MLIR-STD` / `INT-IR-LLVM-STD`**: "Is Tensora's output real
  MLIR/LLVM, or something that only Tensora itself can read?" Each
  compiles `mlp_concrete.tns` and feeds the result to the actual,
  unmodified `mlir-opt`/`llvm-as` tools from LLVM — tools Tensora did not
  write — and checks they accept it without complaint.
- **`make viz`**: "What did an optimization pass actually change?" It
  draws a picture of one program's graph before and after the Level-1
  passes run, so you can see with your own eyes which operations were
  removed or rewritten, instead of just being told a pass "fired."
- **`make benchmark`**: "Is the tiled version of matmul actually
  *faster*, not just structured differently?" It runs the same
  1024x1024x1024 matrix multiply four ways (untiled, plus three tile
  sizes) and times each one for a real answer, rather than assuming
  tiling helps.

## Notes / known build caveats

- **Generated frontend.** ANTLR generates `TensoraLexer`/`TensoraParser`
  from the one combined `grammar/Tensora.g4` into `build/antlr4_generated/`
  on every build (the `antlr4_generate` target); nothing generated is
  committed or hand-edited. This step is fast.
- **First build is slow, once.** `FetchContent` pulls the pinned ANTLR jar
  and C++ runtime on first configure (needs internet), then compiling that
  ~100-file runtime dominates a from-scratch build. It is cached per
  `build/` dir, so incremental builds skip it; `make` passes `-j$(nproc)`
  to parallelize it.
- **Progress % is whole-graph.** A from-scratch build's percentage covers
  every target including the vendored runtime, so it can jump to ~67%
  before any Tensora code compiles — normal, not a fault. Build one phase
  at a time for a percentage scoped to that phase.

See [07_DEVELOPMENT_ENVIRONMENT.md](07_DEVELOPMENT_ENVIRONMENT.md) for the
rationale behind each of these.

## Make target reference

Every convenience target (`make <target>`); CMake commands above are the
authoritative equivalents.

| Target | What it does |
|---|---|
| `build` / `test` / `clean` | configure+build / build+ctest / remove `build/` |
| `lexer` `parser` `semantic` `optimization` `codegen` `full` | build one phase's targets |
| `test-lexer` … `test-codegen` | build a phase, then `ctest -L <phase>` |
| `example` (`examples`) | run all per-phase demos (01…06) in order |
| `lexer-example` … `codegen-example`, `final-example` | one phase demo (01…06) |
| `compiler-demos` | `example`, plus `viz`/`demo-independent`/`benchmark` -- every demo about the compiler itself |
| `tensora-demos` | run all 10 real-world inference demos (skips unprepared/absent-tool ones) |
| `demo` (`demos`) | `compiler-demos` + `tensora-demos` together -- every demo, one command |
| `demo-mlp-placement`, `demo-cnn-image-classifier`, ... `demo-vit-attention-substitution` | one real-world inference demo (10 total; README.md's Real-World Inference Demos section lists all of them in order) |
| `setup-tensora-demos` (`setup-demos`) | prepare every real-world demo's weights/inputs in one command |
| `viz` | render AST/TCG/MLIR graphs to PNGs (needs Graphviz) |
| `demo-independent` | run emitted IR through upstream mlir-opt/llc/lli |
| `benchmark` | measure matmul-tiling speedup |
| `appendix` | regenerate the source-listing appendix |
| `docs` | check every `docs/*.md` cross-reference: link existence (`npx markdown-link-check`) + landing on the right section (`tools/check_doc_link_targets.py`) |

## Regenerate the source-listing appendix

[docs/development/10_APPENDIX_SOURCE_LISTING.md](10_APPENDIX_SOURCE_LISTING.md)
is a mechanical concatenation of the real source tree, not a
hand-maintained document (see its own Section 10.10). After changing a
file it already lists, run:

```bash
make appendix
```

(or [`./tools/regen_appendix.py`](../../tools/regen_appendix.py)
directly). This is a doc-maintenance step, run by hand when needed --
it is not part of `make build`/`make test`, since it rewrites a
committed file and should not fire on every build.

The script only refreshes dumps of files it already has an entry for;
it does not decide where a *new* file's dump goes. Sections 10.3-10.8
group files by compiler phase (a curated breakdown, not something a
manifest diff can infer), so if you add a new source file, add its
`### \`path\`` entry to the right section by hand once, then re-run
the script -- it will keep that entry's content in sync from then on.
The script's own output tells you exactly which files still need this
(a new file with no entry yet, or an entry whose file was deleted).
