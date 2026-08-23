# Dependencies

C++ has no single `requirements.txt` equivalent, because a C++
project's dependencies do not come from one ecosystem. Tensora's
dependencies come from four different places, so this page lists all
four, and [`tools/check-deps.sh`](../../tools/check-deps.sh) checks a machine against this list.
Run it any time to see what is present and what is missing, with the
exact command to fix each gap:

```bash
./tools/check-deps.sh
```

## 1. OS packages (required)

Installed with `apt` on the canonical platform (WSL2 + Ubuntu, or
native Linux; [docs/development/07_DEVELOPMENT_ENVIRONMENT.md](07_DEVELOPMENT_ENVIRONMENT.md) §7.0):

| Package           | Why                                               | Pinned version                          |
| ----------------- | ------------------------------------------------- | --------------------------------------- |
| `build-essential` | `g++`, a C++20 compiler (§7.1)                    | none (any recent GCC)                   |
| `cmake`           | the build system itself                           | none (any recent CMake; 4.2.3 verified) |
| `default-jre`     | runs the ANTLR generator jar CMake fetches (§7.2) | none                                    |

## 2. LLVM/MLIR (required, version-pinned)

**Pinned: 21.1.8**, installed via apt packages, not built from source
(INTEG-003 in [`docs/development/INTEG_DECISIONS.md`](INTEG_DECISIONS.md); [§7.3](07_DEVELOPMENT_ENVIRONMENT.md#73-compiler-infrastructure) has the full
rationale and reproducibility record):

```bash
sudo apt install -y llvm-21-dev libmlir-21-dev mlir-21-tools
```

This is the one dependency where the exact version matters: MLIR's C++
API is not stable across versions, and the project's codegen code is
written against 21.1.8 specifically.

## 3. ANTLR (not an OS package -- do not install it system-wide)

The ANTLR jar and its C++ runtime are fetched reproducibly by CMake's
own `FetchContent` on first configure (`ANTLR4_VERSION` in
`CMakeLists.txt`, currently `4.13.2`), the same way a language-specific
package manager would pin a dependency. Nothing to install by hand;
first `cmake -S . -B build` needs internet access once to fetch it.

## 4. Python (optional -- real-world demo tooling only)

The compiler has no Python runtime dependency. The real-world inference
demos (`make demo-*`, `docs/development/BUILD.md`) do use Python, split
in two:

- **Preparation** (`tools/models/prepare_*.py`, run once): trains the
  models and produces the trusted PyTorch reference. Needs the packages
  in [`tools/models/requirements.txt`](../../tools/models/requirements.txt) (`torch`, `torchvision`, `numpy`,
  `pandas`, `matplotlib`). Install them in a **project-local virtual
  environment**, not the system Python: modern Debian/Ubuntu block
  `pip install` against the system Python (PEP 668,
  `externally-managed-environment`), and `pip --break-system-packages`
  must not be used for this project. `./tools/setup-dev.sh --demos`
  creates `.venv/` and installs them there; or by hand
  (`sudo apt install python3-venv` if needed):

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r tools/models/requirements.txt
  ```
- **Running** (`demos/*_demo.sh`): needs only `numpy` (to compare
  Tensora's output against the saved reference) -- no `torch`, and the
  venv need not be active. The `make viz` demo additionally needs
  Graphviz for the `dot` renderer, an OS package: `sudo apt install
  graphviz`.

[`tools/check-deps.sh`](../../tools/check-deps.sh) does *report* these
optional demo dependencies (Graphviz, the LLVM/MLIR CLI tools,
Python/venv) as advisory `[opt]` lines, but they never affect its exit
code -- the compiler and its own test suite never need them.

## check-deps.sh vs setup-dev.sh

Two scripts, one primitive and one workflow -- not a duplication:

- [`tools/check-deps.sh`](../../tools/check-deps.sh) is the **diagnostic**.
  It only *reports* (present, missing, and the exact fix command) and
  installs nothing, so it is safe to run repeatedly and in CI. It exits
  non-zero if any *required build* dependency is missing (`cmake`, a
  C++20-capable `g++`, `java`, `llvm-config-21`, the MLIR 21 CMake
  package, `libmlir_c_runner_utils.so`); the optional `[opt]` lines above
  never gate that exit code.
- [`tools/setup-dev.sh`](../../tools/setup-dev.sh) is the **workflow** on
  top of it. It runs `check-deps.sh`, and when the required deps are
  present prints the build/test commands to run next; with `--demos` it
  also creates `.venv/` and installs the Python demo deps into it. It
  deliberately does **not** `sudo apt install` the system/LLVM packages --
  installing OS packages on a machine it does not own is the user's call,
  so those it only reports (via `check-deps.sh`). This is why the check is
  a separate script: one source of truth for "what is needed", called both
  standalone and by the bootstrap.

Run `check-deps.sh` any time to audit a machine; run `setup-dev.sh` once
after cloning.
