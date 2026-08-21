# Dependencies

C++ has no single `requirements.txt` equivalent, because a C++
project's dependencies do not come from one ecosystem. Tensora's
dependencies come from four different places, so this page lists all
four, and `tools/check-deps.sh` checks a machine against this list.
Run it any time to see what is present and what is missing, with the
exact command to fix each gap:

```bash
./tools/check-deps.sh
```

## 1. OS packages (required)

Installed with `apt` on the canonical platform (WSL2 + Ubuntu, or
native Linux; docs/development/07_DEVELOPMENT_ENVIRONMENT.md §7.0):

| Package           | Why                                               | Pinned version                          |
| ----------------- | ------------------------------------------------- | --------------------------------------- |
| `build-essential` | `g++`, a C++20 compiler (§7.1)                    | none (any recent GCC)                   |
| `cmake`           | the build system itself                           | none (any recent CMake; 4.2.3 verified) |
| `default-jre`     | runs the ANTLR generator jar CMake fetches (§7.2) | none                                    |

## 2. LLVM/MLIR (required, version-pinned)

**Pinned: 21.1.8**, installed via apt packages, not built from source
(INTEG-003 in `docs/development/INTEG_DECISIONS.md`; §7.3 has the full
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

## What `tools/check-deps.sh` actually checks

Exits non-zero if any of these is missing: `cmake`, a C++20-capable
`g++`, `java`, `llvm-config-21`, the MLIR 21 CMake package,
`libmlir_c_runner_utils.so`.

`tools/setup-dev.sh` runs the same check and then prints the build/test
commands to run next.
