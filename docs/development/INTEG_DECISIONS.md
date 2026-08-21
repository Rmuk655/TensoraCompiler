# Integration Decisions (INTEG-XXX)

## INTEG-001 — Canonical platform: WSL2 + Ubuntu

**Problem.** Where do all team members build and run Tensora, so results
are reproducible and the frontend-generator (and later LLVM/MLIR)
toolchain is easy to install?

**Options.**
- (a) Windows-native (MSVC).
- (b) WSL2 + Ubuntu (or native Linux).
- (c) Per-developer choice.

**Chosen approach.** (b) WSL2 + Ubuntu, frozen team-wide.

**Reason.** A JRE (for the ANTLR generator, INTEG-004), CMake, and
LLVM/MLIR are all first-class on Linux, and install with one `apt`
command. A Windows-native build fights the MLIR setup. One canonical
platform also makes the reproducibility record (compiler, LLVM, OS,
commit) mean something.

**Affected tools.** `docs/development/07_DEVELOPMENT_ENVIRONMENT.md` (§7.0),
`docs/development/BUILD.md`, CI. No change to `CMakeLists.txt` — CMake
already targets this toolchain and generates the Unix Makefiles.

**Migration.** Copy the repo into the WSL native filesystem (e.g.
`~/tensora`, ext4), not `/mnt/c/...`; build there. Keep `build/` out of git
and OneDrive (`.gitignore`). Normalize line endings to LF (`.gitattributes`).

---

## INTEG-002 — Bump toolchain from C++17 to C++20

**Problem.** `INTERFACES.md` §2 freezes the Parser boundary as
`Parser(std::span<const Token>, DiagnosticEngine&)`, but `std::span` is
a C++20 library feature. §7.1 of this document (and `CMakeLists.txt`)
had frozen the toolchain at C++17, which does not compile this. The
team found this direct contradiction, between an architecture/interface
document and the integration/build spec, while building the parser
(Phase 2).

**Options.**
- (a) Bump `CMAKE_CXX_STANDARD` to 20; keep `INTERFACES.md`'s Parser
  signature exactly as frozen.
- (b) Keep C++17; amend `INTERFACES.md` to replace `std::span` with
  `const std::vector<Token>&` (or a hand-rolled span-lite type).

**Chosen approach.** (a) Bump to C++20.

**Reason.** Under the project's documented authority order, an approved
architecture/interface document outranks the integration/build spec
when the two conflict. `INTERFACES.md`'s Parser signature is otherwise
clear, and the frozen design already uses it elsewhere. The bump is a
pure toolchain change with no language or semantic impact: GCC 15 (the
pinned compiler, INTEG-001) fully supports C++20, and nothing else in
the codebase depends on a C++17-only feature that C++20 removes.

**Affected tools.** `CMakeLists.txt` (`CMAKE_CXX_STANDARD 20`),
`docs/development/07_DEVELOPMENT_ENVIRONMENT.md` §7.1.

**Migration.** No source changes are needed beyond the parser itself,
which already assumed C++20 (`std::span`). No existing lexer code used
a C++20-incompatible construct.

---

## INTEG-003 — Pin LLVM/MLIR 21 (apt packages, not a from-source build)

**Problem.** Phase 4 (Code Generation) needs MLIR/LLVM C++ development
headers and CMake config files. Only the LLVM *runtime* library and
`clang` were present; no `llvm-config`, no `MLIRConfig.cmake`/
`LLVMConfig.cmake`.

**Options.**
- (a) `apt install llvm-21-dev libmlir-21-dev mlir-21-tools` (Ubuntu
  universe repo has 17 through 22 available).
- (b) Build LLVM/MLIR from source with `-DLLVM_ENABLE_PROJECTS=mlir`
  (hours, tens of GB, and a from-source toolchain no team member could
  reproduce without the exact same build flags).

**Chosen approach.** (a) apt packages, version 21 (1:21.1.8-6ubuntu1),
installed by the user with `sudo apt install llvm-21-dev libmlir-21-dev
mlir-21-tools`.

**Reason.** A pinned, package-manager-installed toolchain is exactly
what INTEG-001 already committed to: reproducibility through `apt`, no
from-source builds. LLVM 21 is new enough to have the full
`linalg`/`tensor`/`scf`/`func`/`arith` dialects and the
transform-dialect infrastructure Level-2 tiling needs (ARCH-009). With
no specific incompatibility, there is no reason to pin an older
version.

**Affected tools.** `CMakeLists.txt` (`find_package(MLIR)`/
`find_package(LLVM)`, `CMAKE_PREFIX_PATH` pointed at
`/usr/lib/llvm-21/lib/cmake/{llvm,mlir}`),
`docs/development/07_DEVELOPMENT_ENVIRONMENT.md` §7.3.

**Reproducibility record.** `llvm-config-21 --version` -> `21.1.8`;
packages `llvm-21-dev`, `libmlir-21-dev`, `mlir-21-tools` all
`1:21.1.8-6ubuntu1`; installed under WSL2 + Ubuntu (INTEG-001).

**Migration.** None. This is a new capability (`lowering/`), not a
change to existing frontend or sema code. That code has no MLIR/LLVM
dependency (ARCHITECTURE.md §2), so it stays unaffected.

---

## INTEG-004 — ANTLR frontend generator toolchain

**Problem.** ARCH-011 uses ANTLR4 for the frontend. What does that need
from the build and integration environment, and how does it stay
reproducible, with no machine-specific install path?

**Options.**
- (a) Require a system-wide ANTLR install (jar under a fixed path, C++
  runtime built and installed by each developer).
- (b) Fetch the ANTLR jar and C++ runtime reproducibly via CMake
  `FetchContent`, requiring only a system Java runtime.

**Chosen approach.** (b). `CMakeLists.txt` uses `find_package(Java)` to
locate `java` on `PATH`, then `FetchContent` to download the ANTLR jar
(`antlr-${ANTLR4_VERSION}-complete.jar`) and the ANTLR C++ runtime source
into the build tree, building the runtime as part of the normal CMake
build (`antlr4_static`).

**Reason.** A fixed jar path or a pre-built runtime would break "avoid
machine-specific paths" and "make ANTLR generation reproducible." Needing
only `java` (a one-line `apt install default-jre`) keeps the toolchain
install a single, simple step.

**Affected tools.** `CMakeLists.txt` (`find_package(Java)`, two
`FetchContent_Declare`/`FetchContent_MakeAvailable` calls, the
`antlr4_generate` custom command), `docs/development/BUILD.md`,
`docs/development/07_DEVELOPMENT_ENVIRONMENT.md` §7.2.

**Reproducibility record.** ANTLR pinned at `4.13.2` (`ANTLR4_VERSION` in
`CMakeLists.txt`); first configure needs internet access to fetch the jar
and runtime source, same as any other `FetchContent` dependency.

**Migration.** Requires `default-jre` (or any JRE) installed via `apt`;
no other developer action needed -- `cmake -S . -B build` fetches
everything else.
