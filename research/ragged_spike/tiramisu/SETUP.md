# Polyhedral setup (folder-local, does not touch project setup)

This folder tests the Tiramisu/polyhedral *hypothesis*. Two levels:

## 1. islpy (used here) — the real ISL polyhedral engine

Tiramisu, PolyDL, and Pluto are all built on **ISL** (the Integer Set
Library). `islpy` is its Python binding, pip-installable, so we can run
real polyhedral operations (iteration domains, cardinality, dependence)
without building Tiramisu.

```bash
source .venv/bin/activate      # the project's existing venv
pip install islpy              # ~2.5 MB, no system deps
python research/ragged_spike/tiramisu/islpy_ragged.py
```

This is not "simulated" polyhedral math — it is the same library the
polyhedral compilers use.

## 2. Real Tiramisu (not built here) — why

Tiramisu itself is a heavy C++ source build: it needs ISL, a pinned
LLVM, Halide, and Google benchmark, built via its own submodule/CMake
flow. That is a 30-60 min build with real chance of version breakage on
this laptop, and it is not needed to test the hypothesis (islpy gives us
the polyhedral engine directly). If a real Tiramisu comparison is wanted
later, build it in a throwaway dir outside this repo, following
`Tiramisu-Compiler/tiramisu` README, and record the result here. It is
deferred by cost, not by principle.
