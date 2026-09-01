<!-- STRUCTURE TEMPLATE -- not the appendix itself.

tools/regen_appendix.py (make appendix) reads this file and fills
every empty code fence from the source tree to produce
10_APPENDIX_SOURCE_LISTING.md, which is generated, gitignored, and
never edited by hand.

This file holds only what cannot be derived from the source tree:
the prose, the Section 10.3-10.8 grouping, and the "### `path`"
headers saying which file belongs in which section. To add, remove,
or move a file in the appendix, edit those headers here.

This comment is stripped from the generated output.
-->
# 10. Appendix: Complete Source Listing

## 10.1 Purpose

The final report requires the complete source listing of the compiler.
This appendix is generated mechanically (concatenation of the actual
repository files, not hand-transcribed) so it cannot silently diverge
from what was built and tested.

## 10.2 Repository Manifest

```text
```

## 10.3 Frontend Listing (`grammar/`, lexer/parser headers and sources)

### `grammar/Tensora.g4`
```text
```

### `include/tensora/lexer/Token.h`
```text
```

### `include/tensora/lexer/TokenAdapter.h`
```text
```

### `include/tensora/lexer/Lexer.h`
```text
```

### `src/lexer/Lexer.cpp`
```text
```

### `src/lexer/TokenAdapter.cpp`
```text
```

### `include/tensora/shared/Ast.h`
```text
```

### `include/tensora/parser/Parser.h`
```text
```

### `src/parser/Parser.cpp`
```text
```

### `src/parser/AstBuilder.h`
```text
```

### `src/parser/AstBuilder.cpp`
```text
```

### `src/parser/ParserErrorListener.h`
```text
```

### `src/parser/ParserErrorListener.cpp`
```text
```

### `src/parser/ParserErrorStrategy.h`
```text
```

### `src/parser/ParserErrorStrategy.cpp`
```text
```

### `include/tensora/shared/SourceSpan.h`
```text
```

### `include/tensora/shared/SourceBuffer.h`
```text
```

### `include/tensora/shared/Result.h`
```text
```

### `include/tensora/shared/Diagnostic.h`
```text
```

## 10.4 Semantic Listing (symbol table, type system, contracts, `SemanticAnalyzer`)

### `include/tensora/semantic/Symbol.h`
```text
```

### `src/semantic/Symbol.cpp`
```text
```

### `include/tensora/shared/Type.h`
```text
```

### `src/shared/Type.cpp`
```text
```

### `include/tensora/shared/DimSolver.h`
```text
```

### `src/shared/DimSolver.cpp`
```text
```

### `include/tensora/semantic/Contracts.h`
```text
```

### `src/semantic/Contracts.cpp`
```text
```

### `include/tensora/semantic/SemanticAnalyzer.h`
```text
```

### `src/semantic/SemanticAnalyzer.cpp`
```text
```

## 10.5 Typed Graph Listing (node/value definitions, builder, verifier)

### `include/tensora/shared/Ir.h`
```text
```

### `src/shared/Ir.cpp`
```text
```

## 10.6 Optimization Listing (Level-1 graph passes, liveness analysis)

### `include/tensora/optimization/GraphPass.h`
```text
```

### `src/optimization/GraphPasses.cpp`
```text
```

### `include/tensora/optimization/Liveness.h`
```text
```

### `src/optimization/Liveness.cpp`
```text
```

## 10.7 Codegen Listing (Tensora-to-MLIR lowering, Level-2 tiling pass, backend)

### `include/tensora/codegen/DialectSetup.h`
```text
```

### `src/codegen/DialectSetup.cpp`
```text
```

### `include/tensora/codegen/MlirLowerer.h`
```text
```

### `src/codegen/MlirLowerer.cpp`
```text
```

### `include/tensora/codegen/LoopTiling.h`
```text
```

### `src/codegen/LoopTiling.cpp`
```text
```

### `include/tensora/codegen/MatmulTilingPass.h`
```text
```

### `src/codegen/MatmulTilingPass.cpp`
```text
```

### `tools/benchmark_matmul_tiling.cpp`
```text
```

### `tools/codegen_mlp_native.cpp`
```text
```

## 10.8 Driver and Build Listing (CLI driver, example programs, CMake)

### `src/driver/main.cpp`
```text
```

### `src/driver/Npy.h`
```text
```

### `src/driver/Npy.cpp`
```text
```

### `examples/mlp.tns`
```text
```

### `examples/mlp_concrete.tns`
```text
```

### `examples/matmul_large.tns`
```text
```

### `examples/cnn.tns`
```text
```

### `examples/attention.tns`
```text
```

### `examples/mlp_placement.tns`
```text
```

### `examples/cnn_stl10.tns`
```text
```

### `examples/attn_text_classifier.tns`
```text
```

### `examples/attn_relational_eligibility.tns`
```text
```

### `examples/cnn_grid_detector.tns`
```text
```

### `examples/cnn_grid_detector_real.tns`
```text
```

### `examples/transformer_ffn.tns`
```text
```

### `examples/transformer_ffn_concrete.tns`
```text
```

### `examples/vit_attention_block.tns`
```text
```

### `examples/vit_attention_weights.tns`
```text
```

### `examples/optimizations/dead_code.tns`
```text
```

### `examples/optimizations/transpose_cancel.tns`
```text
```

### `examples/optimizations/reshape_cancel.tns`
```text
```

### `examples/optimizations/shape_fold.tns`
```text
```

### `examples/optimizations/linear_bias_activation.tns`
```text
```

### `examples/optimizations/elementwise_fusion.tns`
```text
```

### `cmake/Tests.cmake`
```text
```

### `CMakeLists.txt`
```text
```

### `tools/setup-dev.sh`
```text
```

### `tools/check-deps.sh`
```text
```

### `tools/check_test_matrix.sh`
```text
```

### `tools/check_discovery.sh`
```text
```

### `tools/check_doc_link_targets.py`
```text
```

### `tools/regen_appendix.py`
```text
```

### `tools/models/requirements.txt`
```text
```

### `tools/models/prepare_mlp_demo.py`
```text
```

### `tools/models/prepare_stl10_demo.py`
```text
```

### `tools/models/prepare_text_attn_demo.py`
```text
```

### `tools/models/prepare_relational_demo.py`
```text
```

### `tools/models/prepare_grid_detector_demo.py`
```text
```

### `tools/models/prepare_grid_detector_real_demo.py`
```text
```

### `tools/models/prepare_vit_attention_demo.py`
```text
```

### `tools/models/prepare_vit_substitution_demo.py`
```text
```

### `tools/models/visualize_vit_attention.py`
```text
```

### `tools/models/visualize_vit_substitution.py`
```text
```

### `tools/models/compare_yolo_baseline.py`
```text
```

## 10.9 Test Listing

```text
```

### `tests/optimization/test_passes.cpp` (representative Level-1 pass test)
```text
```

### `tests/codegen/test_cnn_e2e.cpp` (representative MLIR lowering JIT e2e test)
```text
```

### `tests/codegen/test_tiling_e2e.cpp` (Level-2 loop-tiling test)
```text
```

## 10.10 Mechanical Generation

This listing is generated by `tools/regen_appendix.py` (`make
appendix`), which reads `docs/development/appendix_template.md` -- this
same document with every code fence left empty -- and fills each fence
from the file's current content on disk, in a fenced code block tagged
with the language matching its extension (for example ` ```cpp ` for
`.cpp`/`.h`, ` ```bash ` for `.sh`, ` ```cmake ` for `CMakeLists.txt`).
The Section 10.2 and 10.9 manifests are generated the same way, from
`find grammar include src tools examples -type f | sort` and `find
tests -type f | sort`. Nothing here is hand-transcribed, so it cannot
silently diverge from what was actually built and tested. The appendix
itself is not committed: it is rebuilt on demand from the template,
which is.

This repository is under git version control, hosted at
[github.com/Rmuk655/TensoraCompiler](https://github.com/Rmuk655/TensoraCompiler)
(`origin`, branch `main`). Base commit at generation time:
`3b5a167b28fdc3868292fb283871808b9fe79d18`. The concatenation itself
reflects the working tree at generation time, which may include
changes made after that commit and not yet committed -- the manifest
and file contents above are the authoritative record either way, since
they are read directly from disk, not from git history. Generation
date: 2026-09-01. The 192 CTest cases passing at generation time are
the authoritative record of what this listing implements; see
[docs/testing/TEST_MATRIX.md](../testing/TEST_MATRIX.md) and
[docs/architecture/DECISIONS.md](../architecture/DECISIONS.md#implementation-timeline).
