# Convenience wrapper around CMake. CMake is the authoritative build system;
# these targets just save typing. Run on WSL2 + Ubuntu (see docs/development).
BUILD ?= build
# Parallel build jobs. The vendored ANTLR4 C++ runtime alone is ~100
# independent .cpp files; building it (or anything else here) with the
# implicit default of one job at a time wastes every idle core on a
# multi-core machine. Override with `make build JOBS=1` if needed.
JOBS ?= $(shell nproc 2>/dev/null || echo 4)

.PHONY: build test example clean appendix benchmark \
        lexer parser semantic optimization codegen full \
        test-lexer test-parser test-semantic test-optimization test-codegen \
        lexer-example parser-example semantic-example optimization-example codegen-example \
        demo-mlp-placement demo-cnn-image-classifier demo-attn-text-classifier

configure:
	cmake -S . -B $(BUILD)

build: configure
	cmake --build $(BUILD) -j$(JOBS)

test: build
	ctest --test-dir $(BUILD) --output-on-failure

# The `tensora` CLI is one binary with one flag per phase (--emit-
# tokens, --emit-ast, --check, --emit-mlir, --run, ...) -- there is no
# phase-specific executable to build here, unlike `lexer`/`parser`/etc.
# above, so every target below depends on the full `build`, and each
# just runs the matching demos/*.sh script (the existing "one demo per
# phase" mechanism -- see README.md's Phase Demos section) instead of
# duplicating its logic here.
example: build
	./demos/06_final_compiler_demo.sh

lexer-example: build
	./demos/01_lexer_demo.sh

parser-example: build
	./demos/02_parser_demo.sh

semantic-example: build
	./demos/03_semantic_demo.sh

optimization-example: build
	./demos/04_optimization_demo.sh

codegen-example: build
	./demos/05_codegen_demo.sh

clean:
	rm -rf $(BUILD)

# Real-world inference demos (docs/development/BUILD.md). Each runs a
# model trained/prepared once in PyTorch on real data through the full
# Tensora pipeline and checks the result against the PyTorch reference.
# Prepare the artifacts first (needs PyTorch, one time):
#   python3 tools/models/prepare_mlp_demo.py
#   python3 tools/models/prepare_cifar_demo.py
#   python3 tools/models/prepare_text_attn_demo.py
# The demo runners themselves need only numpy (no torch).
demo-mlp-placement: build
	./demos/mlp_placement_demo.sh $(BUILD)

demo-cnn-image-classifier: build
	./demos/cnn_image_classifier_demo.sh $(BUILD)

demo-attn-text-classifier: build
	./demos/attn_text_classifier_demo.sh $(BUILD)

# tools/benchmark_matmul_tiling.cpp measures matmul tiling speedup; it
# is a plain CMake executable, not a CTest case (see docs/development/
# 07_DEVELOPMENT_ENVIRONMENT.md §7.10), so `make test` never runs it.
benchmark: build
	./$(BUILD)/benchmark_matmul_tiling

# Regenerate docs/development/10_APPENDIX_SOURCE_LISTING.md from the
# current source tree (docs/development/10_APPENDIX_SOURCE_LISTING.md
# §10.10). A doc-maintenance step, run by hand after a source change --
# not part of `build`/`test`, since it edits a committed file and
# should not fire on every build.
appendix:
	./tools/regen_appendix.py

# Build one compiler phase at a time (docs/development/BUILD.md "Build
# one phase at a time"). Each depends only on the CMake targets that
# phase actually needs, so this does not build the whole compiler.
# "codegen" here means the MLIR/LLVM backend phase; the underlying
# CMake target is named `backend` because CMake reserves the name
# "codegen" for its own CODEGEN custom-command feature.
lexer: configure
	cmake --build $(BUILD) --target lexer -j$(JOBS)

parser: configure
	cmake --build $(BUILD) --target parser -j$(JOBS)

semantic: configure
	cmake --build $(BUILD) --target semantic -j$(JOBS)

optimization: configure
	cmake --build $(BUILD) --target optimization -j$(JOBS)

codegen: configure
	cmake --build $(BUILD) --target backend -j$(JOBS)

full: configure
	cmake --build $(BUILD) --target full -j$(JOBS)

# Build, then run, only that phase's tests (`ctest -L <phase>`).
test-lexer: lexer
	ctest --test-dir $(BUILD) --output-on-failure -L lexer

test-parser: parser
	ctest --test-dir $(BUILD) --output-on-failure -L parser

test-semantic: semantic
	ctest --test-dir $(BUILD) --output-on-failure -L semantic

test-optimization: optimization
	ctest --test-dir $(BUILD) --output-on-failure -L optimization

test-codegen: codegen
	ctest --test-dir $(BUILD) --output-on-failure -L codegen
