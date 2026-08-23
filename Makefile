# Convenience wrapper around CMake. CMake is the authoritative build system;
# these targets just save typing. Run on WSL2 + Ubuntu (see docs/development).
BUILD ?= build
# Parallel build jobs. The vendored ANTLR4 C++ runtime alone is ~100
# independent .cpp files; building it (or anything else here) with the
# implicit default of one job at a time wastes every idle core on a
# multi-core machine. Override with `make build JOBS=1` if needed.
JOBS ?= $(shell nproc 2>/dev/null || echo 4)

.PHONY: build test example examples demo demos clean appendix benchmark viz demo-independent \
        lexer parser semantic optimization codegen full \
        test-lexer test-parser test-semantic test-optimization test-codegen \
        lexer-example parser-example semantic-example optimization-example codegen-example \
        final-example demo-mlp-placement demo-cnn-image-classifier demo-attn-text-classifier

# Configure only when there is no build tree yet. CMake's generated
# build system re-runs configure by itself when CMakeLists.txt changes
# (the Makefile generator injects that rule), so calling `cmake -S -B`
# on every `make` just reprints the whole configure log for nothing.
configure:
	@test -f $(BUILD)/CMakeCache.txt || cmake -S . -B $(BUILD)

build: configure
	cmake --build $(BUILD) -j$(JOBS)

# Build once, quietly, for the aggregate demo/example targets: run the
# normal build but show its output only if something actually compiled or
# linked (or the build failed). When the tree is already up to date this
# collapses to a single line instead of reprinting the target list. The
# plain `make build` above stays verbose; this is only for the aggregates,
# which would otherwise reprint the build for each demo they run.
define QUIET_BUILD
	@echo "==> build (once)"; log=$$(mktemp); \
	if $(MAKE) --no-print-directory build >$$log 2>&1; then \
	  if grep -qE 'Building|Linking' $$log; then cat $$log; else echo "    already up to date"; fi; \
	  rm -f $$log; \
	else cat $$log; rm -f $$log; exit 1; fi
endef

test: build
	ctest --test-dir $(BUILD) --output-on-failure

# The `tensora` CLI is one binary with one flag per phase (--emit-
# tokens, --emit-ast, --check, --emit-mlir, --run, ...) -- there is no
# phase-specific executable to build here, unlike `lexer`/`parser`/etc.
# above, so every target below depends on the full `build`, and each
# just runs the matching demos/*.sh script (the existing "one demo per
# phase" mechanism -- see README.md's Phase Demos section) instead of
# duplicating its logic here.
# `make example` (alias `examples`) runs ALL phase example demos in
# order -- source text walked through every phase on the bundled
# examples/*.tns, ending at the full-pipeline capstone (demo 06). Each
# phase's demo can still be run on its own via the `*-example` targets
# below. (`make demo` is the other axis: the real-world / showcase
# demos, not the phase walkthrough -- see further down.)
example examples:
	$(QUIET_BUILD)
	./demos/01_lexer_demo.sh
	./demos/02_parser_demo.sh
	./demos/03_semantic_demo.sh
	./demos/04_optimization_demo.sh
	./demos/05_codegen_demo.sh
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

# The full-pipeline capstone on its own (the last step of `make example`).
final-example: build
	./demos/06_final_compiler_demo.sh

# `make demo` (alias `demos`) runs ALL showcase demos: the benchmark,
# the independent-pipeline demo, the visualization demo, and the three
# real-world inference demos. The real-world demos need a one-time
# PyTorch artifact-prep step (see demo-* targets below); this aggregate
# SKIPS each one whose demos/artifacts/<name>/ is absent, with a clear
# message, so `make demo` still succeeds on a fresh clone. The
# visualization demo is skipped if Graphviz (`dot`) is not installed.
# (`make example` is the other axis: the per-phase walkthrough.)
demo demos:
	$(QUIET_BUILD)
	@echo "==> benchmark";            ./$(BUILD)/benchmark_matmul_tiling
	@echo "==> independent pipeline"; ./demos/07_independent_run_demo.sh $(BUILD)
	@if command -v dot >/dev/null 2>&1; then \
	  echo "==> visualization"; ./demos/08_viz_demo.sh $${FILE:-examples/optimizations/dead_code.tns} $(BUILD); \
	else echo "SKIP viz: Graphviz not found (sudo apt install graphviz)"; fi
	@for pair in mlp:mlp_placement cnn:cnn_image_classifier attn:attn_text_classifier; do \
	  name=$${pair%%:*}; script=$${pair##*:}; \
	  if [ -d "demos/artifacts/$$name" ]; then \
	    echo "==> demo ($$script)"; ./demos/$${script}_demo.sh $(BUILD); \
	  else \
	    echo "SKIP $$script: no artifacts in demos/artifacts/$$name/ (run the matching tools/models/prepare_*.py first)"; \
	  fi; \
	done

clean:
	rm -rf $(BUILD)

# Real-world inference demos (docs/development/BUILD.md). Each runs a
# model trained/prepared once in PyTorch on real data through the full
# Tensora pipeline and checks the result against the PyTorch reference.
# Prepare the artifacts first (needs PyTorch, one time):
#   python3 tools/models/prepare_mlp_demo.py
#   python3 tools/models/prepare_stl10_demo.py
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

# Independent-pipeline demo: upstream mlir-opt/llvm tools consume Tensora's
# emitted IR (a different IR world than PyTorch's), on a real-data matmul.
demo-independent: build
	./demos/07_independent_run_demo.sh $(BUILD)

# Render Tensora's graph views (AST, TCG before/after optimization, MLIR DAG)
# to PNGs. Needs Graphviz (`dot`): sudo apt install graphviz.
viz: build
	./demos/08_viz_demo.sh $${FILE:-examples/optimizations/dead_code.tns} $(BUILD)

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
