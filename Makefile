# Convenience wrapper around CMake. CMake is the authoritative build system;
# these targets just save typing. Run on WSL2 + Ubuntu (see docs/development).
BUILD ?= build
# Parallel build jobs. The vendored ANTLR4 C++ runtime alone is ~100
# independent .cpp files; building it (or anything else here) with the
# implicit default of one job at a time wastes every idle core on a
# multi-core machine. Override with `make build JOBS=1` if needed.
JOBS ?= $(shell nproc 2>/dev/null || echo 4)

.PHONY: build test example examples demo demos compiler-demos tensora-demos clean appendix benchmark viz demo-independent \
        lexer parser semantic optimization codegen full \
        test-lexer test-parser test-semantic test-optimization test-codegen \
        lexer-example parser-example semantic-example optimization-example codegen-example \
        final-example demo-mlp-placement demo-cnn-image-classifier demo-attn-text-classifier \
        demo-attn-retrieval demo-cnn-grid-detector demo-cnn-grid-detector-real demo-vit-attention-one-layer demo-vit-attention-all-layers demo-vit-attention-viz \
        demo-vit-attention-substitution setup-demos setup-tensora-demos prep-vit-attention

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
# phase" mechanism -- see README.md's Compiler Demos section) instead of
# duplicating its logic here.
# `make example` (alias `examples`) runs ALL phase example demos in
# order -- source text walked through every phase on the bundled
# examples/*.tns, ending at the full-pipeline capstone (demo 06). Each
# phase's demo can still be run on its own via the `*-example` targets
# below. This is one piece of the broader `make compiler-demos` group
# (every demo about the COMPILER ITSELF); `make tensora-demos` is the
# other axis, real-world models run THROUGH the compiler -- see
# further down.
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

# `make compiler-demos` runs every demo about the COMPILER ITSELF: the
# six per-phase walkthroughs above (`make example`), then the
# benchmark, the independent-pipeline demo, and the visualization demo
# -- none of these run a real-world trained model, they exercise the
# compiler's own phases/tooling. `make tensora-demos` (below) is the
# other group: real-world models run THROUGH Tensora. The
# visualization demo is skipped if Graphviz (`dot`) is not installed.
compiler-demos: example
	@echo "==> benchmark";            ./$(BUILD)/benchmark_matmul_tiling
	@echo "==> independent pipeline"; ./demos/07_independent_run_demo.sh $(BUILD)
	@if command -v dot >/dev/null 2>&1; then \
	  echo "==> visualization"; ./demos/08_viz_demo.sh $${FILE:-examples/optimizations/dead_code.tns} $(BUILD); \
	else echo "SKIP viz: Graphviz not found (sudo apt install graphviz)"; fi

# `make tensora-demos` runs every real-world inference demo (README.md's
# Real-World Inference Demos section lists them in order, with what
# each one demonstrates) -- genuine PyTorch-trained models run through
# Tensora, as opposed to `make compiler-demos`'s tests of the compiler
# itself. Each needs a one-time PyTorch artifact-prep step (see demo-*
# targets below, or `make setup-tensora-demos`); this aggregate SKIPS each one
# whose demos/artifacts/<name>/ is absent, with a clear message, so it
# still succeeds on a fresh clone.
tensora-demos:
	$(QUIET_BUILD)
	@for pair in mlp:mlp_placement cnn:cnn_image_classifier grid_detector:cnn_grid_detector grid_detector_real:cnn_grid_detector_real attn:attn_text_classifier attn_mh:attn_retrieval vit_attn:vit_attention_one_layer; do \
	  name=$${pair%%:*}; script=$${pair##*:}; \
	  if [ -d "demos/artifacts/$$name" ]; then \
	    echo "==> demo ($$script)"; ./demos/$${script}_demo.sh $(BUILD); \
	  else \
	    echo "SKIP $$script: no artifacts in demos/artifacts/$$name/ (run the matching tools/models/prepare_*.py first)"; \
	  fi; \
	done
	@if [ -d "demos/artifacts/vit_attn/eval_all_layers" ]; then \
	  echo "==> demo (vit_attention_all_layers)"; ./demos/vit_attention_all_layers_demo.sh $(BUILD); \
	else \
	  echo "SKIP vit_attention_all_layers: no artifacts in demos/artifacts/vit_attn/eval_all_layers/ (run tools/models/prepare_vit_attention_demo.py first)"; \
	fi
	@if python3 -c "import transformers" >/dev/null 2>&1; then \
	  echo "==> demo (vit_attention_substitution)"; ./demos/vit_attention_substitution_demo.sh $(BUILD); \
	else \
	  echo "SKIP vit_attention_substitution: transformers not installed (pip install transformers safetensors pillow)"; \
	fi

# `make demo` (alias `demos`) is the historical full aggregate: both
# groups above, one command. Prefer `make compiler-demos` or
# `make tensora-demos` directly for a more focused run.
demo demos: compiler-demos tensora-demos

clean:
	rm -rf $(BUILD)

# Real-world inference demos: full descriptions, order, and what each
# one demonstrates are in README.md's "Real-World Inference Demos"
# section (the single canonical listing -- not repeated here). Prepare
# the artifacts first (needs PyTorch, one time); `make setup-tensora-demos`
# (alias `setup-demos`) below runs every preparation script in one
# command -- named to match `tensora-demos`, since this only prepares
# those, not `compiler-demos` (which needs no preparation). The demo
# runners themselves need only numpy (no torch).
setup-demos setup-tensora-demos:
	pip install -q -r tools/models/requirements.txt
	python3 tools/models/prepare_mlp_demo.py
	python3 tools/models/prepare_stl10_demo.py
	python3 tools/models/prepare_grid_detector_demo.py
	python3 tools/models/prepare_grid_detector_real_demo.py
	python3 tools/models/prepare_text_attn_demo.py
	python3 tools/models/prepare_multihead_attn_demo.py
	python3 tools/models/prepare_vit_attention_demo.py
	@echo "==> setup-tensora-demos done. Try: make demo-vit-attention-viz"

# Re-run just the ViT preparation step with a different photo and/or
# encoder layer, e.g.:
#   make prep-vit-attention IMAGE=my-cat.jpg LAYER=8
# IMAGE may be a local file path or an http(s) URL; LAYER is 0..11.
# Defaults to the original dog photo and layer 11 if not given.
prep-vit-attention:
	@if [ -n "$(IMAGE)" ]; then export TENSORA_VIT_IMAGE="$(IMAGE)"; fi; \
	 if [ -n "$(LAYER)" ]; then export TENSORA_VIT_LAYER="$(LAYER)"; fi; \
	 python3 tools/models/prepare_vit_attention_demo.py

# Demos below run in order of difficulty (README.md's Real-World
# Inference Demos section explains what each one demonstrates; not
# repeated here).
demo-mlp-placement: build
	./demos/mlp_placement_demo.sh $(BUILD)

demo-cnn-image-classifier: build
	./demos/cnn_image_classifier_demo.sh $(BUILD)

demo-cnn-grid-detector: build
	./demos/cnn_grid_detector_demo.sh $(BUILD)

demo-cnn-grid-detector-real: build
	./demos/cnn_grid_detector_real_demo.sh $(BUILD)

demo-attn-text-classifier: build
	./demos/attn_text_classifier_demo.sh $(BUILD)

demo-attn-retrieval: build
	./demos/attn_retrieval_demo.sh $(BUILD)

demo-vit-attention-one-layer: build
	./demos/vit_attention_one_layer_demo.sh $(BUILD)

demo-vit-attention-all-layers: build
	./demos/vit_attention_all_layers_demo.sh $(BUILD)

# Optional: renders PNGs from Tensora's own compiled output for
# examples/vit_attention_weights.tns. Needs matplotlib in addition to
# demo-vit-attention-one-layer's prerequisites; not part of
# `make tensora-demos` (produces image files for inspection, not a
# pass/fail check, though it does assert Tensora's weights match the
# real model).
demo-vit-attention-viz: build
	python3 tools/models/visualize_vit_attention.py $(BUILD)

demo-vit-attention-substitution: build
	./demos/vit_attention_substitution_demo.sh $(BUILD)

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
