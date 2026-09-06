# Tensora test + demo-target registration, included from the top-level
# CMakeLists.txt. Kept separate so the top file stays focused on the
# build (libraries, driver), and all test wiring lives in one place.
# Included (not add_subdirectory'd) so targets still build into build/.

# Tests run through CTest. tests/support/TestHarness.h is the shared,
# no-framework harness (Reporter/Case registry). Every test binary uses it.
enable_testing()

# --- Test helpers ---------------------------------------------------------
# One line per test target instead of the repeated add_executable +
# target_include_directories + target_link_libraries boilerplate.

# A plain unit test: links one phase library, sees the tests/ include dir.
function(tensora_unit_test name src lib)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE ${lib})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
endfunction()

# A codegen test: additionally needs the MLIR/LLVM headers/libs and the
# c-runner-utils path (the -D is harmless for the few tests that ignore it).
function(tensora_mlir_test name)
  add_executable(${name} ${ARGN})  # one or more source files
  target_include_directories(${name} SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
  target_link_libraries(${name} PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
  target_compile_definitions(${name} PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
endfunction()

# Register one CTest per CASE("<id>", ...) in a harness test's source, so
# the case IDs live ONLY in the .cpp -- CMake no longer repeats them. The
# INT-TESTMATRIX-SYNC check below keeps docs/testing/TEST_MATRIX.md in sync.
# Extraction uses grep+sed (the project is Linux-only, INTEG-001) because
# file(STRINGS ... REGEX) merges adjacent matched lines in its list form.
function(tensora_discover_cases bin srcfile)
  execute_process(
    COMMAND bash -c "grep -oE '(CASE|TEST_CASE)\\([A-Za-z0-9_]+, *\"[^\"]+\"' '${srcfile}' | sed -E 's/.*\"([^\"]+)\"/\\1/'"
    OUTPUT_VARIABLE ids OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "tensora_discover_cases: failed to read cases from ${srcfile}")
  endif()
  string(REPLACE "\n" ";" ids "${ids}")
  foreach(id IN LISTS ids)
    if(id)
      add_test(NAME ${id} COMMAND ${bin} ${id})
    endif()
  endforeach()
endfunction()

# Same as tensora_discover_cases, but for a fixture file's
# '### <ID> | <description>' lines (tests/support/FixtureLoader.h reads
# the same file at runtime, so this and the runtime registry never see a
# different case list -- there is no generated intermediate file to keep
# in sync). See tests/fixtures/*.tns for the file format.
function(tensora_discover_fixture_cases bin fixturefile)
  execute_process(
    COMMAND bash -c "grep -oE '^### [A-Za-z0-9_-]+ \\|' '${fixturefile}' | sed -E 's/^### ([A-Za-z0-9_-]+) .*/\\1/'"
    OUTPUT_VARIABLE ids OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "tensora_discover_fixture_cases: failed to read cases from ${fixturefile}")
  endif()
  string(REPLACE "\n" ";" ids "${ids}")
  foreach(id IN LISTS ids)
    if(id)
      add_test(NAME ${id} COMMAND ${bin} ${id})
    endif()
  endforeach()
endfunction()

# Harness-based unit tests: build each, then discover its CASE(...) IDs from
# the source so every case still gets its own CTest (ctest reports which
# rule failed), with the ID list living only in the .cpp.
tensora_unit_test(test_lexer    tests/lexer/test_lexer.cpp         tensora_lexer)
tensora_discover_cases(test_lexer    ${CMAKE_SOURCE_DIR}/tests/lexer/test_lexer.cpp)

tensora_unit_test(test_parser   tests/parser/test_parser.cpp       tensora_parser)
target_compile_definitions(test_parser PRIVATE
  FIXTURE_PARSER_PATH="${CMAKE_SOURCE_DIR}/tests/fixtures/parser.tns"
  FIXTURE_PARSER_STRUCTURAL_PATH="${CMAKE_SOURCE_DIR}/tests/fixtures/parser_structural.tns")
tensora_discover_cases(test_parser   ${CMAKE_SOURCE_DIR}/tests/parser/test_parser.cpp)
tensora_discover_fixture_cases(test_parser ${CMAKE_SOURCE_DIR}/tests/fixtures/parser.tns)

tensora_unit_test(test_semantic tests/semantic/test_semantic.cpp   tensora_semantic)
target_compile_definitions(test_semantic PRIVATE
  FIXTURE_SEMANTIC_PATH="${CMAKE_SOURCE_DIR}/tests/fixtures/semantic.tns"
  FIXTURE_SEMANTIC_STRUCTURAL_PATH="${CMAKE_SOURCE_DIR}/tests/fixtures/semantic_structural.tns")
tensora_discover_cases(test_semantic ${CMAKE_SOURCE_DIR}/tests/semantic/test_semantic.cpp)
tensora_discover_fixture_cases(test_semantic ${CMAKE_SOURCE_DIR}/tests/fixtures/semantic.tns)

tensora_unit_test(test_graph    tests/semantic/test_graph.cpp      tensora_semantic)
target_compile_definitions(test_graph PRIVATE FIXTURE_GRAPH_PATH="${CMAKE_SOURCE_DIR}/tests/fixtures/graph.tns")
tensora_discover_cases(test_graph    ${CMAKE_SOURCE_DIR}/tests/semantic/test_graph.cpp)

tensora_unit_test(test_passes   tests/optimization/test_passes.cpp tensora_optimization)
tensora_discover_cases(test_passes   ${CMAKE_SOURCE_DIR}/tests/optimization/test_passes.cpp)

# SMOKE-MLIR (TEST_PLAN §2b): minimal MLIR/LLVM link+JIT check, before any
# codegen is built on it -- so it deliberately does NOT link tensora_codegen
# (kept explicit rather than via the helper for that reason).
add_executable(smoke_mlir tests/smoke/smoke_mlir.cpp)
target_include_directories(smoke_mlir SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(smoke_mlir PRIVATE MLIR MLIRExecutionEngineShared)
add_test(NAME SMOKE-MLIR COMMAND smoke_mlir)

# LOOP-001: TilingDecision is plain C++ (no MLIR headers); discover its cases.
tensora_unit_test(test_tiling_decision tests/codegen/test_tiling_decision.cpp tensora_codegen)
tensora_discover_cases(test_tiling_decision ${CMAKE_SOURCE_DIR}/tests/codegen/test_tiling_decision.cpp)

# Codegen e2e tests: one binary, one CASE per check (each check's numeric
# JIT logic is a run_*() in its own translation unit; test_codegen_e2e.cpp
# registers them as CASEs). Discovered like every other harness test, so
# each still runs as its own isolated CTest (`test_codegen_e2e <ID>`):
# LOW-E2E-MLP, LOOP-002, LOOP-005, LOW-NEWOPS, RANK-E2E, E2E-002/003/004.
tensora_mlir_test(test_codegen_e2e
  tests/codegen/test_codegen_e2e.cpp
  tests/codegen/test_mlp_e2e.cpp
  tests/codegen/test_tiling_e2e.cpp
  tests/codegen/test_liveness_tiling_e2e.cpp
  tests/codegen/test_new_ops_e2e.cpp
  tests/codegen/test_rank_e2e.cpp
  tests/codegen/test_cnn_e2e.cpp
  tests/codegen/test_attention_e2e.cpp
  tests/codegen/test_ffn_e2e.cpp)
tensora_discover_cases(test_codegen_e2e ${CMAKE_SOURCE_DIR}/tests/codegen/test_codegen_e2e.cpp)

# Tools (built, not CTest cases): the AOT object emitter (links LLVM, not
# the JIT engine -- kept explicit) and the tiling benchmark.
add_executable(codegen_mlp_native tools/codegen_mlp_native.cpp)
target_include_directories(codegen_mlp_native SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(codegen_mlp_native PRIVATE tensora_codegen MLIR LLVM)
tensora_mlir_test(benchmark_matmul_tiling tools/benchmark_matmul_tiling.cpp)

# NATIVE-001: the same MLP TypedGraph emitted as a real object file via
# LLVM's TargetMachine (no JIT), then linked with a small C driver by the
# system toolchain into a standalone executable CTest runs directly.
set(MLP_NATIVE_OBJ ${CMAKE_CURRENT_BINARY_DIR}/mlp_native.o)
add_custom_command(
  OUTPUT ${MLP_NATIVE_OBJ}
  COMMAND codegen_mlp_native ${MLP_NATIVE_OBJ}
  DEPENDS codegen_mlp_native
  COMMENT "Emitting native object code for the MLP workload via Tensora's MLIR/LLVM backend")
add_custom_target(mlp_native_object DEPENDS ${MLP_NATIVE_OBJ})
add_executable(test_mlp_native tests/codegen/mlp_native_driver.c ${MLP_NATIVE_OBJ})
set_source_files_properties(${MLP_NATIVE_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_dependencies(test_mlp_native mlp_native_object)
target_link_libraries(test_mlp_native PRIVATE m)
add_test(NAME NATIVE-001 COMMAND test_mlp_native)

# INT-* CLI integration tests are data-driven: tests/cli/cli_tests.txt is a
# table of `<name> | <expect> | <args>`, read here and registered by one
# generic loop. Adding a CLI test is a one-line edit in that file, not new
# CMake. These exercise the real `tensora` driver end to end (source ->
# JIT), distinct from the C++ e2e tests that build the graph by hand. The
# passes'/operators' own correctness stays in the C++ unit tests -- no
# overlap. (Re-run `cmake` after editing the table, as `make`/`ctest` do.)
file(STRINGS ${CMAKE_SOURCE_DIR}/tests/cli/cli_tests.txt CLI_TEST_LINES)
foreach(line ${CLI_TEST_LINES})
  string(STRIP "${line}" line)
  if(line STREQUAL "" OR line MATCHES "^#")
    continue()
  endif()
  string(REPLACE "|" ";" fields "${line}")
  list(GET fields 0 t_name)
  list(GET fields 1 t_expect)
  list(GET fields 2 t_args)
  string(STRIP "${t_name}" t_name)
  string(STRIP "${t_expect}" t_expect)
  string(STRIP "${t_args}" t_args)
  string(REPLACE "@SRC@" "\"${CMAKE_SOURCE_DIR}\"" t_args "${t_args}")
  separate_arguments(t_argv UNIX_COMMAND "${t_args}")
  add_test(NAME ${t_name} COMMAND tensora ${t_argv})
  if(t_expect STREQUAL "FAIL")
    set_tests_properties(${t_name} PROPERTIES WILL_FAIL TRUE)
  elseif(t_expect MATCHES "^PASS=(.+)$")
    set_tests_properties(${t_name} PROPERTIES PASS_REGULAR_EXPRESSION "${CMAKE_MATCH_1}")
  endif()
endforeach()

# INT-IR-*-STD: prove Tensora emits STANDARD MLIR/LLVM IR that the upstream
# tools accept on their own -- independent of Tensora's runtime, and a
# different IR world than PyTorch's. Gated on the tools being present so the
# suite stays green where they are not installed. These validate the IR (no
# execution): full upstream lowering/execution pipelines are version-
# sensitive and live in the independent-run demo, not the gated suite.
find_program(TENSORA_MLIR_OPT mlir-opt PATHS /usr/lib/llvm-21/bin)
find_program(TENSORA_LLVM_AS llvm-as PATHS /usr/lib/llvm-21/bin)
if(TENSORA_MLIR_OPT)
  add_test(NAME INT-IR-MLIR-STD COMMAND bash -c
    "$<TARGET_FILE:tensora> --emit-mlir ${CMAKE_SOURCE_DIR}/examples/mlp_concrete.tns | ${TENSORA_MLIR_OPT} -o /dev/null")
endif()
if(TENSORA_LLVM_AS)
  add_test(NAME INT-IR-LLVM-STD COMMAND bash -c
    "$<TARGET_FILE:tensora> --emit-llvm ${CMAKE_SOURCE_DIR}/examples/mlp_concrete.tns | ${TENSORA_LLVM_AS} -o /dev/null")
endif()

# INT-TESTMATRIX-SYNC: the drift guard. Every CASE("<id>",...) in the test
# sources must be documented in docs/testing/TEST_MATRIX.md, so the matrix
# cannot silently fall out of step with the actual suite. The case IDs live
# only in the .cpp (CMake discovers them; the matrix documents them).
add_test(NAME INT-TESTMATRIX-SYNC COMMAND bash ${CMAKE_SOURCE_DIR}/tools/check_test_matrix.sh
         $<TARGET_FILE_DIR:test_lexer>)

# INT-DISCOVERY-SYNC: assert CMake's grep-based case discovery
# (tensora_discover_cases) exactly matches each test binary's own
# authoritative `--list` (the registry). So the tests themselves are the
# single source of truth and a silent grep miss fails loudly.
add_test(NAME INT-DISCOVERY-SYNC COMMAND bash ${CMAKE_SOURCE_DIR}/tools/check_discovery.sh
         $<TARGET_FILE_DIR:test_lexer>)

# INT-DOCLINKS-SYNC: every numbered cross-reference link in docs/*.md
# (e.g. "[RM 3.6](...)") must land on the section its own label names,
# not merely on some anchor that happens to exist. Gated on python3
# (already an optional dependency for tools/regen_appendix.py and the
# demo scripts), so the suite stays green on a machine without it.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME INT-DOCLINKS-SYNC COMMAND ${Python3_EXECUTABLE}
           ${CMAKE_SOURCE_DIR}/tools/check_doc_link_targets.py)
endif()

# Category labels for `ctest -L <category>` filtering. These come from
# the same ID-prefix rule the C++ harness uses
# (tests/support/TestHarness.h categoryOf). This block sets the label
# once here, for every test already registered above, instead of
# repeating it per test.
get_property(ALL_TEST_NAMES DIRECTORY PROPERTY TESTS)
foreach(t ${ALL_TEST_NAMES})
  if(t MATCHES "^BND-")
    set(kind_label boundary)
  elseif(t MATCHES "^SMOKE")
    set(kind_label smoke)
  elseif(t MATCHES "^INT-")
    set(kind_label integration)
  elseif(t MATCHES "^(E2E-|LOW-|NATIVE-|LOOP-|LIVE-|RANK-)")
    set(kind_label integration)
  else()
    set(kind_label unit)
  endif()

  # Compiler-phase label (`ctest -L <phase>`), so a phase can be tested
  # in isolation the same way it can be built in isolation (the
  # `lexer`/`parser`/`semantic`/`optimization`/`codegen` CMake targets
  # below). Boundary-case IDs (BND-LEX-*, BND-MM-*, etc.) carry the
  # same phase prefix rule as the unit-test IDs they extend.
  if(t MATCHES "^(LEX-|BND-LEX-)")
    set(phase_label lexer)
  elseif(t MATCHES "^(PAR-|BND-PAR-)")
    set(phase_label parser)
  elseif(t MATCHES "^(SEM-|SYM-|ATT-|GR-|BND-MM-|BND-BC-|BND-RS-|BND-FL-|BND-CV-|BND-RD-|BND-TR-|BND-DIM-)")
    set(phase_label semantic)
  elseif(t MATCHES "^(OPT-|LIVE-)")
    set(phase_label optimization)
  elseif(t MATCHES "^(SMOKE|LOW-|LOOP-|NATIVE-|RANK-|E2E-|INT-)")
    set(phase_label codegen)
  else()
    set(phase_label unknown)
  endif()

  set_tests_properties(${t} PROPERTIES LABELS "${kind_label};${phase_label}")
endforeach()

# Named per-phase build targets (docs/development/BUILD.md "Build one
# phase at a time"). Each just depends on the executables that phase
# actually needs, so `cmake --build build --target lexer` (etc.)
# builds only that slice of the DAG instead of everything -- useful
# when submitting or demoing one phase at a time.
add_custom_target(lexer DEPENDS test_lexer)
add_custom_target(parser DEPENDS test_parser)
add_custom_target(semantic DEPENDS test_semantic test_graph)
add_custom_target(optimization DEPENDS test_passes)
# Named "backend", not "codegen": CMake reserves the target name
# "codegen" for its own CODEGEN custom-command feature (CMP0171).
add_custom_target(backend DEPENDS
  smoke_mlir test_tiling_decision test_codegen_e2e test_mlp_native
  codegen_mlp_native benchmark_matmul_tiling tensora)
add_custom_target(full DEPENDS lexer parser semantic optimization backend)
