cmake_minimum_required(VERSION 3.16)
project(tensora CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Frontend generator: ANTLR4. The build fetches both the generator tool (a
# jar, run through a system `java`) and the C++ runtime it needs. This
# keeps the build reproducible. No machine-specific install path is
# ever hard-coded.
find_package(Java QUIET COMPONENTS Runtime)
if(NOT Java_JAVA_EXECUTABLE)
  message(FATAL_ERROR "ANTLR code generation needs a Java runtime (`java`) on PATH; "
                       "see docs/development/BUILD.md.")
endif()

set(ANTLR4_VERSION 4.13.2)
include(FetchContent)

FetchContent_Declare(antlr4_jar
  URL https://www.antlr.org/download/antlr-${ANTLR4_VERSION}-complete.jar
  DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(antlr4_jar)
set(ANTLR4_JAR_PATH ${antlr4_jar_SOURCE_DIR}/antlr-${ANTLR4_VERSION}-complete.jar)

set(ANTLR4_INSTALL OFF CACHE BOOL "" FORCE)
set(WITH_DEMO OFF CACHE BOOL "" FORCE)
set(WITH_LIBCXX OFF CACHE BOOL "" FORCE)
set(ANTLR_BUILD_CPP_TESTS OFF CACHE BOOL "" FORCE)
set(ANTLR_BUILD_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(antlr4_runtime
  URL https://www.antlr.org/download/antlr4-cpp-runtime-${ANTLR4_VERSION}-source.zip)
FetchContent_MakeAvailable(antlr4_runtime)

# Generate TensoraLexer/TensoraParser (and a visitor base) from
# grammar/Tensora.g4 into the build tree. Generated code is never
# committed and never hand-edited. It stays separate from project-owned
# C++ in src/parser/AstBuilder.cpp, which walks the parse tree.
set(ANTLR4_GEN_DIR ${CMAKE_BINARY_DIR}/antlr4_generated)
file(MAKE_DIRECTORY ${ANTLR4_GEN_DIR})
set(ANTLR4_GENERATED_SRCS
  ${ANTLR4_GEN_DIR}/TensoraLexer.cpp
  ${ANTLR4_GEN_DIR}/TensoraParser.cpp)
add_custom_command(
  OUTPUT ${ANTLR4_GENERATED_SRCS}
  COMMAND ${Java_JAVA_EXECUTABLE} -jar ${ANTLR4_JAR_PATH}
          -Dlanguage=Cpp -no-listener -no-visitor
          -o ${ANTLR4_GEN_DIR}
          -package tensora::generated
          ${CMAKE_SOURCE_DIR}/grammar/Tensora.g4
  DEPENDS ${CMAKE_SOURCE_DIR}/grammar/Tensora.g4
  COMMENT "ANTLR: generating TensoraLexer/TensoraParser from grammar/Tensora.g4")
add_custom_target(antlr4_generate DEPENDS ${ANTLR4_GENERATED_SRCS})

include_directories(${CMAKE_SOURCE_DIR}/include)

# -Wall -Wextra for project-owned targets only (docs/development/
# 07_DEVELOPMENT_ENVIRONMENT.md §7.5). Placed after the ANTLR
# FetchContent/generate setup above, so it does not apply to the
# vendored ANTLR runtime target, only to the libraries and executables
# this CMakeLists.txt defines below.
add_compile_options(-Wall -Wextra)

# Frontend library: hand-written C++ (Lexer/Parser adapters, the
# AstBuilder that walks ANTLR's parse tree, and the parser error
# listener) plus the generated ANTLR lexer/parser.
add_library(tensora_frontend
  src/lexer/Lexer.cpp
  src/lexer/TokenAdapter.cpp
  src/parser/Parser.cpp
  src/parser/AstBuilder.cpp
  src/parser/ParserErrorListener.cpp
  src/parser/ParserErrorStrategy.cpp
  ${ANTLR4_GENERATED_SRCS})
add_dependencies(tensora_frontend antlr4_generate)
target_include_directories(tensora_frontend PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tensora_frontend PRIVATE ${ANTLR4_GEN_DIR})
target_link_libraries(tensora_frontend PUBLIC antlr4_static)
# The generated headers include <antlr4-runtime.h>. So every consumer of
# the generated code (this library, and anything that links against it)
# needs the runtime's public include dir. The PUBLIC link to
# antlr4_static already carries that dir, through its own
# INTERFACE_INCLUDE_DIRECTORIES.

# Semantic analysis phase. shared/{Type,DimSolver,Ir}.cpp holds the
# types and the Typed Computation Graph (used by more than one phase).
# semantic/ holds the symbol table, the operator contracts, and the
# analyzer driver. This library has no MLIR/LLVM dependency
# (ARCHITECTURE.md §2).
add_library(tensora_semantic
  src/shared/Type.cpp
  src/shared/DimSolver.cpp
  src/shared/Ir.cpp
  src/semantic/Symbol.cpp
  src/semantic/Contracts.cpp
  src/semantic/SemanticAnalyzer.cpp)
target_include_directories(tensora_semantic PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tensora_semantic PUBLIC tensora_frontend)

# Optimization phase (RM 3.21): the Level-1 graph passes, plus the
# tensor-liveness analysis. Plain C++ over the TCG. No MLIR/LLVM
# dependency (ARCHITECTURE.md §2).
add_library(tensora_optimization src/optimization/GraphPasses.cpp src/optimization/Liveness.cpp)
target_include_directories(tensora_optimization PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tensora_optimization PUBLIC tensora_semantic)

# Codegen phase: the MLIR/LLVM backend (INTEG-003 pins LLVM/MLIR 21, apt
# packages). Only codegen/ touches MLIR/LLVM headers (ARCHITECTURE.md
# §2); semantic/ and shared/ above never do. This block must come
# before tensora_codegen below, which needs MLIR_INCLUDE_DIRS and the
# MLIR imported target.
list(APPEND CMAKE_PREFIX_PATH
  /usr/lib/llvm-21/lib/cmake/mlir
  /usr/lib/llvm-21/lib/cmake/llvm)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
message(STATUS "Found MLIR (${MLIR_DIR})")
add_definitions(${LLVM_DEFINITIONS})

# A JIT-run module can lower to a non-identity-layout memref (tiling's
# subviews, flatten's collapse_shape, conv2d's pad). That code calls
# MLIR's own `memrefCopy` runtime helper. The JIT must know where to
# find that helper. So this line finds it once, here, for every target
# below that needs it (the driver and several end-to-end test binaries).
find_library(MLIR_C_RUNNER_UTILS_LIB mlir_c_runner_utils PATHS ${LLVM_LIBRARY_DIR} ${MLIR_LIBRARY_DIR} REQUIRED)
message(STATUS "Found mlir_c_runner_utils: ${MLIR_C_RUNNER_UTILS_LIB}")

# TCG -> MLIR (ARCHITECTURE.md §7.1). Only this library touches MLIR/
# LLVM headers (ARCHITECTURE.md §2).
add_library(tensora_codegen
  src/codegen/DialectSetup.cpp
  src/codegen/MlirLowerer.cpp
  src/codegen/LoopTiling.cpp
  src/codegen/MatmulTilingPass.cpp)
target_include_directories(tensora_codegen PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tensora_codegen SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(tensora_codegen PUBLIC tensora_optimization PRIVATE MLIR)

# Command-line driver: one flag per compiler phase, plus --run/-o,
# which run the full pipeline (docs/language/02_LANGUAGE_TUTORIAL.md
# §2.12).
add_executable(tensora src/driver/main.cpp)
target_include_directories(tensora SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(tensora PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(tensora PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")

# Tests run through CTest. tests/support/TestHarness.h is the shared,
# no-framework harness (Reporter/Case registry). Every test binary uses it.
enable_testing()

add_executable(test_lexer tests/lexer/test_lexer.cpp)
target_link_libraries(test_lexer PRIVATE tensora_frontend)
target_include_directories(test_lexer PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_parser tests/parser/test_parser.cpp)
target_link_libraries(test_parser PRIVATE tensora_frontend)
target_include_directories(test_parser PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_semantic tests/semantic/test_semantic.cpp)
target_link_libraries(test_semantic PRIVATE tensora_semantic)
target_include_directories(test_semantic PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_graph tests/semantic/test_graph.cpp)
target_link_libraries(test_graph PRIVATE tensora_semantic)
target_include_directories(test_graph PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_passes tests/optimization/test_passes.cpp)
target_link_libraries(test_passes PRIVATE tensora_optimization)
target_include_directories(test_passes PRIVATE ${CMAKE_SOURCE_DIR}/tests)

# Each case ID gets its own CTest test (test_lexer/test_parser <ID>
# runs just that case). So `ctest` reports which rule failed, instead
# of one pass/fail result for everything. These IDs match
# docs/testing/TEST_MATRIX.md. Keep the two lists in sync.
set(LEXER_TEST_CASES
  LEX-001 LEX-002 LEX-003 LEX-004 LEX-005 LEX-006 LEX-007 LEX-008
  LEX-009 LEX-010 LEX-011
  BND-LEX-001 BND-LEX-002 BND-LEX-003 BND-LEX-004)
foreach(case_id ${LEXER_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_lexer ${case_id})
endforeach()

set(PARSER_TEST_CASES
  PAR-001 PAR-002 PAR-003 PAR-004 PAR-005 PAR-006 PAR-007 PAR-008 PAR-009
  PAR-010 PAR-011 PAR-012 PAR-013 PAR-014 PAR-015 PAR-016 PAR-017 PAR-018
  PAR-019 PAR-020 PAR-021 PAR-022 PAR-023 PAR-024 PAR-025 PAR-026 PAR-027
  PAR-028 PAR-029 PAR-030 PAR-031 PAR-032
  BND-PAR-001 BND-PAR-002 BND-PAR-003 BND-PAR-004)
foreach(case_id ${PARSER_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_parser ${case_id})
endforeach()

set(SEMANTIC_TEST_CASES
  SEM-DECL-001 SEM-DECL-002 SEM-DECL-003 SEM-DECL-004 SEM-DECL-005 SEM-NAME-001 SEM-NAME-002 SEM-NAME-003
  SEM-SCOPE-001 SEM-SCOPE-002 SEM-TYPE-001
  SEM-ARITY-001 SEM-ARITY-002 SEM-ARITY-003 SEM-ARITY-004 SEM-ARITY-005
  SEM-POISON-001 SEM-POISON-002
  SEM-MM-001 SEM-MM-002 SEM-MM-003 SEM-MM-004
  SEM-LIN-001 SEM-LIN-002 SEM-LIN-003
  SEM-EW-001 SEM-EW-002 SEM-EW-003 SEM-EW-004 SEM-EW-005
  SEM-RS-001 SEM-RS-002 SEM-RS-003 SEM-RS-004
  SEM-FL-001 SEM-FL-002
  SEM-TR-001 SEM-TR-002 SEM-TR-003
  SEM-RD-001 SEM-RD-002
  SEM-SM-001 SEM-SM-002
  SEM-CV-001 SEM-CV-002 SEM-CV-003 SEM-CV-004 SEM-CV-005
  SEM-PL-001 SEM-PL-002
  SEM-DIM-001 SEM-DIM-002 SEM-DIM-003
  SYM-001 SYM-002 SYM-003 SYM-004 SYM-005 SYM-006 SYM-007 SYM-008
  ATT-001 ATT-002
  BND-MM-001 BND-MM-002 BND-BC-001 BND-RS-001 BND-FL-001
  BND-CV-001 BND-CV-002 BND-RD-001 BND-TR-001 BND-DIM-001)
foreach(case_id ${SEMANTIC_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_semantic ${case_id})
endforeach()

set(GRAPH_TEST_CASES GR-001 GR-002 GR-003 GR-004 GR-005 GR-006 GR-007)
foreach(case_id ${GRAPH_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_graph ${case_id})
endforeach()

set(PASSES_TEST_CASES
  OPT-SF-001 OPT-SF-002 OPT-DCE-001 OPT-DCE-002 OPT-DCE-003
  OPT-RC-001 OPT-RC-002 OPT-RC-003 OPT-TC-001 OPT-TC-002 OPT-TC-003
  OPT-EF-001 OPT-EF-002 OPT-LBA-001 OPT-LBA-002 OPT-FIX-001 OPT-FIX-002
  LIVE-001 LIVE-002 LIVE-003 LIVE-004)
# (LOW-E2E-MLP is registered separately below, next to its executable.)
foreach(case_id ${PASSES_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_passes ${case_id})
endforeach()

# SMOKE-MLIR (TEST_PLAN §2b): checks the MLIR/LLVM link+JIT path works,
# before any real lowering code is built on it. This is not a CTest
# case. It has no checks beyond "did it crash." Run it directly.
add_executable(smoke_mlir tests/smoke/smoke_mlir.cpp)
target_include_directories(smoke_mlir SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(smoke_mlir PRIVATE MLIR MLIRExecutionEngineShared)
add_test(NAME SMOKE-MLIR COMMAND smoke_mlir)

# LOW-E2E-MLP: checks that MlirLowerer's matmul/linear/relu lowering
# gives the right numbers, through real JIT execution (not just a
# golden-MLIR-text check).
add_executable(test_mlp_e2e tests/codegen/test_mlp_e2e.cpp)
target_include_directories(test_mlp_e2e PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_mlp_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
add_test(NAME LOW-E2E-MLP COMMAND test_mlp_e2e)

# LOOP-001: TilingDecision is plain C++. It needs no MLIR headers here.
add_executable(test_tiling_decision tests/codegen/test_tiling_decision.cpp)
target_link_libraries(test_tiling_decision PRIVATE tensora_codegen)
target_include_directories(test_tiling_decision PRIVATE ${CMAKE_SOURCE_DIR}/tests)
foreach(case_id LOOP-001-A LOOP-001-B LOOP-001-C LOOP-001-D LOOP-001-E LOOP-001-F)
  add_test(NAME ${case_id} COMMAND test_tiling_decision ${case_id})
endforeach()

# LOOP-002/003/004: the project-authored MLIR tiling pass. Checks its
# structure, checks that its numbers match the untiled version, and
# reports timing.
add_executable(test_tiling_e2e tests/codegen/test_tiling_e2e.cpp)
target_include_directories(test_tiling_e2e PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_tiling_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_tiling_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME LOOP-002 COMMAND test_tiling_e2e)

# LOOP-005: proves MlirLowerer's own internal LivenessAnalysis call
# (not a hand-built LivenessResult) drives TilingDecision's memory-
# pressure branch for real, through the real lowering path.
add_executable(test_liveness_tiling_e2e tests/codegen/test_liveness_tiling_e2e.cpp)
target_include_directories(test_liveness_tiling_e2e PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_liveness_tiling_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
add_test(NAME LOOP-005 COMMAND test_liveness_tiling_e2e)

# NATIVE-001: a real ahead-of-time-compiled executable. codegen_mlp_native
# lowers the same MLP TypedGraph as LOW-E2E-MLP, all the way to a real
# object file, through LLVM's TargetMachine (no JIT or ExecutionEngine
# involved). A small C driver, with no other dependency, is then built
# and linked against that object file by the system toolchain. This
# makes a standalone executable that CTest runs directly.
add_executable(codegen_mlp_native tools/codegen_mlp_native.cpp)
target_include_directories(codegen_mlp_native SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(codegen_mlp_native PRIVATE tensora_codegen MLIR LLVM)

# Reporting tool, not a CTest case: measures real wall-clock time for a
# few candidate matmul tile sizes so the tiling policy's defaults are an
# evidence-based choice, not an assumed one.
add_executable(benchmark_matmul_tiling tools/benchmark_matmul_tiling.cpp)
target_include_directories(benchmark_matmul_tiling SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(benchmark_matmul_tiling PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(benchmark_matmul_tiling PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")

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

# LOW-006..009: transpose, flatten, softmax, and conv2d. Each op is
# checked on its own through JIT execution, before the CNN and
# attention end-to-end tests below combine them.
add_executable(test_new_ops_e2e tests/codegen/test_new_ops_e2e.cpp)
target_include_directories(test_new_ops_e2e PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_new_ops_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_new_ops_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME LOW-NEWOPS COMMAND test_new_ops_e2e)

# RANK-001..004: checks the numbers for rank-1/rank>=3 `linear` and
# rank>=2 (with broadcasting) `@` lowering. This closes a gap between
# the language and the backend: semantic analysis accepted ranks that
# MlirLowerer could not lower.
add_executable(test_rank_e2e tests/codegen/test_rank_e2e.cpp)
target_include_directories(test_rank_e2e PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_rank_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_rank_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME RANK-E2E COMMAND test_rank_e2e)

# E2E-002: CNN classifier (conv2d->relu->flatten->linear->softmax).
add_executable(test_cnn_e2e tests/codegen/test_cnn_e2e.cpp)
target_include_directories(test_cnn_e2e PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_cnn_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_cnn_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME E2E-002 COMMAND test_cnn_e2e)

# E2E-004: single-head scaled dot-product attention.
add_executable(test_attention_e2e tests/codegen/test_attention_e2e.cpp)
target_include_directories(test_attention_e2e PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_attention_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_attention_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME E2E-004 COMMAND test_attention_e2e)

# E2E-003: Transformer FFN sublayer (linear->relu->linear at FFN
# shapes). Reuses lowering already proven above. It adds no new ops.
add_executable(test_ffn_e2e tests/codegen/test_ffn_e2e.cpp)
target_include_directories(test_ffn_e2e PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_ffn_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
add_test(NAME E2E-003 COMMAND test_ffn_e2e)

# INT-FULL-*: the real `tensora` CLI driver, run from source text all
# the way to JIT execution, against examples/*.tns. This differs from
# every e2e test above, which builds its TypedGraph by hand in C++.
# Expected values are checked by hand (see each .tns file's header
# comment) against all-ones synthetic input data, which --run always
# uses (Tensora source declares shapes, not values).
# INT-FULL-SYMBOLS: proof, with a golden output, that --emit-symbols
# prints SemanticAnalyzer's real symbol table (RM 3.6) for a real
# program: global input/weight declarations, graph-local bindings
# (including h's shadowing, which collapses to one entry), and the
# resolution log. The output is the same every run.
add_test(NAME INT-FULL-SYMBOLS COMMAND tensora --emit-symbols ${CMAKE_SOURCE_DIR}/examples/mlp.tns)
set_tests_properties(INT-FULL-SYMBOLS PROPERTIES PASS_REGULAR_EXPRESSION
  "x  input  Tensor\\[batch=B, feat=784\\]")

add_test(NAME INT-FULL-MLP-CHECK COMMAND tensora --check ${CMAKE_SOURCE_DIR}/examples/mlp_concrete.tns)
add_test(NAME INT-FULL-MLP-RUN COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/mlp_concrete.tns)
set_tests_properties(INT-FULL-MLP-RUN PROPERTIES PASS_REGULAR_EXPRESSION "element\\[0\\] = 145")

add_test(NAME INT-FULL-MM-RUN COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/matmul_large.tns)
set_tests_properties(INT-FULL-MM-RUN PROPERTIES PASS_REGULAR_EXPRESSION "element\\[0\\] = 256")

add_test(NAME INT-FULL-CNN-RUN COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/cnn.tns)
set_tests_properties(INT-FULL-CNN-RUN PROPERTIES PASS_REGULAR_EXPRESSION "element\\[0\\] = 0.5")

add_test(NAME INT-FULL-ATTN-RUN COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/attention.tns)
set_tests_properties(INT-FULL-ATTN-RUN PROPERTIES PASS_REGULAR_EXPRESSION "element\\[0\\] = 1")

# This is a deliberate negative case. mlp.tns has a symbolic batch dim.
# --run must reject it with the documented limitation message. It must
# not crash, and it must not silently give a wrong answer.
add_test(NAME INT-FULL-SYMBOLIC-REJECT COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/mlp.tns)
set_tests_properties(INT-FULL-SYMBOLIC-REJECT PROPERTIES WILL_FAIL TRUE)

# Category labels for `ctest -L <category>` filtering. These come from
# the same ID-prefix rule the C++ harness uses
# (tests/support/TestHarness.h categoryOf). This block sets the label
# once here, for every test already registered above, instead of
# repeating it per test.
get_property(ALL_TEST_NAMES DIRECTORY PROPERTY TESTS)
foreach(t ${ALL_TEST_NAMES})
  if(t MATCHES "^BND-")
    set_tests_properties(${t} PROPERTIES LABELS boundary)
  elseif(t MATCHES "^SMOKE")
    set_tests_properties(${t} PROPERTIES LABELS smoke)
  elseif(t MATCHES "^INT-")
    set_tests_properties(${t} PROPERTIES LABELS integration)
  elseif(t MATCHES "^(E2E-|LOW-|NATIVE-|LOOP-|LIVE-)")
    set_tests_properties(${t} PROPERTIES LABELS integration)
  else()
    set_tests_properties(${t} PROPERTIES LABELS unit)
  endif()
endforeach()
