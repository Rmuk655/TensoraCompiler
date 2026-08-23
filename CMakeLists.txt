cmake_minimum_required(VERSION 3.16)

# NEW: re-extract FetchContent URL downloads' timestamps at extraction
# time instead of preserving the archive's own timestamps, so a change
# to a pinned URL's content is always seen as a change. Set before
# `project()` per the policy's own documented scope.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(tensora CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Give the vendored ANTLR4 runtime's own CMakeLists a build type so it
# does not warn and fall back to Release itself; Release is also this
# project's own default when none is requested on the command line.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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

# FORCE here always wins over a `-D...` passed on the `cmake` command
# line, on purpose: none of these are options we want a developer to
# be able to turn on by accident (WITH_DEMO in particular builds
# ANTLR's own vendored demo CLI, not anything of ours).
set(ANTLR4_INSTALL OFF CACHE BOOL "" FORCE)
set(WITH_DEMO OFF CACHE BOOL "" FORCE)
set(WITH_LIBCXX OFF CACHE BOOL "" FORCE)
set(ANTLR_BUILD_CPP_TESTS OFF CACHE BOOL "" FORCE)
set(ANTLR_BUILD_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(antlr4_runtime
  URL https://www.antlr.org/download/antlr4-cpp-runtime-${ANTLR4_VERSION}-source.zip)
FetchContent_MakeAvailable(antlr4_runtime)
# The "Building without demo. To enable demo build use: -DWITH_DEMO=True"
# line just above comes from the vendored ANTLR runtime's own
# CMakeLists.txt -- it has no way to know we FORCE that flag off above,
# so its hint does not actually work in this project. This message
# clarifies that in the same configure log, since we cannot edit or
# suppress the vendored one.
message(STATUS "(the WITH_DEMO hint above is from the vendored ANTLR runtime and does not apply here -- this project always disables it; see CMakeLists.txt)")

# antlr4_static's own CMakeLists exports its headers as normal (not
# SYSTEM) INTERFACE_INCLUDE_DIRECTORIES, so every consumer that links
# it -- including tensora_lexer/tensora_parser below -- was compiling vendored
# headers such as RuleContext.h/ParseTree.h as if they were our own
# code, surfacing -Woverloaded-virtual/-Wunused-parameter warnings from
# a project we do not own. Re-declaring the same paths as
# INTERFACE_SYSTEM_INCLUDE_DIRECTORIES (a target property CMake reads
# specifically for this situation) fixes that without patching the
# vendored CMakeLists itself.
set_property(TARGET antlr4_static PROPERTY
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
  "$<TARGET_PROPERTY:antlr4_static,INTERFACE_INCLUDE_DIRECTORIES>")

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

# Lexer library: the project's own Lexer/Token wrapper (used by the
# standalone lexer demo/tests only -- Parser.cpp below drives the
# generated ANTLR lexer directly, per Parser.h's own comment) plus the
# generated ANTLR lexer. Kept as its own target, separate from
# tensora_parser below, so `cmake --build build --target lexer` (docs/
# development/BUILD.md) does not also compile any parser-only source.
add_library(tensora_lexer
  src/lexer/Lexer.cpp
  src/lexer/TokenAdapter.cpp
  ${ANTLR4_GEN_DIR}/TensoraLexer.cpp)
add_dependencies(tensora_lexer antlr4_generate)
target_include_directories(tensora_lexer PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tensora_lexer PRIVATE ${ANTLR4_GEN_DIR})
target_link_libraries(tensora_lexer PUBLIC antlr4_static)
# The generated headers include <antlr4-runtime.h>. So every consumer of
# the generated code (this library, and anything that links against it)
# needs the runtime's public include dir. The PUBLIC link to
# antlr4_static already carries that dir, through its own
# INTERFACE_INCLUDE_DIRECTORIES.

# Parser library: AstBuilder (walks ANTLR's parse tree into the AST),
# the parser error listener/strategy, and the generated ANTLR parser.
# Parser.cpp instantiates the generated ANTLR lexer itself (Parser.h's
# own comment: no project Token vector sits between the two here), and
# calls TokenAdapter's adaptToken() for its one diagnostic path -- both
# reasons this links tensora_lexer, which is otherwise correct anyway:
# a parser always needs a lexer.
add_library(tensora_parser
  src/parser/Parser.cpp
  src/parser/AstBuilder.cpp
  src/parser/ParserErrorListener.cpp
  src/parser/ParserErrorStrategy.cpp
  ${ANTLR4_GEN_DIR}/TensoraParser.cpp)
add_dependencies(tensora_parser antlr4_generate)
target_include_directories(tensora_parser PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tensora_parser PRIVATE ${ANTLR4_GEN_DIR})
target_link_libraries(tensora_parser PUBLIC tensora_lexer)

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
target_link_libraries(tensora_semantic PUBLIC tensora_parser)

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
add_executable(tensora src/driver/main.cpp src/driver/Npy.cpp)
target_include_directories(tensora SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(tensora PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(tensora PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")

# All test targets, CTest registration, and per-phase build aliases live in
# one place (see the file for the helpers and the data-driven registration).
include(${CMAKE_SOURCE_DIR}/cmake/Tests.cmake)
