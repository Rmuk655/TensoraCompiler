# Tensora C++ Coding Guidelines

## Purpose

These are permanent coding guidelines for Tensora.

Apply them whenever project code is created, modified, generated, or refactored. The main goal is correct compiler code that a developer with basic C++ knowledge can read, maintain, test, and explain.

These guidelines do not define one-time migrations such as changing parser generators. One-time architecture changes belong in separate task documents.

## 1. Optimize for understanding

Optimize for low cognitive load, not for minimum line count or maximum abstraction.

Prefer:

- direct code
- clear names
- simple data structures
- shallow call chains
- explicit control flow
- small public interfaces
- one obvious representation for a concept
- one obvious place for a responsibility

Avoid clever code when simple code is sufficient.

## 2. Keep compiler phases clear

Organize code around these main phases:

```text
lexer
parser
semantic
optimization
codegen
```

Use `shared/` for data structures and services that multiple phases genuinely use.

Typical shared concepts can include:

- Token and TokenKind
- SourceSpan
- Diagnostic and DiagnosticEngine
- AST
- IR / TCG
- shared type representations

Do not place code in `shared/` only because its correct owner is unclear.

AST and IR are cross-phase representations. They are not compiler phases by themselves.

Diagnostics and source locations are shared infrastructure.

## 3. Prefer a simple directory structure

Keep `include`, `src`, and tests conceptually aligned with compiler phases.

Prefer a structure such as:

```text
include/tensora/
    lexer/
    parser/
    semantic/
    optimization/
    codegen/
    shared/

src/
    lexer/
    parser/
    semantic/
    optimization/
    codegen/
    shared/

tests/
    lexer/
    parser/
    semantic/
    optimization/
    codegen/
    integration/
```

Add special test grouping such as boundary, smoke, or performance only when it improves navigation.

Avoid deep directory nesting without enough files to justify it.

Do not create one directory per class.

## 4. Use modern C++ only when it improves the code

The project can use modern C++, but the code must remain understandable.

Review uses of:

- templates
- `std::unique_ptr`
- `std::shared_ptr`
- `std::variant`
- `std::span`
- advanced RAII
- complex lambdas
- metaprogramming
- type traits
- elaborate generic helpers
- excessive `std::move`

Keep an advanced feature when it solves a real ownership, safety, or design problem.

Do not replace safe ownership with raw pointers only to make syntax shorter.

Prefer `std::unique_ptr` when one object clearly owns another dynamic object.
Avoid `std::shared_ptr` unless ownership is genuinely shared.
Prefer values and references when dynamic ownership is not required.

## 5. Prefer simple data structures

Use a simple `struct` when the type primarily stores data.

Use a class when it has meaningful invariants, ownership, or behavior that should be protected.

Do not create getters and setters for every field without a reason.

Do not create inheritance hierarchies merely for extensibility.

For every class or struct, ask:

1. What compiler concept does it represent?
2. Which phase owns it?
3. Which other phases use it?
4. Does it protect an invariant?
5. Could a simpler structure represent the same concept?
6. Is it only a wrapper around another type?
7. Does it exist only because of obsolete tooling?

Remove types that no longer have a clear purpose.

## 6. Reduce duplicate representations

Trace information across:

```text
source -> token -> AST -> semantic representation -> IR -> optimized IR -> codegen
```

Look for unnecessary duplication of:

- token types
- source locations
- identifiers
- dimensions
- types
- symbols
- diagnostic data
- operator names
- argument representations

Merge representations only when they have the same meaning and lifetime.

Keep separate representations when phase separation or invariants require them.

For example, an AST and a typed IR should remain separate if the AST represents source syntax and the IR represents verified compiler operations.

## 7. Reduce boilerplate

Search for repeated:

- diagnostic construction
- source-span handling
- token conversion
- AST initialization
- test registration
- test reporting
- command-line selection
- CMake target configuration
- error formatting

Use a small helper when it makes repeated code easier to understand.

Prefer:

```cpp
reportError(code, span, message);
```

over repeated blocks that manually construct the same diagnostic fields.

Do not replace a few repeated lines with a complicated framework.

Avoid unnecessary Factory, Manager, Builder, Registry, Service, Provider, or Adapter layers.

Use such abstractions only when they have a clear repeated purpose.

## 8. Keep interfaces small

A phase should expose only what the next phase needs.

Do not expose private implementation details across phase boundaries.

Prefer:

```text
Lexer: source -> tokens
Parser: tokens -> AST
Semantic: AST -> verified IR
Optimization: IR -> optimized IR
Codegen: optimized IR -> backend representation
```

Do not make later phases depend on parser-generator internals.

Do not expose MLIR objects to the AST or semantic analyzer.

## 9. Keep AST responsibilities limited

The AST stores source program structure.

Do not make AST nodes responsible for:

- type checking
- shape inference
- optimization
- MLIR construction
- LLVM construction

Store source spans when later diagnostics need them.

Keep the AST hierarchy as small as the language requires.

## 10. Keep IR responsibilities clear

The IR / TCG represents verified operations for optimization and code generation.

A reader should be able to identify:

- who creates the IR
- what one IR node means
- how values connect
- which optimization changes it
- how code generation consumes it

Do not preserve duplicate IR layers without a demonstrated need.

## 11. Use one diagnostic system

All compiler phases should use one clear diagnostic representation when practical.

A diagnostic should have only the information required by the project, such as:

```text
severity
code
source span
message
```

Add notes or structured expected/actual data only when tests or user diagnostics need them.

Use simple shared reporting helpers to reduce repeated setup.

## 12. Use one source-location representation

Prefer one shared `SourceSpan` representation across project-owned compiler phases.

Do not create lexer, parser, semantic, and IR location types unless their information is genuinely different.

Generated third-party tools can have their own location representation at the integration boundary. Convert it to the project representation in one clear place.

## 13. Error handling must be explicit

Do not hide user-program errors as internal compiler errors.

Do not use exceptions as ordinary compiler control flow when a simple result or diagnostic path is clearer.

Recovery logic must preserve diagnostics and avoid silently accepting invalid programs.

## 14. Tests must be easy to add

Centralize test harness boilerplate.

Individual tests should focus on:

- test ID
- category
- input
- expected output or diagnostic
- expected pass/fail behavior

Keep registration, execution, selection, and reporting in shared test infrastructure.

Preserve useful CTest integration.

Test categories can include:

- UNIT
- SMOKE
- BOUNDARY
- INTEGRATION
- PERFORMANCE

Do not build a large custom testing framework when a small harness is sufficient.

## 15. Keep CMake understandable

Use clear target names and explicit dependencies.

Avoid unnecessary CMake functions, macros, and custom machinery.

Generated third-party frontend code should be reproducible and isolated from hand-written compiler logic.

Do not commit machine-specific absolute paths.

Keep the normal workflow simple:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## 16. Naming

Use one term for one concept.

Prefer clear names such as:

- Lexer
- Parser
- Token
- SourceSpan
- AstProgram
- SemanticAnalyzer
- Diagnostic
- IrNode

Avoid unexplained abbreviations.

Use `AST`, `IR`, `MLIR`, `LLVM`, and `TCG` only when the project already defines those terms.

Do not rename the same concept differently in different phases without a semantic reason.

## 17. No unnecessary abstraction

Do not preserve complexity because it looks professional or extensible.

Before adding an abstraction, ask:

- Is it used more than once?
- Does it protect an important invariant?
- Does it make the code easier to understand?
- Does it remove meaningful repetition?
- Can someone else explain why it exists?

If not, prefer direct code.

## 18. No monolithic simplification

Simpler code does not mean one large `Compiler.cpp`.

Preserve clear compiler phase boundaries.

Prefer a small number of understandable modules over either extreme:

- hundreds of tiny wrappers
- one giant compiler module

## 19. Generated code

Do not manually simplify third-party generated source code.

Keep generated code separate from project-owned code.

Apply these coding guidelines to:

- hand-written C++
- project-owned generated/scaffolded C++
- test harnesses
- CMake maintained by the project

Do not edit generated ANTLR/MLIR/LLVM implementation files merely to match project style.

## 20. Review standard

Before finishing a code change, ask:

- Is the code easier to understand than before?
- Is each responsibility in the correct phase?
- Did I introduce an unnecessary type or layer?
- Did I duplicate an existing representation?
- Is ownership clear?
- Is the control flow clear?
- Is repeated boilerplate reduced without hiding behavior?
- Can someone else explain this code without you?

Follow the mandatory build and test rules in `CLAUDE.md`.
