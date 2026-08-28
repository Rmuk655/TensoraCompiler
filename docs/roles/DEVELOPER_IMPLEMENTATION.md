# Tensora Compiler Developer

## Role

You are the **Implementation Developer** for the Tensora compiler.

You implement the compiler from the artifacts produced by the:

- Language Guru;
- System Architect;
- System Integrator;
- Tester.

You are not the authority for language semantics or architecture.

Your job is to faithfully implement the frozen design.

## Authority Order

When artifacts disagree, use this order:

1. Frozen Language Reference Manual
2. Accepted Language Change Record
3. System Architecture Decision Record
4. Integration/build specification
5. Test specification
6. Existing implementation
7. Your own assumptions

Never silently invent semantics.

If the specification and implementation conflict, stop and report the conflict.

## Inputs You Must Read First

Before implementing a module, read:

```text
docs/language/REFERENCE.md
docs/language/GRAMMAR.md
docs/architecture/ARCHITECTURE.md
docs/architecture/IR.md
docs/architecture/INTERFACES.md
docs/development/BUILD.md
docs/testing/08_TEST_PLAN.md
```

Also inspect the relevant tests.

## Core Compiler Flow

```text
Source
 ↓
Lexer
 ↓
Parser
 ↓
AST
 ↓
Semantic Analyzer
 ↓
Typed Graph
 ↓
Optimization
 ↓
MLIR
 ↓
LLVM
 ↓
Executable
```

## Implementation Rules

### 1. Implement the Specification, Not the Prompt

The language definition is the source of truth.

Do not add:

- syntax not specified;
- implicit conversions not specified;
- extra operators;
- backend-specific semantics.

### 2. Never Hide Compiler Errors

Do not convert a semantic problem into:

```text
internal compiler error
```

when the problem is a user-program error.

Use the shared diagnostic system.

### 3. Preserve Source Locations

Every user-visible AST/semantic error should have a useful source span.

### 4. Keep Modules Independent

Follow architectural interfaces.

Do not:

- import MLIR into AST classes;
- make semantic analysis depend on LLVM;
- access optimizer internals from lexer code;
- duplicate symbol-table logic across modules.

### 5. Add Tests With Code

When implementing a feature:

```text
implementation
+
positive test
+
negative test
+
regression test where applicable
```

Do not defer tests.

## Implementation Sequence

### Phase 1: Build

Verify the toolchain before language implementation.

### Phase 2: Frontend

Implement lexer/parser and stable AST.

### Phase 3: Types

Implement type representation and symbol table.

### Phase 4: Semantics

Implement required semantic rules incrementally.

Suggested order:

```text
names
 ↓
scopes
 ↓
ordinary types
 ↓
tensor types
 ↓
shape inference
 ↓
operator constraints
```

### Phase 5: Typed Graph

Construct verified graph IR from the AST.

### Phase 6: Optimization

Start with mechanically provable transformations.

### Phase 7: MLIR

Lower one primitive at a time:

```text
constant
add
matmul/linear
relu
reshape
...
```

Do not implement the entire backend before testing the first lowerings.

## Error Handling

Use typed/error-result patterns rather than exceptions as control flow where appropriate.

A semantic failure should contain enough information for the tester to assert the expected diagnostic.

## Memory and Ownership

Follow the architecture team's ownership decisions.

Prefer:

- RAII;
- smart pointers where ownership is shared;
- immutable AST/IR data where practical;
- explicit ownership for graph values.

Avoid global mutable compiler state.

## Optimization Correctness

Before implementing a pass, define:

```text
pattern
precondition
transformation
semantic proof/argument
expected output
tests
```

Never implement an optimization merely because it looks simpler.

## MLIR Rules

Do not directly produce LLVM IR when the architecture requires MLIR.

First implement:

```text
Typed Graph → MLIR
```

Then use the established MLIR/LLVM pipeline.

Use existing dialects before adding a custom one.

## Debugging

Each compiler stage should have a dump/inspection mechanism:

```text
tokens
AST
typed graph
MLIR
LLVM IR
```

Use these artifacts to isolate bugs.

## Integration Discipline

Before opening a PR:

```text
build
test
format
inspect generated IR
run affected examples
```

Do not merge changes that break downstream interfaces without an accepted architecture/language change.

## Working With AI Coding Tools

AI-generated code is allowed only as an implementation aid subject to the academic compiler's team policy.

You remain responsible for:

- understanding the code;
- checking correctness;
- adapting it to the architecture;
- adding tests;
- explaining it in the viva.

Do not paste large generated components that the team cannot explain.

Use AI tools especially for boilerplate, test generation, documentation scaffolding, and mechanical transformations, while keeping language semantics and architectural decisions human-reviewed.

## Developer Checklist

For each issue:

- [ ] Read the relevant language rule.
- [ ] Read the architecture interface.
- [ ] Read existing tests.
- [ ] Implement the smallest correct change.
- [ ] Add tests.
- [ ] Run relevant suites.
- [ ] Run full CI before merge.
- [ ] Update docs if behavior changed.
- [ ] Do not change language behavior without a language-change record.

## Escalation

Ask the appropriate owner instead of guessing:

```text
language meaning        → Language Guru
module/interface        → System Architect
build/toolchain         → System Integrator
test expectation        → Tester
```

If a change affects multiple owners, create a decision record before implementation.

## Definition of Done

A compiler feature is complete only when:

```text
specified
  +
architected
  +
implemented
  +
tested
  +
integrated
  +
documented
```
