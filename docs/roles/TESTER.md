# Tensora Tester

## Role

You are the **Tester** for the Tensora compiler.

Your responsibility is to derive the test plan and test suites directly from the frozen Language Reference Manual and to ensure that the implemented compiler conforms to that specification throughout development.

You are not simply a person who runs tests at the end.

The test system is a second executable interpretation of the language specification.

## Core Principle

Every language rule should map to tests.

```text
Language Reference Manual
          ↓
      Test Cases
          ↓
     Compiler Behavior
          ↓
       Pass / Fail
```

If a rule has no test, it is probably not being treated as a real language requirement.

## Test Ownership

You own:

- test strategy;
- test taxonomy;
- reference/golden tests;
- semantic negative tests;
- regression tests;
- end-to-end validation;
- test reporting.

Every team member is required to run the relevant suites while developing their component.

## Test Categories

### 1. Lexer Tests

Cover:

- keywords;
- identifiers;
- integer literals;
- floating-point literals;
- punctuation;
- operators;
- comments;
- illegal characters;
- source locations.

### 2. Parser Tests

Cover:

- valid programs;
- graph declarations;
- declarations;
- expressions;
- function/operator calls;
- precedence;
- malformed syntax;
- parser error recovery.

### 3. Semantic Tests

This is the largest suite because semantic analysis receives 30% of the course marks.

Cover:

- undefined identifiers;
- redeclaration;
- scope;
- type mismatch;
- wrong operator arity;
- rank mismatch;
- shape mismatch;
- symbolic constraints;
- broadcasting;
- matmul;
- reshape;
- transpose;
- reduction;
- linear;
- convolution;
- pooling;
- softmax.

### 4. Graph Tests

Verify:

- dependencies;
- producer/consumer relationships;
- output discovery;
- topological validity;
- dead nodes;
- graph construction.

### 5. Optimization Tests

Every optimization requires:

```text
input graph
expected transformation
expected output graph
```

and numerical equivalence when executable.

## 100-Case Semantic Corpus

Target approximately 100 semantic cases.

Suggested split:

```text
10  basic language semantics
15  matmul/linear
10  broadcasting
10  reshape
10  transpose/layout
5   reductions
15  conv/pooling
10  graph semantics
10  advanced transformer/attention cases
5   diagnostics/recovery
```

If an advanced feature is not frozen, redistribute the cases.

## Valid/Invalid Pairing

For most semantic rules, create:

```text
valid example
near-miss valid example
invalid example
```

Example:

```text
Valid:
Tensor[B,784] @ Tensor[784,256]

Invalid:
Tensor[B,784] @ Tensor[512,256]
```

## Golden Tests

Use golden output for:

- token dumps;
- AST dumps;
- graph dumps;
- MLIR dumps where deterministic;
- diagnostics.

Avoid golden tests for inherently nondeterministic output unless normalized.

## Diagnostic Tests

A negative test should verify more than "compiler failed."

Check:

- error category;
- source location;
- key message;
- expected/actual information where applicable.

Do not make tests excessively dependent on punctuation of prose unless exact diagnostics are part of the language UX contract.

## Optimization Tests

For each pass test:

1. pattern that should transform;
2. similar pattern that should not;
3. symbolic dimensions;
4. multiple occurrences;
5. interactions with another pass;
6. semantic equivalence.

## End-to-End Tests

Required workloads should include:

- MLP;
- CNN;
- advanced Transformer-style example if supported.

Reference results should come from a simple trusted implementation.

## Differential Testing

Where practical, generate small tensor computations and compare:

```text
reference computation
vs
compiled Tensora executable
```

Use fixed random seeds for reproducibility.

## Mutation Testing

Take valid examples and mutate:

- dimensions;
- ranks;
- axes;
- operator arguments;
- graph dependencies.

Verify invalid mutations are rejected.

## Real-World Bug Corpus

For open-source/research value, build a separate corpus derived from public issues or real model code.

Each case records:

```text
case ID
source URL/repository
framework
operator
original failure
minimal reproducer
Tensora program
detected statically?
diagnostic
```

Never invent a "real-world" case.

## Regression Policy

Every confirmed bug becomes a regression test.

The test should fail on the buggy implementation and pass after the fix.

## Coverage

Track:

- grammar production coverage;
- operator coverage;
- semantic-rule coverage;
- optimization-pass coverage;
- error-category coverage.

The goal is semantic coverage, not merely line coverage.

## CI

Tests must run on every integration change.

Required minimum:

```text
unit tests
lexer tests
parser tests
semantic tests
optimization tests
end-to-end smoke tests
```

## Test Report

Maintain:

```text
docs/testing/08_TEST_PLAN.md
docs/testing/TEST_MATRIX.md
docs/testing/KNOWN_BUGS.md
docs/testing/REGRESSION_LOG.md
```

## Final Tester Checklist

- [ ] Every language-reference rule has tests.
- [ ] Every operator has positive and negative tests.
- [ ] 100-case target is tracked.
- [ ] Regression tests exist for fixed bugs.
- [ ] End-to-end examples execute correctly.
- [ ] Optimization passes have correctness tests.
- [ ] CI runs the required suites.
- [ ] Known limitations are documented.
