# Tensora Regression Log

> Every confirmed bug gets a regression test. The test must fail on the
> buggy build and pass after the fix. This log links each defect
> (KNOWN_BUGS.md) to its regression test ID. It was empty at project
> start.

## Policy

1. Reproduce the bug as a minimal test that **fails** on the current build.
2. Assign a regression ID (`REG-XXX`) and add the test under the owning
   suite in `tests/` (08_TEST_PLAN.md §2).
3. Fix the bug; the test must now pass.
4. Record the row below and mark the `BUG-XXX` entry `fixed -> REG-XXX`.
5. CI runs all regression tests on every integration change.

## Format

``` text
REG-XXX | BUG-XXX | phase | test path/ID | RM/ARCH ref | date fixed | 1-line description
```

## Log

```text
REG-001 | BUG-001 | lexer | tests/lexer/test_lexer.cpp LEX-011 | RM 3.1 / LANG-001 | 2026-08-18 | intLiteral wider than int64 now rejected with LexIntegerOverflow instead of an uncaught std::out_of_range from std::stoll
REG-002 | BUG-002 | parser | tests/parser/test_parser.cpp PAR-031 | RM 3.24 | 2026-08-19 | AstBuilder::buildOperand segfaulted on a bare int literal binop operand (float literal required); now reports a clean diagnostic
REG-003 | BUG-003 | parser | tests/parser/test_parser.cpp PAR-032 | RM 3.24 | 2026-08-19 | AstBuilder::buildDimSize segfaulted on an illegal axis dimSize token; now reports a clean diagnostic
```
