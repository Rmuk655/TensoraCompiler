# Tensora Test Matrix

> Maps every Language Reference Manual rule to test cases. Kept in sync
> with [../language/03_LANGUAGE_REFERENCE_MANUAL.md](../language/03_LANGUAGE_REFERENCE_MANUAL.md)
> and [08_TEST_PLAN.md](08_TEST_PLAN.md). Status: `planned` until the case
> exists and runs in CI. A rule with no row is a coverage gap.

## Legend

- **Result**: `accept` (compiles clean) or `reject` (+ expected `DiagCode`).
- **Kind**: `V` valid, `N` near-miss valid, `X` invalid.
- IDs are stable and referenced by the corpus and regression log.
- **Test**: link to the file (and line, where the case is a single named
  `CASE`) that implements the row. `—` means no case exists yet
  (`Status: planned`) or the row is a multi-file/aggregate claim
  described in the prose below its table.

## Frontend — Lexer (RM 3.1)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| LEX-001 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | V | keywords + identifiers, incl. keyword-prefix maximal munch | tokens | done | [test_lexer.cpp:46](../../tests/lexer/test_lexer.cpp#L46) |
| LEX-002 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | V | int and float literal values, incl. leading-zero int | tokens | done | [test_lexer.cpp:72](../../tests/lexer/test_lexer.cpp#L72) |
| LEX-003 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | V | every operator/delimiter, spaced | tokens | done | [test_lexer.cpp:87](../../tests/lexer/test_lexer.cpp#L87) |
| LEX-004 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | V | every operator/delimiter with no whitespace (adjacency) | tokens | done | [test_lexer.cpp:106](../../tests/lexer/test_lexer.cpp#L106) |
| LEX-005 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | V | comment variants (empty, no-newline, back-to-back, contents ignored) | tokens | done | [test_lexer.cpp:128](../../tests/lexer/test_lexer.cpp#L128) |
| LEX-006 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | V | whitespace/newline insensitivity across separator styles | kinds equal | done | [test_lexer.cpp:173](../../tests/lexer/test_lexer.cpp#L173) |
| LEX-007 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | X | malformed floats `3.` `.5` `3.x` `1..2` | reject: LexIllegalChar | done | [test_lexer.cpp:198](../../tests/lexer/test_lexer.cpp#L198) |
| LEX-008 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | X | illegal-char variety incl. 7 in a row; scanning continues | reject: LexIllegalChar | done | [test_lexer.cpp:240](../../tests/lexer/test_lexer.cpp#L240) |
| LEX-009 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | V | `SourceSpan` line/col/offset incl. tabs + CRLF; lexeme == source slice | spans correct | done | [test_lexer.cpp:266](../../tests/lexer/test_lexer.cpp#L266) |
| LEX-010 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | V | full representative program, golden token-kind dump | tokens | done | [test_lexer.cpp:311](../../tests/lexer/test_lexer.cpp#L311) |
| LEX-011 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | X | int literal at/over int64 max (LANG-001; fixed REG-001) | reject: LexIntegerOverflow | done | [test_lexer.cpp:359](../../tests/lexer/test_lexer.cpp#L359) |

Boundary cases for RM 3.1 are tracked as `BND-LEX-*` in the Boundary
section below (empty file, comments-only file, token at EOF, first-token
position).

## Frontend — Parser (RM 3.2, 3.3, 3.24)

Audited against the RM 3.24 grammar summary (the authoritative syntax).
Every production and alternative below maps to at least one case; the
previous 8-row table only exercised a handful of representative shapes
and no error-recovery or grammar-fidelity (decl\* before graphDecl+)
cases, so it did not meet 08_TEST_PLAN.md's "exhaustive when: every AST
production has a direct test" criterion (§2a Unit).

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| PAR-001 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | MLP program (whitepaper 10.1) -> AST golden | ast | done | [test_parser.cpp:55](../../tests/parser/test_parser.cpp#L55) |
| PAR-002 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | CNN program (whitepaper 10.2) -> AST golden | ast | done | [test_parser.cpp:90](../../tests/parser/test_parser.cpp#L90) |
| PAR-003 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | Transformer FFN program (whitepaper 10.3) -> AST golden | ast | done | [test_parser.cpp:127](../../tests/parser/test_parser.cpp#L127) |
| PAR-004 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | multi-head attention program (whitepaper 10.4) -> AST golden | ast | done | [test_parser.cpp:145](../../tests/parser/test_parser.cpp#L145) |
| PAR-005 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | V | `tensorType`/`axis`: 1 axis, 2 axes, 4 axes; concrete and symbolic `dimSize` | ast | done | [test_parser.cpp:183](../../tests/parser/test_parser.cpp#L183) |
| PAR-006 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure) | V | `decl`: `inputDecl` alone, `weightDecl` alone, both, multiple of each | ast | done | [test_parser.cpp:206](../../tests/parser/test_parser.cpp#L206) |
| PAR-007 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | `program`: multiple `graphDecl`s (`graphDecl+` repetition) | ast | done | [test_parser.cpp:223](../../tests/parser/test_parser.cpp#L223) |
| PAR-008 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure) | V | `statement`: two statements with no terminator; a bare-identifier statement (`y = h`) | ast | done | [test_parser.cpp:240](../../tests/parser/test_parser.cpp#L240) |
| PAR-009 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | `expression`/`binop`: each of `@ + - * /` with identifier and float-literal `operand` | ast | done | [test_parser.cpp:259](../../tests/parser/test_parser.cpp#L259) |
| PAR-010 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | `callExpr`: one positional arg, multiple positional args | ast | done | [test_parser.cpp:286](../../tests/parser/test_parser.cpp#L286) |
| PAR-011 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | `arg` kinds: keyword arg (int value), keyword arg (identifier value), reshape axis-list arg, all three mixed in one call | ast | done | [test_parser.cpp:303](../../tests/parser/test_parser.cpp#L303) |
| PAR-012 | [3.12](../language/03_LANGUAGE_REFERENCE_MANUAL.md#312-reshape), [3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | `argValue` kinds: `intLit` (`stride=1`), identifier (`axis=kseq`), `intList` (`perm=[0,1,3,2]`) | ast | done | [test_parser.cpp:333](../../tests/parser/test_parser.cpp#L333) |
| PAR-013 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | V | `conv2d`-shaped call: multiple positional + multiple keyword args together | ast | done | [test_parser.cpp:354](../../tests/parser/test_parser.cpp#L354) |
| PAR-014 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure) | X | missing `=` in a statement | reject: ParseExpected | done | [parser.tns:12](../../tests/fixtures/parser.tns#L12) |
| PAR-015 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | X | missing `]` closing a `tensorType` | reject: ParseExpected | done | [parser.tns:19](../../tests/fixtures/parser.tns#L19) |
| PAR-016 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | X | missing `[` opening a `tensorType` | reject: ParseExpected | done | [parser.tns:24](../../tests/fixtures/parser.tns#L24) |
| PAR-017 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | X | empty `tensorType` (`Tensor[]`; `axis` list requires >= 1) | reject: ParseExpected | done | [parser.tns:29](../../tests/fixtures/parser.tns#L29) |
| PAR-018 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | X | missing `=` in an `axis` | reject: ParseExpected | done | [parser.tns:34](../../tests/fixtures/parser.tns#L34) |
| PAR-019 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | `graphDecl` missing `returnStmt` before `}` | reject: ParseExpected | done | [parser.tns:39](../../tests/fixtures/parser.tns#L39) |
| PAR-020 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure) | X | `graphDecl` missing closing `}` | reject: ParseExpected | done | [test_parser.cpp:369](../../tests/parser/test_parser.cpp#L369) |
| PAR-021 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | `returnStmt` missing its identifier | reject: ParseExpected | done | [parser.tns:45](../../tests/fixtures/parser.tns#L45) |
| PAR-022 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | zero-arg call (`relu()`; `arg` list requires >= 1) | reject: ParseExpected | done | [parser.tns:51](../../tests/fixtures/parser.tns#L51) |
| PAR-023 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | missing closing `)` on a call | reject: ParseExpected | done | [parser.tns:58](../../tests/fixtures/parser.tns#L58) |
| PAR-024 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | trailing comma in an arg list (`relu(h,)`) | reject: ParseExpected | done | [parser.tns:65](../../tests/fixtures/parser.tns#L65) |
| PAR-025 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | keyword arg with no value (`stride=)`) | reject: ParseExpected | done | [parser.tns:72](../../tests/fixtures/parser.tns#L72) |
| PAR-026 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | a `decl` after a `graphDecl` (violates `decl* graphDecl+` ordering) | reject: ParseExpected | done | [parser.tns:79](../../tests/fixtures/parser.tns#L79) |
| PAR-027 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | a program with zero `graphDecl`s (`graphDecl+` requires >= 1) | reject: ParseExpected | done | [parser.tns:84](../../tests/fixtures/parser.tns#L84) |
| PAR-028 | recovery | X | two independent syntax errors in different sections (one in a `decl`, one in a graph body) both reported | 2 diagnostics | done | [test_parser.cpp:379](../../tests/parser/test_parser.cpp#L379) |
| PAR-029 | recovery | X | two independent malformed statements in the same graph body both reported | 2 diagnostics | done | [test_parser.cpp:404](../../tests/parser/test_parser.cpp#L404) |
| PAR-030 | span | X | a missing-token diagnostic's `SourceSpan` (line/col) points at the actual offending token | span correct | done | [test_parser.cpp:416](../../tests/parser/test_parser.cpp#L416) |
| PAR-031 | regression | X | bare int literal as a binop operand (BUG-002/REG-002: used to segfault) | reject: diagnostic, not a crash | done | [test_parser.cpp:435](../../tests/parser/test_parser.cpp#L435) |
| PAR-032 | regression | X | illegal axis `dimSize` token (BUG-003/REG-003: used to segfault) | reject: diagnostic, not a crash | done | [test_parser.cpp:447](../../tests/parser/test_parser.cpp#L447) |

Boundary cases for RM 3.2/3.3/3.24 are tracked as `BND-PAR-*` below (empty
statement list, minimal program, axis-count and arg-count extremes).

**Known coverage gap:** `DiagCode::ParseUnexpectedToken` is declared but
currently unreachable. `ParserErrorListener` (`src/parser/`) genuinely
distinguishes the two syntax-error shapes ANTLR can report: a
`NoViableAltException` (no expected token can be named) becomes
`ParseUnexpectedToken`; every other `RecognitionException` (a mismatched-
or missing-token error, which does name an expected set) becomes
`ParseExpected`. Every malformed input tried against this grammar --
including cases designed to look ambiguous, like a bare `f(@)` or an
empty `Tensor[]` -- still resolves to a mismatched-token error, because
ANTLR's adaptive LL(*) prediction picks the one viable alternative before
failing, in every state this small (~24-token) grammar can reach. Not
fixed further here since forcing `NoViableAltException` would require
deliberately engineering an unrealistic grammar state rather than testing
real language behavior.

## Semantic — required academic-compiler checks (RM 3.6, 3.8, 3.7–3.17)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| SEM-DECL-001 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | duplicate global name | reject: SemaRedeclaration | done | [semantic.tns:12](../../tests/fixtures/semantic.tns#L12) |
| SEM-DECL-002 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | duplicate graph name | reject: SemaRedeclaration | done | [semantic.tns:18](../../tests/fixtures/semantic.tns#L18) |
| SEM-DECL-003 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | graph-local assignment shadows a global input/weight name (different kind) | reject: SemaRedeclaration | done | [semantic.tns:24](../../tests/fixtures/semantic.tns#L24) |
| SEM-DECL-004 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | X | duplicate axis name within one tensor type (ARCH-013) | reject: SemaDuplicateAxisName | done | [semantic.tns:32](../../tests/fixtures/semantic.tns#L32) |
| SEM-DECL-005 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | X | duplicate axis name in a `reshape` target axis list (ARCH-013) | reject: SemaDuplicateAxisName | done | [semantic.tns:37](../../tests/fixtures/semantic.tns#L37) |
| SEM-NAME-001 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | use of undefined name | reject: SemaUndefinedName | done | [semantic.tns:45](../../tests/fixtures/semantic.tns#L45) |
| SEM-NAME-002 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | `return` of unbound name | reject: SemaUndefinedName | done | [semantic.tns:52](../../tests/fixtures/semantic.tns#L52) |
| SEM-NAME-003 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | call to an operator name outside the frozen set | reject: SemaUndefinedName | done | [semantic.tns:60](../../tests/fixtures/semantic.tns#L60) |
| SEM-SCOPE-001 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | V | rebinding `h = relu(h)` (shadowing) | accept | done | [test_semantic.cpp:53](../../tests/semantic/test_semantic.cpp#L53) |
| SEM-SCOPE-002 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | name from graph A used in graph B | reject: SemaUndefinedName | done | [semantic.tns:68](../../tests/fixtures/semantic.tns#L68) |
| SYM-001 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | V | `SemanticAnalyzer::symbolTable()`: globals recorded in declaration order with correct kind/type | 2 globals, `x`(Input) then `W`(Weight), verified decl type | done | [test_semantic.cpp:371](../../tests/semantic/test_semantic.cpp#L371) |
| SYM-002 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | V | graph-local symbol recorded with its inferred (not declared) type | `h`, kind Value, relu's inferred type | done | [test_semantic.cpp:392](../../tests/semantic/test_semantic.cpp#L392) |
| SYM-003 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | V | shadowing (`h = relu(h)`) collapses to one recorded local entry | 1 entry, not 2 | done | [test_semantic.cpp:412](../../tests/semantic/test_semantic.cpp#L412) |
| SYM-004 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | V | resolution log: `global::`/`local::`, deduped per distinct referenced name | 3 entries for x/h/y referenced across 3 statements | done | [test_semantic.cpp:427](../../tests/semantic/test_semantic.cpp#L427) |
| SYM-005 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | resolution log records an unresolved reference as `undefined`, not silently absent | logged, no crash | done | [test_semantic.cpp:446](../../tests/semantic/test_semantic.cpp#L446) |
| SYM-006 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | a rejected global redeclaration never enters the recorded global scope | 1 entry (the original), not 2 | done | [test_semantic.cpp:464](../../tests/semantic/test_semantic.cpp#L464) |
| SYM-007 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | X | a rejected local rebind of a global name never enters the recorded local scope | 0 local entries | done | [test_semantic.cpp:479](../../tests/semantic/test_semantic.cpp#L479) |
| SYM-008 | [3.6](../language/03_LANGUAGE_REFERENCE_MANUAL.md#36-name-and-scope-rules) | V | multiple graphs each get their own recorded scope, in program order | 2 graph scopes, no cross-graph leakage | done | [test_semantic.cpp:493](../../tests/semantic/test_semantic.cpp#L493) |
| SEM-TYPE-001 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | X | scalar where tensor operand required (`x @ 0.5`) | reject: SemaTypeMismatch | done | [semantic.tns:75](../../tests/fixtures/semantic.tns#L75) |
| SEM-ARITY-001 | [3.9](../language/03_LANGUAGE_REFERENCE_MANUAL.md#39-linear) | X | `linear` with 2 args | reject: SemaArgCount | done | [semantic.tns:83](../../tests/fixtures/semantic.tns#L83) |
| SEM-ARITY-002 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d) | X | `conv2d` missing `stride` | reject: SemaArgCount | done | [semantic.tns:92](../../tests/fixtures/semantic.tns#L92) |
| SEM-ARITY-003 | [3.11](../language/03_LANGUAGE_REFERENCE_MANUAL.md#311-softmax) | X | `softmax` axis given as integer | reject: SemaArgKind | done | [semantic.tns:102](../../tests/fixtures/semantic.tns#L102) |
| SEM-ARITY-004 | [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | X | `relu` with an unknown keyword argument | reject: SemaArgKind | done | [semantic.tns:110](../../tests/fixtures/semantic.tns#L110) |
| SEM-ARITY-005 | [3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose) | X | `transpose` missing `perm` | reject: SemaArgCount | done | [semantic.tns:118](../../tests/fixtures/semantic.tns#L118) |
| SEM-POISON-001 | ARCH-007 | X | one bad statement does not cascade into spurious follow-on errors later in the same graph | 1 diagnostic only | done | [test_semantic.cpp:63](../../tests/semantic/test_semantic.cpp#L63) |
| SEM-POISON-002 | recovery | X | two independent semantic errors (different graphs) are both reported in one pass | 2 diagnostics | done | [test_semantic.cpp:73](../../tests/semantic/test_semantic.cpp#L73) |

## Semantic — matmul / linear (RM 3.7, 3.9)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| SEM-MM-001 | [3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication) | V | `[batch=B,feat=784]@[in=784,out=256]` | accept -> `[batch=B,out=256]` | done | [test_semantic.cpp:83](../../tests/semantic/test_semantic.cpp#L83) |
| SEM-MM-002 | [3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication) | X | `784 != 512` | reject: SemaShapeMismatch (784!=512) | done | [semantic.tns:126](../../tests/fixtures/semantic.tns#L126) |
| SEM-MM-003 | [3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication) | N | symbolic contraction `K` unifies | accept | done | [semantic.tns:135](../../tests/fixtures/semantic.tns#L135) |
| SEM-MM-004 | [3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication) | X | batched leading-axis mismatch | reject: SemaShapeMismatch | done | [semantic.tns:144](../../tests/fixtures/semantic.tns#L144) |
| SEM-LIN-001 | [3.9](../language/03_LANGUAGE_REFERENCE_MANUAL.md#39-linear) | V | linear with matching bias | accept | done | [test_semantic.cpp:96](../../tests/semantic/test_semantic.cpp#L96) |
| SEM-LIN-002 | [3.9](../language/03_LANGUAGE_REFERENCE_MANUAL.md#39-linear) | X | bias size mismatch | reject: SemaShapeMismatch | done | [semantic.tns:153](../../tests/fixtures/semantic.tns#L153) |
| SEM-LIN-003 | [3.9](../language/03_LANGUAGE_REFERENCE_MANUAL.md#39-linear) | X | bias axis-name mismatch | reject: SemaAxisNameMismatch | done | [semantic.tns:163](../../tests/fixtures/semantic.tns#L163) |

## Semantic — elementwise / broadcasting / axis names (RM 3.8)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| SEM-EW-001 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | V | equal shapes and names | accept | done | [semantic.tns:173](../../tests/fixtures/semantic.tns#L173) |
| SEM-EW-002 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | V | size-1 broadcast | accept | done | [test_semantic.cpp:108](../../tests/semantic/test_semantic.cpp#L108) |
| SEM-EW-003 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | X | **equal sizes, swapped axis names** | reject: SemaAxisNameMismatch | done | [semantic.tns:182](../../tests/fixtures/semantic.tns#L182) |
| SEM-EW-004 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | X | incompatible sizes | reject: SemaBroadcast | done | [semantic.tns:191](../../tests/fixtures/semantic.tns#L191) |
| SEM-EW-005 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | V | tensor-scalar `x * 0.125` | accept | done | [test_semantic.cpp:120](../../tests/semantic/test_semantic.cpp#L120) |

## Semantic — reshape / flatten (RM 3.12, 3.13)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| SEM-RS-001 | [3.12](../language/03_LANGUAGE_REFERENCE_MANUAL.md#312-reshape) | V | `768 -> [head=12,dim=64]` | accept | done | [test_semantic.cpp:131](../../tests/semantic/test_semantic.cpp#L131) |
| SEM-RS-002 | [3.12](../language/03_LANGUAGE_REFERENCE_MANUAL.md#312-reshape) | X | `768 -> [head=10,dim=64]` (640) | reject: SemaReshapeCount (640!=768) | done | [semantic.tns:200](../../tests/fixtures/semantic.tns#L200) |
| SEM-RS-003 | [3.4](../language/03_LANGUAGE_REFERENCE_MANUAL.md#34-symbolic-dimensions), [3.12](../language/03_LANGUAGE_REFERENCE_MANUAL.md#312-reshape) | X | symbolic-factor multiset mismatch (`[batch=B,feat=768]` vs `[batch=C,head=12,dim=64]`, unrelated symbols) -- outside the bounded subset (RM 3.4), not a decided count mismatch | reject: SemaUnresolvableShape | done | [semantic.tns:208](../../tests/fixtures/semantic.tns#L208) |
| SEM-RS-004 | [3.12](../language/03_LANGUAGE_REFERENCE_MANUAL.md#312-reshape) | N | symbolic factor cancels correctly (`[batch=B,feat=768]` -> `[batch=B,head=12,dim=64]`), near-miss of RS-003 | accept | done | [semantic.tns:216](../../tests/fixtures/semantic.tns#L216) |
| SEM-FL-001 | [3.13](../language/03_LANGUAGE_REFERENCE_MANUAL.md#313-flatten) | V | concrete tail flattens to `flat=200704` | accept | done | [test_semantic.cpp:144](../../tests/semantic/test_semantic.cpp#L144) |
| SEM-FL-002 | [3.13](../language/03_LANGUAGE_REFERENCE_MANUAL.md#313-flatten) | X | symbolic tail dim | reject: SemaUnresolvableShape | done | [semantic.tns:224](../../tests/fixtures/semantic.tns#L224) |

## Semantic — transpose / reductions / softmax (RM 3.14, 3.11, 3.15)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| SEM-TR-001 | [3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose) | V | valid perm on rank-4 | accept | done | [test_semantic.cpp:160](../../tests/semantic/test_semantic.cpp#L160) |
| SEM-TR-002 | [3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose) | X | repeated perm index | reject: SemaTransposePerm | done | [semantic.tns:232](../../tests/fixtures/semantic.tns#L232) |
| SEM-TR-003 | [3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose) | X | out-of-range perm index | reject: SemaTransposePerm | done | [semantic.tns:240](../../tests/fixtures/semantic.tns#L240) |
| SEM-RD-001 | [3.15](../language/03_LANGUAGE_REFERENCE_MANUAL.md#315-sum--mean) | V | `sum(x, axis=feat)` existing axis | accept (axis removed) | done | [test_semantic.cpp:172](../../tests/semantic/test_semantic.cpp#L172) |
| SEM-RD-002 | [3.15](../language/03_LANGUAGE_REFERENCE_MANUAL.md#315-sum--mean) | X | nonexistent axis name | reject: SemaAxisNotFound | done | [semantic.tns:248](../../tests/fixtures/semantic.tns#L248) |
| SEM-SM-001 | [3.11](../language/03_LANGUAGE_REFERENCE_MANUAL.md#311-softmax) | V | `softmax(scores, axis=kseq)` | accept | done | [test_semantic.cpp:182](../../tests/semantic/test_semantic.cpp#L182) |
| SEM-SM-002 | [3.11](../language/03_LANGUAGE_REFERENCE_MANUAL.md#311-softmax) | X | softmax nonexistent axis | reject: SemaAxisNotFound | done | [semantic.tns:256](../../tests/fixtures/semantic.tns#L256) |

## Semantic — conv / pooling (RM 3.16, 3.17)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| SEM-CV-001 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d) | V | conv `stride=1,padding=1` preserves HxW | accept | done | [test_semantic.cpp:195](../../tests/semantic/test_semantic.cpp#L195) |
| SEM-CV-002 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d) | X | channel mismatch `3 != 16` | reject: SemaConvChannel | done | [semantic.tns:264](../../tests/fixtures/semantic.tns#L264) |
| SEM-CV-003 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d) | X | non-dividing spatial arithmetic | reject: SemaConvArithmetic | done | [semantic.tns:274](../../tests/fixtures/semantic.tns#L274) |
| SEM-CV-004 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d) | X | stride=2 at 224x224 that would only floor-divide (a common-framework convention) is rejected, not floored | reject: SemaConvArithmetic | done | [semantic.tns:284](../../tests/fixtures/semantic.tns#L284) |
| SEM-CV-005 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d) | V | stride=2 at 224x224 exactly halves H/W when the division is exact | accept, output [64,112,112] | done | [test_semantic.cpp:209](../../tests/semantic/test_semantic.cpp#L209) |
| SEM-PL-001 | [3.17](../language/03_LANGUAGE_REFERENCE_MANUAL.md#317-maxpool) | V | `maxpool size=2` halves HxW | accept | done | [test_semantic.cpp:223](../../tests/semantic/test_semantic.cpp#L223) |
| SEM-PL-002 | [3.17](../language/03_LANGUAGE_REFERENCE_MANUAL.md#317-maxpool) | N | `stride` defaults to `size` | accept | done | [semantic.tns:294](../../tests/fixtures/semantic.tns#L294) |
| SEM-PL-003 | [3.17](../language/03_LANGUAGE_REFERENCE_MANUAL.md#317-maxpool) | X | `size=0` (a zero-sized pooling window) is rejected, not silently accepted | reject: SemaConvArithmetic | done | [semantic.tns:302](../../tests/fixtures/semantic.tns#L302) |

## Semantic — bounded solver (RM 3.4)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| SEM-DIM-001 | [3.4](../language/03_LANGUAGE_REFERENCE_MANUAL.md#34-symbolic-dimensions) | V | equality/unification of symbolic `B` | accept | done | [semantic.tns:302](../../tests/fixtures/semantic.tns#L302) |
| SEM-DIM-002 | [3.4](../language/03_LANGUAGE_REFERENCE_MANUAL.md#34-symbolic-dimensions) | X | requires nonlinear reasoning (same shape as SEM-RS-003, exercised generically) | reject: SemaUnresolvableShape | done | [semantic.tns:311](../../tests/fixtures/semantic.tns#L311) |
| SEM-DIM-003 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | X | a zero-sized dimension (ARCH-013) | reject: SemaNonPositiveDimension | done | [semantic.tns:319](../../tests/fixtures/semantic.tns#L319) |

**SEM-RS-003 / SEM-DIM-002 note:** `DimSolver::sameElementCount` (refining
`INTERFACES.md`'s illustrative `bool` signature to an
`ElementCountResult`, see `shared/DimSolver.h`) distinguishes two failure
modes: symbolic factors that match in identity/multiplicity but whose
constant products differ (`SemaReshapeCount`, a *decided* mismatch --
SEM-RS-002) vs. symbolic factors that don't match at all, which is
outside the bounded affine-in-one-variable subset (RM 3.4) and always
`SemaUnresolvableShape` regardless of which operator triggered it
(SEM-RS-003, SEM-DIM-002). This corrects the matrix's earlier draft,
which had listed SEM-RS-003 as `SemaReshapeCount`.

## Semantic — attention / multi-head (whitepaper §10.4)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| ATT-001 | [3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication), [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting), [3.11](../language/03_LANGUAGE_REFERENCE_MANUAL.md#311-softmax), [3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose) | V | the full attention program (whitepaper 10.4) type-checks end to end: batched `@`, rank-4 `transpose`, scalar `*`, `softmax(axis=kseq)` | accept | done | [test_semantic.cpp:238](../../tests/semantic/test_semantic.cpp#L238) |
| ATT-002 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | X | mutated ATT-001: swap `Q`'s and `K`'s leading axis names (`batch`/`head` transposed on one operand) so the batched matmul's leading-axis broadcast fails | reject: SemaAxisNameMismatch | done | [semantic.tns:324](../../tests/fixtures/semantic.tns#L324) |

## Graph (RM 3.18, 3.20; IR.md)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| GR-001 | [3.20](../language/03_LANGUAGE_REFERENCE_MANUAL.md#320-typed-computation-graph) | V | MLP program analyzed end to end: golden structural TCG (node count/kinds/order, param count) | graph | done | [test_graph.cpp:33](../../tests/semantic/test_graph.cpp#L33) |
| GR-002 | IR §1 | X | hand-built `GraphBuilder` fixture with a cycle | GraphVerifier reject: IrInvariantViolation | done | [test_graph.cpp:52](../../tests/semantic/test_graph.cpp#L52) |
| GR-003 | IR §1 | X | hand-built fixture with a dangling operand (out-of-range ValueId) | GraphVerifier reject: IrInvariantViolation | done | [test_graph.cpp:74](../../tests/semantic/test_graph.cpp#L74) |
| GR-004 | [3.18](../language/03_LANGUAGE_REFERENCE_MANUAL.md#318-graph-semantics) | V | producer/consumer queries correct on a small hand-built chain | accept | done | [test_graph.cpp:91](../../tests/semantic/test_graph.cpp#L91) |
| GR-005 | [3.20](../language/03_LANGUAGE_REFERENCE_MANUAL.md#320-typed-computation-graph) | V | single reachable output; every node in topological order | accept | done | [test_graph.cpp:113](../../tests/semantic/test_graph.cpp#L113) |
| GR-006 | ARCH-008 | V | a `GraphVerifier`-clean fixture built entirely by hand (no sema) passes, demonstrating the mock-enabler contract | accept | done | [test_graph.cpp:129](../../tests/semantic/test_graph.cpp#L129) |
| GR-007 | [3.18](../language/03_LANGUAGE_REFERENCE_MANUAL.md#318-graph-semantics) | V | a graph referencing the same global input twice adds it as exactly one param (lazy dedup) | 1 param, 2 consumers | done | [test_graph.cpp:154](../../tests/semantic/test_graph.cpp#L154) |

`--emit-graph`'s textual dump format (IR.md §7) is not implemented by this
phase (driver/CLI work); GR-001's "golden" comparison is a structural
C++ assertion over the built `TypedGraph`, matching how parser golden
tests already work in this project (no invented textual format).

## Optimization — per pass (RM 3.21)

| ID | Pass | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| OPT-SF-001, OPT-SF-002 | ShapeFolding | V/N | constant shape -> literal / symbolic shape untouched | changed / unchanged | done | [test_passes.cpp:28](../../tests/optimization/test_passes.cpp#L28), [test_passes.cpp:50](../../tests/optimization/test_passes.cpp#L50) |
| OPT-DCE-001 | DeadOpElimination | V | drops value not reaching return | changed | done | [test_passes.cpp:70](../../tests/optimization/test_passes.cpp#L70) |
| OPT-DCE-002 | DeadOpElimination | N | keeps value reaching return | unchanged | done | [test_passes.cpp:90](../../tests/optimization/test_passes.cpp#L90) |
| OPT-DCE-003 | DeadOpElimination | V | removes a whole dead chain (2 nodes) next to a same-shaped live chain, not just one dead node | changed (2 nodes removed) | done | [test_passes.cpp:111](../../tests/optimization/test_passes.cpp#L111) |
| OPT-RC-001, OPT-RC-002, OPT-RC-003 | ReshapeCancellation | V/V/N | reshape immediately undone / chained / non-inverse shapes | changed / changed / unchanged | done | [test_passes.cpp:137](../../tests/optimization/test_passes.cpp#L137), [test_passes.cpp:166](../../tests/optimization/test_passes.cpp#L166), [test_passes.cpp:195](../../tests/optimization/test_passes.cpp#L195) |
| OPT-TC-001, OPT-TC-002, OPT-TC-003 | TransposeCancellation | V/N/V | inverse perms cancel / non-inverse perms / chained inverse | changed / unchanged / changed | done | [test_passes.cpp:223](../../tests/optimization/test_passes.cpp#L223), [test_passes.cpp:250](../../tests/optimization/test_passes.cpp#L250), [test_passes.cpp:276](../../tests/optimization/test_passes.cpp#L276) |
| OPT-EF-001, OPT-EF-002 | ElementwiseFusion | V/N | chain fuses / non-fusable interruption | changed / unchanged | done | [test_passes.cpp:302](../../tests/optimization/test_passes.cpp#L302), [test_passes.cpp:324](../../tests/optimization/test_passes.cpp#L324) |
| OPT-LBA-001 | LinearBiasActivationFusion | V | linear+relu fuses | changed | done | [test_passes.cpp:344](../../tests/optimization/test_passes.cpp#L344) |
| OPT-LBA-002 | LinearBiasActivationFusion | N | linear+softmax not fused | unchanged | done | [test_passes.cpp:367](../../tests/optimization/test_passes.cpp#L367) |
| OPT-FIX-001 | PassManager (fixed point) | V | DeadOpElimination drops a linear's dead consumer; a later sweep then fuses the surviving relu | changed, 2+ sweeps | done | [test_passes.cpp:402](../../tests/optimization/test_passes.cpp#L402) |
| OPT-FIX-002 | PassManager (fixed point) | N | the same pair, run through one sweep only (no re-run) | not tagged | done | [test_passes.cpp:439](../../tests/optimization/test_passes.cpp#L439) |
| OPT-EQ-001 | all | V | numerical equivalence before/after | equal (e2e) | partial — LOW-E2E-MLP/NATIVE-001 verify LinearBiasActivationFusion+DeadOpElimination numerically end to end; not yet repeated per-pass for the other 6 | [test_mlp_e2e.cpp](../../tests/codegen/test_mlp_e2e.cpp), [mlp_native_driver.c](../../tests/codegen/mlp_native_driver.c) |

All 6 passes and their `GraphPass`/`PassManager` infrastructure live in
`include/tensora/optimization/GraphPass.h` + `src/optimization/GraphPasses.cpp`,
exercised by `tests/optimization/test_passes.cpp` (20 CTest cases, 54 checks,
all passing).

## Analysis — tensor liveness (ARCH-009)

| ID | Ref | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| LIVE-001 | ARCH-009 | V | live intervals on a producer/consumer chain | intervals correct | done | [test_passes.cpp:466](../../tests/optimization/test_passes.cpp#L466) |
| LIVE-002 | ARCH-009 | V | byte size of a concrete-dim tensor | matches hand calc | done | [test_passes.cpp:489](../../tests/optimization/test_passes.cpp#L489) |
| LIVE-003 | ARCH-009 | V | peak-memory = max simultaneously-live | matches hand calc | done | [test_passes.cpp:496](../../tests/optimization/test_passes.cpp#L496) |
| LIVE-004 | ARCH-009 | V | a decision consumes peakLiveBytes | decision changes with estimate | done — feeds `TilingDecision` (LOOP-001-F), a real consumer, not just a reported number | [test_passes.cpp:515](../../tests/optimization/test_passes.cpp#L515) |

`include/tensora/optimization/Liveness.h` + `src/optimization/Liveness.cpp`;
exercised by `tests/optimization/test_passes.cpp`'s LIVE-* cases (4 CTest
cases).

## Loop optimization — matmul tiling (Level-2, ARCH-009, USP)

| ID | Ref | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| LOOP-001-A, LOOP-001-B, LOOP-001-C, LOOP-001-D, LOOP-001-E, LOOP-001-F | 8.4 | V/N | large/tiny/exact-divide/remainder/symbolic-skip/memory-pressure `TilingDecision` cases | plan matches shapes (pure C++, no MLIR) | done | [test_tiling_decision.cpp:29](../../tests/codegen/test_tiling_decision.cpp#L29), [test_tiling_decision.cpp:42](../../tests/codegen/test_tiling_decision.cpp#L42), [test_tiling_decision.cpp:53](../../tests/codegen/test_tiling_decision.cpp#L53), [test_tiling_decision.cpp:64](../../tests/codegen/test_tiling_decision.cpp#L64), [test_tiling_decision.cpp:76](../../tests/codegen/test_tiling_decision.cpp#L76), [test_tiling_decision.cpp:95](../../tests/codegen/test_tiling_decision.cpp#L95) |
| LOOP-002 | 8.4 | V | tiled loop nest structure after `MatmulTilingPass` | tiled copy has `scf.for`, untiled copy does not | done | [test_codegen_e2e.cpp:27](../../tests/codegen/test_codegen_e2e.cpp#L27) |
| LOOP-003 | 8.4 | V | tiled vs untiled numerical equivalence | both compute the hand-verified reference value, JIT-executed | done | [test_tiling_e2e.cpp:230](../../tests/codegen/test_tiling_e2e.cpp#L230) |
| LOOP-004 | 8.4 | V | tiled vs untiled wall-clock, 1024x1024x1024, tile=128 | reported (real speedup, single-run methodology, not a rigorous claim on its own) | done | [test_tiling_e2e.cpp:246](../../tests/codegen/test_tiling_e2e.cpp#L246) |
| LOOP-005-A/B | ARCH-009 | V/V | `MlirLowerer`'s own `LivenessAnalysis` call (not a hand-built `LivenessResult`) drives the tile-size choice through the real lowering path | tile_sizes={32,32,32} with no memory pressure; {16,16,16} once an unused 1.37 MiB weight alone pushes peak live bytes over the 1 MiB budget | done | [test_liveness_tiling_e2e.cpp](../../tests/codegen/test_liveness_tiling_e2e.cpp) |
| LOOP-005-C | ARCH-009 | V | a plain 512x512x512 matmul (no extra weight fixture) already exceeds the default 1 MiB budget from its own operand sizes (3 MiB total) | tile_sizes={16,16,16} | done | [test_liveness_tiling_e2e.cpp:138](../../tests/codegen/test_liveness_tiling_e2e.cpp#L138) |
| LOOP-005-D | ARCH-009 | V | the identical 512x512x512 matmul, lowered under a 4 MiB `TilingPolicy::memoryBudgetBytes` instead of the 1 MiB default | tile_sizes={32,32,32} -- confirms the "budget" is a policy field compared against a static shape-based estimate, not a real memory measurement, so modeling a machine with more/less available memory for this decision is exactly changing this one field, not an OS-level memory limit on the process | done | [test_liveness_tiling_e2e.cpp:158](../../tests/codegen/test_liveness_tiling_e2e.cpp#L158) |

**Measured tile-size evidence** (`tools/benchmark_matmul_tiling`, not a
CTest case — a reporting tool, best-of-5-runs per config, this
machine, 1024x1024x1024 f32 matmul):

| config | seconds (best of 5) | vs. untiled |
|---|---|---|
| untiled | ~2.96–3.44 | 1.00x |
| 16x16x16 | ~1.30–1.34 | ~2.3–2.6x |
| 32x32x32 | ~1.32–1.37 | ~2.2–2.5x |
| 64x64x64 | ~1.32–1.35 | ~2.2–2.6x |

Run `make benchmark` (or `build/benchmark_matmul_tiling` directly) to
reproduce on your own machine —
these are real measurements, not assumed numbers, and this project does
not claim they generalize beyond this workload/machine. All three tile
sizes beat untiled by roughly the same margin here; the differences
*between* 16, 32, and 64 are within run-to-run noise on this machine, so
this workload does not, by itself, identify a single "optimal" tile
size among the three -- it does show tiling itself is worth doing, and
that the existing default (32) is a reasonable, evidence-checked choice
among options that are empirically close, not a value picked by feel.

`TilingDecision` (`include/tensora/codegen/LoopTiling.h`, pure C++,
tested in `tests/codegen/test_tiling_decision.cpp`) is the
Tensora-authored decision; `MatmulTilingPass`
(`include/tensora/codegen/MatmulTilingPass.h`, a real
`PassWrapper<..., OperationPass<func::FuncOp>>` using MLIR's own
`TilingInterface`/`scf::tileUsingSCF`) is the project-authored MLIR pass
that applies it — this is the Level-2 / USP deliverable. Both are
distinguished from standard MLIR/LLVM infrastructure (bufferization,
loop/control-flow/arith/memref-to-LLVM conversion) in
`tests/codegen/test_tiling_e2e.cpp`'s pipeline, which runs the
tiling pass first, then the standard lowering passes, then JIT-executes
and compares results.

## Lowering (RM 3.22)

| ID | RM | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|----|
| LOW-001 | [3.22](../language/03_LANGUAGE_REFERENCE_MANUAL.md#322-execution-model) | V | matmul (rank 2) -> `linalg.matmul` (zero-filled accumulator); matmul (rank >= 2 with leading batch axes, incl. broadcasting) -> `linalg.generic` contraction; linear (rank(x) 1 or >= 3) -> `linalg.generic` batched vector-matrix contraction, no separate M axis | numerically verified via JIT (LOW-E2E-MLP, LOOP-002/003, NATIVE-001, RANK-E2E) | done | [test_rank_e2e.cpp](../../tests/codegen/test_rank_e2e.cpp), [test_mlp_e2e.cpp](../../tests/codegen/test_mlp_e2e.cpp) |
| LOW-002 | [3.22](../language/03_LANGUAGE_REFERENCE_MANUAL.md#322-execution-model) | V | conv -> `linalg.conv_2d_nchw_fchw`, zero-filled accumulator, `tensor.pad` when padding>0, channel-broadcast bias | numerically verified via JIT, padding=0 (LOW-009, E2E-002) and padding=1 (LOW-009b, exercises the `tensor.pad` path specifically) | done | [test_new_ops_e2e.cpp](../../tests/codegen/test_new_ops_e2e.cpp) |
| LOW-003 | [3.22](../language/03_LANGUAGE_REFERENCE_MANUAL.md#322-execution-model) | V | relu/bias -> `linalg.generic` (incl. linear+relu fusion into one generic body) | numerically verified via JIT (LOW-E2E-MLP, NATIVE-001) | done | [test_mlp_e2e.cpp](../../tests/codegen/test_mlp_e2e.cpp) |
| LOW-004 | [3.22](../language/03_LANGUAGE_REFERENCE_MANUAL.md#322-execution-model) | V | softmax expansion (max-shift, `math.exp`, sum, normalize — 4 chained `linalg.generic` ops) | numerically verified via JIT against a non-trivial (non-saturated) input (LOW-008, E2E-002, E2E-004) | done | [test_new_ops_e2e.cpp](../../tests/codegen/test_new_ops_e2e.cpp) |
| LOW-005 | [3.22](../language/03_LANGUAGE_REFERENCE_MANUAL.md#322-execution-model) | V | concrete dim -> static; symbolic -> dynamic `?` | `lowerTensorType` implements this per ARCH-005 | broken (confirmed, not just untested) — `lowerTensorType` correctly lowers a symbolic dim to a dynamic `?` *type*, but every lowering function that materializes a fresh tensor (`tensor.empty()` in elementwise, relu, matmul/conv2d's zero-filled accumulator, etc.) calls it with zero dynamic-size operands, so a module with any symbolic dimension fails MLIR verification (`incorrect number of dynamic sizes`). Found by running the real `tensora` CLI driver against `examples/mlp.tns` (a symbolic-batch source file that already passes semantic analysis); the driver now rejects symbolic shapes before lowering with a clear message (INT-FULL-SYMBOLIC-REJECT) rather than hitting this. Fixing it properly means threading a same-shape existing SSA value (to query dynamic dims from via `tensor.dim`) through every lowering function -- a real, separate piece of work, not a one-line patch. | [cli_tests.txt](../../tests/cli/cli_tests.txt) |
| LOW-006 | [3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose) | V | transpose -> `linalg.transpose` | numerically verified via JIT (LOW-006, E2E-004) | done | [test_new_ops_e2e.cpp:1](../../tests/codegen/test_new_ops_e2e.cpp#L1) |
| LOW-007 | [3.13](../language/03_LANGUAGE_REFERENCE_MANUAL.md#313-flatten) | V | flatten -> `tensor.collapse_shape` (reassociation `[[0],[1..rank-1]]`) | numerically verified via JIT (LOW-007, E2E-002) | done | [test_new_ops_e2e.cpp:204](../../tests/codegen/test_new_ops_e2e.cpp#L204) |
| LOW-010 | [3.15](../language/03_LANGUAGE_REFERENCE_MANUAL.md#315-sum--mean) | V | sum -> rank-reducing `linalg.generic` reduction (drops the named axis entirely, unlike softmax's keep-dims) | numerically verified via JIT (LOW-010) | done | [test_new_ops_e2e.cpp:451](../../tests/codegen/test_new_ops_e2e.cpp#L451) |
| LOW-011 | [3.15](../language/03_LANGUAGE_REFERENCE_MANUAL.md#315-sum--mean) | V | mean -> sum + divide-by-axis-size elementwise pass | numerically verified via JIT (LOW-011) | done | [test_new_ops_e2e.cpp:499](../../tests/codegen/test_new_ops_e2e.cpp#L499) |
| LOW-012 | [3.17](../language/03_LANGUAGE_REFERENCE_MANUAL.md#317-maxpool) | V | maxpool -> `linalg.pooling_nchw_max` (-inf-filled accumulator, shape-only window operand) | numerically verified via JIT (LOW-012) | done | [test_new_ops_e2e.cpp:547](../../tests/codegen/test_new_ops_e2e.cpp#L547) |
| LOW-013 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | V | elementwise add with real tensor-tensor broadcasting (a shorter-rank operand, not a tensor-scalar op) -> `linalg.generic` with `broadcastOperandMap` | numerically verified via JIT (LOW-013): `[[1,2,3],[4,5,6]] + [10,20,30]` | done | [test_new_ops_e2e.cpp:101](../../tests/codegen/test_new_ops_e2e.cpp#L101) |

The transpose/flatten/softmax/conv2d(+padding)/sum/mean/maxpool lowering
checks above (LOW-006..013) run as one CTest, **LOW-NEWOPS**
(`tests/codegen/test_new_ops_e2e.cpp`, a case of the consolidated
`test_codegen_e2e` binary).

General reshape (`tensor.reshape` with a runtime shape operand —
flatten's fixed collapse-to-2D reassociation was sufficient for every
workload attempted so far) is still not in `MlirLowerer`'s switch — out
of scope for now, not silently miscompiled (the switch's default case
is `llvm_unreachable`, not a fallthrough). Batched matmul (RANK-003/004)
and rank-1/rank>=3 linear (RANK-001/002) are now lowered -- this closed
a real language/backend gap: sema accepted these ranks (RM 3.7/3.9) but
MlirLowerer previously only handled rank 2, which would have produced
an invalid `linalg.matmul` rather than a clean rejection. Multi-head
attention (which needs batched matmul) is not otherwise attempted;
LOW-006..009/E2E-004 still only demonstrate single-head attention.
Every RM 3.x frozen operator except general reshape is now lowered.

## End-to-End (RM 3.22, whitepaper §10)

| ID | Kind | Case | Result | Status | Test |
|----|----|----|----|----|----|
| E2E-001 | V | MLP compile+run vs hand-computed reference | within tol (exact match) | done — `LOW-E2E-MLP` (JIT) and `NATIVE-001` (ahead-of-time object file + system linker + standalone executable) both compute linear->relu->linear = 3.0 | [test_mlp_e2e.cpp](../../tests/codegen/test_mlp_e2e.cpp), [mlp_native_driver.c](../../tests/codegen/mlp_native_driver.c) |
| E2E-002 | V | CNN classifier vs reference | within tol (1e-4) | done — `test_cnn_e2e.cpp`: conv2d(stride=2)->relu->flatten->linear->softmax, checked against a Python-double-computed reference (0.539902); actual JIT result 0.539915, well within f32 tolerance | [test_codegen_e2e.cpp:39](../../tests/codegen/test_codegen_e2e.cpp#L39) |
| E2E-003 | V | Transformer FFN vs reference | within tol (exact match) | done — `test_ffn_e2e.cpp`: position-wise linear->relu->linear at FFN shapes (seq_len=2, d_model=4, hidden=8), reusing the same lowering LOW-E2E-MLP already proved (no new ops needed); residual connection and layer-norm are not modeled (neither is a lowered op) | [test_codegen_e2e.cpp:45](../../tests/codegen/test_codegen_e2e.cpp#L45) |
| E2E-004 | V | (single-head) scaled dot-product attention vs reference | within tol (1e-4) | done for single head — `test_attention_e2e.cpp`: matmul->transpose->scale->softmax->matmul, checked against a Python-double-computed reference (1.660477); multi-head specifically is not started (needs batched matmul) | [test_codegen_e2e.cpp:42](../../tests/codegen/test_codegen_e2e.cpp#L42) |
| E2E-005 | V | matmul-heavy model, tiled vs untiled vs reference | within tol + timing | done — `LOOP-002/003/004` (single 8x8x8 matmul; timing reported, not asserted, since the workload is too small for tiling to pay off) | [test_tiling_e2e.cpp](../../tests/codegen/test_tiling_e2e.cpp) |
| E2E-006 | V | differential random small graphs | — | not started | — |
| E2E-007 | V | real-world inference demos vs PyTorch (MLP placement / STL-10 10-class CNN / fixed-grid CNN object detector, synthetic then real photos / self-attention text classification (single-head) / Associative Recall, single-query form (multi-head)) | within tol (1e-3/1e-4) | done — `make demo-mlp-placement`/`demo-cnn-image-classifier`/`demo-cnn-grid-detector`/`demo-cnn-grid-detector-real`/`demo-attn-text-classifier`/`demo-attn-retrieval`: each runs PyTorch-trained weights on real held-out inputs through the real CLI (`--run --data-dir`) and checks against a saved PyTorch reference. These exercise **multi-channel** conv2d/maxpool (STL-10, natively 96x96 -- no upscaling, see ARCH-017: 3->16->32->64->64 channels, 4 conv/pool blocks), a **spatial, per-cell** classification head (`demo-cnn-grid-detector`, `examples/cnn_grid_detector.tns` -- a fixed 4x4 detection grid over a 32x32 synthetic image with 1-3 non-overlapping shapes, built entirely from a 1x1 `conv2d` head plus `softmax(axis=channel)`, no reshape/flatten needed, and no second graph output for box regression since v1's `return` names exactly one tensor; PASS 480/480 per-cell predictions, max prob error 1.2e-7), the same architecture family on **real, human-annotated photographs** at finer resolution (`demo-cnn-grid-detector-real`, `examples/cnn_grid_detector_real.tns` -- Pascal VOC 2012 photos, 16x16 grid, 5 object classes; PASS 6144/6144 per-cell predictions, max prob error 4.5e-7; the small from-scratch model's own precision is honestly weak — recall ~49%, many false-positive cells — a training-data limitation, not a compiler one), batched rank-3 attention (single-head), and **rank-4, 4-head** attention (`demo-attn-retrieval`, `examples/attn_multihead_retrieval.tns` -- a deliberately different task from the single-head demo, Associative Recall (AR) -- a standard synthetic self-attention benchmark (Ba et al. 2016; Arora et al.'s "Zoology", arXiv:2312.04927, which generalizes it to the multi-query MQAR variant this demo does not implement; see examples/attn_multihead_retrieval.tns's header) -- with a reshuffled binding every example, so it cannot be memorized; exercises batched matmul/transpose/sum at rank 4, the exact shape family whitepaper Section 10.4/`ATT-001` type-check but do not run, plus a one-hot-selector read-out pattern in place of indexing) end-to-end,
and a genuinely different, harder task on the same multi-head attention shape family
(`demo-attn-relational`, `examples/attn_relational_eligibility.tns` — course eligibility, a two-hop
relational-reasoning task: given which courses a student has completed and which single
prerequisite each candidate course needs, find the one course now eligible; unlike AR the answer is
never a fact stated directly in the input, so it needs two chained attention/lookup steps — attend
over completion facts for a "profile" (hop 1), then check each candidate's prerequisite and
completion status against that profile (hop 2, plain matmul/elementwise, no second attention op) —
built entirely from matmul/transpose/softmax/sum/elementwise, same operator vocabulary as every other
v1 demo; PASS 200/200 held-out predictions, max logit error 2.9e-6; the trained model's own accuracy
against ground truth is honestly ~75% (well above the 1/6 chance rate, confirming genuine two-hop
fact-chaining, but clearly below AR's ~100%, since this is a strictly harder task — see
tools/models/prepare_relational_demo.py's module docstring) — coverage the C++ e2e tests above (single-channel conv, rank-2 attention) do not. They also exercise graphs with >7 inputs/weights (the CNN has 11), i.e. the packed JIT invocation (ARCH-016). Not CTest cases (need PyTorch to prepare artifacts); the loader path they rely on is CTest-covered by `INT-DATALOAD-*` | — |
| E2E-008 | V | one self-attention block from a real, third-party, open-source pretrained Transformer (google/vit-base-patch16-224-in21k) vs that model's own output | within tol (1e-4) | done — `make demo-vit-attention-one-layer` (`tools/models/prepare_vit_attention_demo.py`, `demos/vit_attention_demo.sh`, `examples/vit_attention_block.tns`): downloads the real ViT-Base-in21k checkpoint, runs a real photograph through it, hooks the last encoder layer's self-attention module to capture its real input hidden state and real output, extracts its real Q/K/V/output-projection weights and biases (12 heads, 768-dim, 197 positions -- full production scale, not a toy size), and recomputes the block through the real `tensora --run --data-dir` CLI; checked against the real model's own output for that block (max abs error 2.3e-5). `make demo-vit-attention-all-layers` (`demos/vit_attention_all_layers_demo.sh`) repeats that check independently at **all 12 encoder layers** (each layer's own real input hidden state and real pretrained weights, compared against that layer's own real output), printing a per-layer max-error table and `PASS 12/12` (errors ranging ~1.1e-5 to ~2.3e-5) -- Tensora independently executes and verifies self-attention for every encoder layer of the real pretrained model, not only the last one. `make demo-vit-attention-viz` (`tools/models/visualize_vit_attention.py`, `examples/vit_attention_weights.tns`) additionally checks Tensora's own attention weights (the block stopped right after `softmax`, all 12 heads) against the real model's own attention weights (`attn_implementation="eager"`, max abs error ~1.9e-7), and renders (a) a per-head attention-heatmap PNG over the real photograph, (b) a mean-over-heads/attention-rollout PNG (Abnar & Zuidema 2020, computed from the real model's attention at every one of its 12 layers), (c) an attention-microscope PNG inspecting three patches at once (face/body/background, picked ahead of time from the image's patch-grid colors) and showing, per patch, the mean-over-heads "attends to" map alongside three individual heads' "attends to" maps (lowest/middle/highest [CLS]-attention entropy) plus the mean "attended back" column -- the per-head columns matter because the mean-over-heads maps look similar across all three patches (a few globally-salient patches dominate every average) while at least one individual head attends very differently depending on the query patch, a structure the mean alone hides, (d) an attention-asymmetry PNG showing the top-3 most directionally-asymmetric patch pairs as distinctly colored, value-labeled arrows (`Attention(A→B) != Attention(B→A)`), and (e) an attention-diagnostics PNG with per-head entropy (labeled `head 0`-`head 11` with each bar's numeric value) plus a real-model-vs-Tensora attention error map at a dramatically enlarged scale, and (f) an attention-topk-boxes PNG showing the same "attends to" row with no heatmap/interpolation at all -- a box around exactly the top-5 attended patches, each labeled with its printed weight, since a colormap overlay with bilinear interpolation between 14x14 cells adds visual information the model didn't produce -- visual, not just numerical, evidence that different heads attend to different image regions, that attention is directional, and that Tensora's output is numerically indistinguishable from the real model's. Every panel is bordered to show provenance: solid green "TENSORA-VERIFIED" for Tensora's own compiled/checked output, dashed gray "reference only" for real-model data Tensora did not compute (used only for the rollout panel and the real-model reference in the error map). Unlike E2E-007's demos, which train a small model from scratch for realistic-but-synthetic weights, this is validated against genuine third-party pretrained parameters. Full end-to-end compilation of the whole model (patch-embedding conv, layernorm, GELU, classification head) is out of scope for v1 -- to be considered in future. Not a CTest case (needs `transformers`/`pillow`/internet to prepare artifacts) | — |
| E2E-009 | V | splicing Tensora's own recomputation of one real attention block into a LIVE real ImageNet-1k classifier (google/vit-base-patch16-224), mid-inference, and checking the real predicted label + confidence survive, on 5 visibly different photographs | label unchanged on 5/5; final logits within tol (~1e-6) | done — `make demo-vit-attention-substitution` (`tools/models/prepare_vit_substitution_demo.py`, `demos/vit_attention_substitution_demo.sh`, `tools/models/visualize_vit_substitution.py`, reuses `examples/vit_attention_block.tns`): downloads a real, pretrained, 1000-way ImageNet-1k *classifier* checkpoint (distinct from E2E-008's headless `-in21k` encoder, which has no classification head), then for each of 5 real photographs (dog/car/elephant/airplane/cat) runs it through the model unmodified to get a baseline top-1 label + confidence, hooks the last encoder layer's real input hidden state and real Q/K/V/output-projection weights, recomputes that block through the real `tensora --run --data-dir` CLI (block output max abs error ~1.7e-5 to ~4.5e-5, consistent with E2E-008), then monkey-patches that one block's `forward` in the LIVE model to return Tensora's recomputed tensor and re-runs the SAME photograph through the SAME model end to end (patch-embedding conv, the other 11 attention blocks, all MLP blocks/layernorms, and the classifier head all run unmodified). Result: identical predicted label on all 5 photographs, final-logit max abs error ~1e-6 after 11 more encoder layers plus the classifier head downstream of the substitution point. Where E2E-007/E2E-008 prove a purely numerical claim ("Tensora's output tensor matches PyTorch's, in isolation"), this is the first demo to prove a *functional* one: Tensora's compiled output is a drop-in replacement inside a real production inference pipeline, not merely numerically close in a standalone comparison. Also renders a visual dashboard (input photo, top-5 probability bars baseline-vs-substituted, layer-substitution schematic, numeric-error meter) per photograph plus a summary gallery, if `matplotlib` is installed. Not a CTest case (needs `transformers`/`pillow`/internet to prepare artifacts, and a built `tensora` binary at prep time -- unlike every other demo here, it shells out to the compiled binary directly rather than only comparing pre-saved `.npy` files) | — |

E2E-001/002/004/005 use a hand-computed (or, for E2E-002/004, a
Python-double-precision-computed) reference value rather than a literal
NumPy invocation from the C++ test suite (no Python/NumPy runtime
dependency introduced) — the numerical rigor (an exact expected value,
checked via real JIT and, for E2E-001, native execution) is the same
either way.

## Smoke (TEST_PLAN §2b) — week-1 gate marked

| ID | Checks | Week 1 | Status | Test |
|----|----|----|----|----|
| SMOKE-MLIR | trivial `func.func`+`linalg.matmul` builds, lowers to LLVM, JIT-runs, checks 2x2 matmul result | yes (gate) | done | [smoke_mlir.cpp:1](../../tests/smoke/smoke_mlir.cpp#L1) |
| SMOKE-TILING | MLIR tiling/transform interface callable on a matmul loop nest | yes (gate) | done — folded into `LOOP-002` rather than a separate smoke case, since `MatmulTilingPass` (the real deliverable) already exercises exactly this | [test_tiling_e2e.cpp](../../tests/codegen/test_tiling_e2e.cpp) |
| SMOKE-BUILD | cmake configure+build; ANTLR generates `TensoraLexer`/`TensoraParser` from `grammar/Tensora.g4` & compiles | yes | done — verified via a full `cmake -S . -B build && cmake --build build` from a clean tree; INTEG-003 additionally pins the MLIR/LLVM 21 apt packages required | — |
| SMOKE-CLI | `tensora --check hello.tns` exits 0 |  | planned | — |
| SMOKE-E2E | `tensora matmul.tns -o m && ./m` runs |  | planned | — |

## Integration (TEST_PLAN §9a) — real phases, no mocks

| ID | Boundary (INTERFACES) | Case | Result | Status | Test |
|----|----|----|----|----|----|
| INT-LEXPARSE | Lexer→Parser | source → AST (golden) | ast | planned | — |
| INT-PARSESEMA | Parser→Sema | source → typed SSA graph (golden) | graph | planned | — |
| INT-SEMAOPT | Sema→Passes | typed graph → optimized (before/after) | changed | planned | — |
| INT-ANALYSISDEC | Analysis→Decision | liveness estimate drives a choice | decision varies | done — `LIVE-004`/`LOOP-001-F`: a `LivenessResult` over budget selects `TilingDecision`'s conservative tile size instead of the default | — |
| INT-GRAPHMLIR | Lowering | typed graph → MLIR (golden) | mlir | partial — verified via numerical JIT/native-execution correctness (LOW-E2E-MLP, LOOP-002/003, NATIVE-001), not a literal golden-MLIR-text diff | — |
| INT-MLIRLLVM | Backend | MLIR → LLVM → object | links | done — `NATIVE-001`: `codegen_mlp_native` emits a real x86-64 ELF object via `llvm::TargetMachine` (verified with `file`), linked by the system linker into a standalone executable (verified with `ldd`: only libc, no MLIR/LLVM runtime) | [codegen_mlp_native.cpp](../../tools/codegen_mlp_native.cpp) |
| INT-FULL-MLP | full pipeline | MLP source → executable | runs | done — the real `tensora` CLI (`src/driver/main.cpp`) now runs `examples/mlp_concrete.tns` through Lexer→Parser→SemanticAnalyzer→Level-1 passes→`MatmulTilingPass`→`MlirLowerer`→standard MLIR/LLVM lowering, both JIT (`--run`, `INT-FULL-MLP-RUN`, hand-verified result 145 for all-ones synthetic input) and ahead-of-time object-file emission (`-o`, verified via `file`) | [cli_tests.txt:16](../../tests/cli/cli_tests.txt#L16), [cli_tests.txt:17](../../tests/cli/cli_tests.txt#L17) |
| INT-FULL-MM | full pipeline | matmul-heavy source → executable | runs | done — `examples/matmul_large.tns` (256x256x256, above `MatmulTilingPass`'s default tiling threshold) compiles and JIT-runs through the same CLI (`INT-FULL-MM-RUN`); `--emit-mlir` shows 3 nested `scf.for` loops confirming the tiling pass actually fired, and the result (256, the contracted dimension, for all-ones inputs) is hand-verified | [cli_tests.txt:18](../../tests/cli/cli_tests.txt#L18) |
| INT-FULL-CNN | full pipeline | CNN source → executable | runs | done — `examples/cnn.tns` compiles and JIT-runs through the same CLI (`INT-FULL-CNN-RUN`), hand-verified result 0.5/0.5 (symmetric logits from an all-ones conv+linear stack) | [cli_tests.txt:19](../../tests/cli/cli_tests.txt#L19) |
| INT-FULL-SYMBOLS | Sema (RM 3.6) | `--emit-symbols` on a non-trivial program | golden symbol-table text | done — `INT-FULL-SYMBOLS`: `examples/mlp.tns` through the real CLI, checked against a `PASS_REGULAR_EXPRESSION` golden fragment; full determinism/scope/shadowing/error-path coverage is in `SYM-001`..`008` above | [cli_tests.txt:15](../../tests/cli/cli_tests.txt#L15) |
| INT-DATALOAD-EQUIV | Driver `--data-dir` (ARCH-016) | all-ones `.npy` inputs reproduce the built-in all-ones result | element[0] = 145 | done — `examples/mlp_concrete.tns` with `--data-dir tests/driver/mlp_ones` gives the same result as the default all-ones `--run` (`INT-FULL-MLP-RUN`), proving the `.npy` loader is transparent | [cli_tests.txt:25](../../tests/cli/cli_tests.txt#L25) |
| INT-DATALOAD-MISSING | Driver `--data-dir` | a declared param has no `.npy` file | reject: clear error, nonzero exit | done — `WILL_FAIL` | [cli_tests.txt:26](../../tests/cli/cli_tests.txt#L26) |
| INT-DATALOAD-BADSHAPE | Driver `--data-dir` | a param's `.npy` shape != declared shape | reject: clear error, nonzero exit | done — `WILL_FAIL` | [cli_tests.txt:27](../../tests/cli/cli_tests.txt#L27) |
| INT-DATALOAD-BADDTYPE | Driver `--data-dir` | a param's `.npy` dtype is not float32 | reject: clear error, nonzero exit | done — `WILL_FAIL` | [cli_tests.txt:28](../../tests/cli/cli_tests.txt#L28) |
| INT-POLY-* | Driver `--polyhedral` (ARCH-018) | linalg lowered through the MLIR affine dialect + `affine-loop-tile`, instead of the default `convert-linalg-to-loops` | result matches the default path exactly | done — `INT-POLY-MLP-RUN`/`MM-RUN`/`CNN-RUN`/`ATTN-RUN`/`CNNSTL10-RUN` each reuse their `INT-FULL-*-RUN` twin's expected value, so a match proves the polyhedral path is numerically identical, not just that it runs. The generated LLVM IR differs (real transform, not a no-op); see ARCH-018 | [cli_tests.txt:41](../../tests/cli/cli_tests.txt#L41), [cli_tests.txt:42](../../tests/cli/cli_tests.txt#L42), [cli_tests.txt:43](../../tests/cli/cli_tests.txt#L43), [cli_tests.txt:44](../../tests/cli/cli_tests.txt#L44), [cli_tests.txt:48](../../tests/cli/cli_tests.txt#L48) |

The CLI driver (`tensora --emit-tokens/--emit-ast/--check/--emit-symbols/--emit-graph/--explain/--emit-mlir/--emit-llvm/--run/-o`)
did not exist before this pass; building and testing it against real `.tns` source is what surfaced the LOW-005
symbolic-dimension bug above -- exactly the value of testing the full pipeline rather than only its hand-built-fixture
unit tests. `--run`/`-o`/`--emit-mlir`/`--emit-llvm` all require concrete (non-symbolic) shapes for this reason
(enforced with a clear error, `INT-FULL-SYMBOLIC-REJECT`); `--emit-tokens`/`--emit-ast`/`--check`/`--emit-graph`/
`--explain` work on any well-formed program, symbolic or not, since they stop before lowering.
`examples/attention.tns` (single-head) is also verified this way but has no dedicated INT-FULL-ATTN row in
RM/whitepaper terms -- see `INT-FULL-ATTN-RUN`. Batched matmul itself is no longer a gap (RANK-003/004 above); a
rank-4 multi-head attention example is simply not yet written, not blocked on missing lowering support.

## Boundary (TEST_PLAN §9b) — min / max / just-inside / just-outside

| ID | Rule | Case | Result | Status | Test |
|----|----|----|----|----|----|
| BND-MM-001 | [3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication) | matmul `K=1` | accept | done | [semantic.tns:336](../../tests/fixtures/semantic.tns#L336) |
| BND-MM-002 | [3.7](../language/03_LANGUAGE_REFERENCE_MANUAL.md#37-matrix-multiplication) | matmul `M=1` or `N=1` | accept | done | [test_semantic.cpp:252](../../tests/semantic/test_semantic.cpp#L252) |
| BND-BC-001 | [3.8](../language/03_LANGUAGE_REFERENCE_MANUAL.md#38-elementwise-arithmetic-and-broadcasting) | broadcast size-1 each side | accept | done | [test_semantic.cpp:265](../../tests/semantic/test_semantic.cpp#L265) |
| BND-RS-001 | [3.12](../language/03_LANGUAGE_REFERENCE_MANUAL.md#312-reshape) | reshape product off by one | reject: SemaReshapeCount | done | [semantic.tns:345](../../tests/fixtures/semantic.tns#L345) |
| BND-FL-001 | [3.13](../language/03_LANGUAGE_REFERENCE_MANUAL.md#313-flatten) | flatten symbolic trailing dim | reject: SemaUnresolvableShape | done | [semantic.tns:353](../../tests/fixtures/semantic.tns#L353) |
| BND-CV-001 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d) | conv output exact-divide vs remainder | accept / reject | done | [test_semantic.cpp:278](../../tests/semantic/test_semantic.cpp#L278) |
| BND-CV-002 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d) | kernel = input size (output 1) | accept | done | [test_semantic.cpp:297](../../tests/semantic/test_semantic.cpp#L297) |
| BND-RD-001 | [3.15](../language/03_LANGUAGE_REFERENCE_MANUAL.md#315-sum--mean) | reduce axis 0 / last / only axis | accept | done | [test_semantic.cpp:310](../../tests/semantic/test_semantic.cpp#L310) |
| BND-TR-001 | [3.14](../language/03_LANGUAGE_REFERENCE_MANUAL.md#314-transpose) | identity perm; full-reverse perm | accept | done | [test_semantic.cpp:328](../../tests/semantic/test_semantic.cpp#L328) |
| BND-DIM-001 | [3.4](../language/03_LANGUAGE_REFERENCE_MANUAL.md#34-symbolic-dimensions) | just inside vs just outside affine subset | accept / SemaUnresolvableShape | done | [test_semantic.cpp:348](../../tests/semantic/test_semantic.cpp#L348) |
| BND-LIVE-001 | ARCH-009 | value never used / only at return / all live | intervals correct | planned | — |
| BND-TILE-001 | 8.4 | dim divisible / = tile / < tile / = 1 | plan correct | planned | — |
| BND-LEX-001 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | empty file | only eof, at line 1 col 1 | done | [test_lexer.cpp:393](../../tests/lexer/test_lexer.cpp#L393) |
| BND-LEX-002 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | comments-only file (single + multiple lines) | only eof | done | [test_lexer.cpp:400](../../tests/lexer/test_lexer.cpp#L400) |
| BND-LEX-003 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | token at EOF, no trailing newline, per token category | token then eof | done | [test_lexer.cpp:415](../../tests/lexer/test_lexer.cpp#L415) |
| BND-LEX-004 | [3.1](../language/03_LANGUAGE_REFERENCE_MANUAL.md#31-lexical-structure) | first token of any file starts at line 1, col 1, offset 0 | span correct | done | [test_lexer.cpp:438](../../tests/lexer/test_lexer.cpp#L438) |
| BND-PAR-001 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | graph body that is only `return` (zero statements) vs a single statement | ast | done | [test_parser.cpp:456](../../tests/parser/test_parser.cpp#L456) |
| BND-PAR-002 | [3.2](../language/03_LANGUAGE_REFERENCE_MANUAL.md#32-program-structure), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | minimal legal program: zero decls, exactly one graph, return-only body | ast | done | [test_parser.cpp:472](../../tests/parser/test_parser.cpp#L472) |
| BND-PAR-003 | [3.3](../language/03_LANGUAGE_REFERENCE_MANUAL.md#33-tensor-types) | `tensorType` axis-count extremes: 1 axis (min) vs 4 axes (conv-style) | ast | done | [test_parser.cpp:484](../../tests/parser/test_parser.cpp#L484) |
| BND-PAR-004 | [3.16](../language/03_LANGUAGE_REFERENCE_MANUAL.md#316-conv2d), [3.24](../language/03_LANGUAGE_REFERENCE_MANUAL.md#324-grammar-summary) | call arg-count extreme: `conv2d`'s 5-arg call (3 positional + 2 keyword) | ast | done | [test_parser.cpp:503](../../tests/parser/test_parser.cpp#L503) |

## User / Acceptance (TEST_PLAN §9c) — CLI + diagnostics + demo programs

| ID | Case | Result | Status | Test |
|----|----|----|----|----|
| USER-CLI-001 | each stage flag emits documented artifact + exit code | correct | partial — `--emit-symbols`/`--emit-optimized-graph-dot` artifacts asserted automatically (INT-FULL-SYMBOLS, INT-VIZ-OPTDOT); `--check`/`--run`/`-o` exit codes asserted across INT-FULL-\*/INT-POLY-\*; `--emit-tokens`/`--emit-parse-tree`/`--emit-ast`/`--emit-ast-dot`/`--emit-graph`/`--emit-graph-dot`/`--explain`/`--emit-mlir`/`--emit-llvm` are demoed (`demos/01..05,08_*_demo.sh`) but not asserted against a documented format | [cli_tests.txt](../../tests/cli/cli_tests.txt) |
| USER-LEARN-001 | MLP: check → explain → build → run vs reference | matches tutorial | partial — `--check` and `--run` legs asserted exactly (`element[0] = 145`, the tutorial's hand-computed reference); `--explain` and `-o` (build) legs only demoed manually (`demos/06_final_compiler_demo.sh`), not asserted | [INT-FULL-MLP-CHECK/RUN](../../tests/cli/cli_tests.txt) |
| USER-FIXIT-001 | shape bug rejected with located values; fix compiles | clear diagnostic | partial — every SEM-\*/BND-\* pair already exercises reject-then-fix (a broken fixture rejected with the right `DiagCode`, its sibling fixture accepted); every diagnostic carries a `SourceSpan` by construction, but no test asserts a diagnostic's rendered line/col or expected-vs-actual values appear in the message text | [semantic.tns](../../tests/fixtures/semantic.tns) |
| USER-DIAG-001 | every error class: category+location+expected/actual | UX contract met | partial — every `DiagCode` has at least one fixture that triggers it (see the SEM-\*/BND-\* rows above), and `DiagnosticEngine::report` requires a span on every call; no test enumerates all `DiagCode`s and asserts location+expected/actual are present in the rendered message | [semantic.tns](../../tests/fixtures/semantic.tns) |
| USER-DEMO-001 | 2 working programs run correctly | pass | done — INT-FULL-MLP-RUN, INT-FULL-MM-RUN, INT-FULL-CNN-RUN, INT-FULL-ATTN-RUN each assert an exact expected `element[0]` end to end via the real CLI (4, exceeding the 2 required) | [cli_tests.txt](../../tests/cli/cli_tests.txt) |
| USER-DEMO-002 | 2 failing programs rejected with right diagnostics | pass | partial — INT-FULL-SYMBOLIC-REJECT and INT-DATALOAD-MISSING/BADSHAPE/BADDTYPE (4, exceeding the 2 required) assert rejection (non-zero exit) but `WILL_FAIL` does not check *which* diagnostic fired, so "right diagnostics" is unverified at the CLI layer (the underlying `DiagCode` is verified at the C++ unit level instead) | [cli_tests.txt](../../tests/cli/cli_tests.txt) |
| USER-EXPLAIN-001 | `explain` output matches tutorial for reference programs | matches | planned — no automated test runs `--explain` and compares its output against the tutorial's prose/examples | — |

## Coverage roll-up

| Dimension | Target | Notes |
|---|---|---|
| RM rules mapped | 100% | 3.1–3.23 all have >=1 row |
| Operators (15) | pos + neg each | matmul,linear,+,-,*,/,relu,softmax,reshape,flatten,transpose,sum,mean,conv2d,maxpool — **lowered: matmul (rank 2 and rank >= 2 batched/broadcasting), linear (rank 1, rank 2, rank >= 3), relu, +,-,*,/ (elementwise, incl. broadcast), softmax, flatten, transpose, conv2d (incl. padding), sum, mean, maxpool; not yet lowered: general reshape (arbitrary regrouping beyond flatten's fixed collapse, documented scope boundary, `MlirLowerer`'s default case is `llvm_unreachable`, not silent)** |
| Symbol table (RM 3.6) | direct | `SymbolTable`/`SymbolTableReport` (SYM-001..008, done) — declaration order, scope isolation, shadowing, both error paths; `--emit-symbols` golden output (`INT-FULL-SYMBOLS`, done) |
| Analysis | direct | tensor liveness intervals + peak-memory (LIVE-*, done) — feeds a real decision in `TilingDecision`, not just a reported number |
| Loop opt (Level-2 / USP) | decision + equivalence | matmul tiling: `TilingDecision` (LOOP-001-A..F, done) + project-authored `MatmulTilingPass` (LOOP-002/003/004, done — structural, numerical, and reported timing, all via real JIT execution) |
| DiagCodes | all exercised | every code in INTERFACES.md §0 |
| Passes (6) | positive + near-miss each | see TEST_PLAN §7; constant folding was implemented then retired (DECISIONS.md ARCH-014) |
| 100-case corpus | tracked | split in TEST_PLAN §10 |
| Test type: unit | exhaustive | phase sections; every INTERFACES fn + contract + DiagCode |
| Test type: smoke | exhaustive | SMOKE-* ; per dependency + subsystem; week-1 gate |
| Test type: integration | exhaustive | INT-* ; every adjacent interface + every workload |
| Test type: boundary | exhaustive | BND-* ; min/max/just-in/just-out per limited rule |
| Test type: user | exhaustive | USER-* ; every CLI flag + error class + 4 demo programs |
