```text
LANG-001
```

Title:
    Bound `intLiteral` to a signed 64-bit integer. Reject overflow with a
    lexical diagnostic. Do not crash.

Motivation:
    See KNOWN_BUGS.md BUG-001 and test LEX-011. RM 3.1 defines
    `intLiteral ::= [0-9]+`. It gives no width bound. The lexer
    (src/lexer/Lexer.cpp) assumes the value fits in `int64_t`. It
    converts every `SC_INT` lexeme with `std::stoll`. A literal wider
    than int64 (for example, a 40-digit number) makes `std::stoll` throw
    `std::out_of_range`. Nothing catches this exception inside
    `Lexer::tokenize()`. This breaks `Lexer.h`'s documented contract: an
    illegal input must become an `Error` token plus a diagnostic, and
    scanning must continue. Every statically compiled language with
    fixed-width integers must make this same decision (C, Java, Rust,
    Go): an over-wide literal is a compile-time error. It is not silent
    wraparound, not truncation, and not a crash.

Current behavior:
    The grammar places no bound on `intLiteral`. The implementation
    crashes with an uncaught exception instead of reporting an error, for
    a literal that exceeds `INT64_MAX` (9223372036854775807).

Proposed syntax:
    No change to the surface syntax. `intLiteral ::= [0-9]+` stays the
    same.

Proposed semantics:
    An `intLiteral` token's decimal value must fit in a signed 64-bit
    integer: `0 <= value <= 9223372036854775807`. If a literal's value
    exceeds this bound, the compiler rejects it with a new diagnostic
    code, `LexIntegerOverflow`, at the literal's source span. Scanning
    continues after the error, the same way it continues after
    `LexIllegalChar`. So a file with more than one oversized literal
    gets a report for each one. The rejected token still takes a slot in
    the token stream, as an `Error` token. This matches how the compiler
    already handles `LexIllegalChar`.

    This bound fits real Tensora programs. Axis sizes, `stride`,
    `padding`, `size`, and reduction-axis indices never come close to
    2^63 in any tensor program. No valid example program is affected.

Affected grammar:
    RM 3.1 -- add a bound note under `Literals`:
    "An `intLiteral` value must fit in a signed 64-bit integer
    (0 to 9223372036854775807). A literal outside this range causes a
    `LexIntegerOverflow` diagnostic. The compiler does not truncate it
    silently."

Affected semantic rules:
    None. The lexer catches this case before semantic analysis sees any
    value. Every later stage that reads an int literal (axis sizes,
    operator integer arguments) already assumes the value fits in
    `int64_t`. So the semantic phase needs no change.

Affected compiler modules:
    - include/tensora/shared/Diagnostic.h -- add
      `DiagCode::LexIntegerOverflow`.
    - src/lexer/Lexer.cpp -- replace the unchecked `std::stoll(lexeme)`
      call with a checked parse. The checked parse reports
      `LexIntegerOverflow` and continues, instead of throwing.
    - grammar/lexer.l -- no change. The lexical pattern for `SC_INT`
      stays the same. Only the C++ value-conversion step changes.

Affected tests:
    - tests/lexer/test_lexer.cpp LEX-011 -- change this test from "does
      not crash" to a real negative case. It must assert
      `DiagCode::LexIntegerOverflow`, the correct span, an `Error`-kind
      token, and that scanning continues past the error.
    - Add a boundary pair per 08_TEST_PLAN.md 9b (min/max/just-inside/
      just-outside): `9223372036854775807` (accept, exactly `INT64_MAX`)
      and `9223372036854775808` (reject, one past it).
    - CMakeLists.txt -- drop `WILL_FAIL` on `LEX-011` once it becomes a
      normal passing negative test.

Affected documentation:
    - docs/language/03_LANGUAGE_REFERENCE_MANUAL.md RM 3.1 (the bound).
    - docs/testing/TEST_MATRIX.md LEX-011: change `open (BUG-001)` to
      `done`.
    - docs/testing/KNOWN_BUGS.md BUG-001: change `open` to
      `fixed -> REG-001`.
    - docs/testing/REGRESSION_LOG.md -- add the REG-001 row.
    - docs/architecture/INTERFACES.md `DiagCode` list (Section 0) -- add
      `LexIntegerOverflow`.

Migration impact:
    None, for realistic Tensora programs. Only a pathological or
    adversarial literal (tens of digits) is affected. For that literal,
    the compiler moves from "crashes" to "reports a normal, located
    compile error." This is a strict improvement. No accepted program
    changes behavior.

Decision:        ACCEPTED
Decision date:    2026-08-18
Decided by:       krishnanr
