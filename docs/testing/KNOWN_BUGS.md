# Tensora Known Bugs

> Open defects and known limitations. Once fixed, each confirmed bug
> also gets a regression test (see [REGRESSION_LOG.md](REGRESSION_LOG.md)).
> This was empty at project start.

## Format

``` text
BUG-XXX
Title:
Reported:      YYYY-MM-DD by <owner>
Phase:         lexer | parser | sema | ir | passes | lowering | driver
Severity:      blocker | major | minor
RM/ARCH ref:   e.g. RM 3.8 / ARCH-007
Repro:         minimal .tns program or test ID
Expected:
Actual:
Status:        open | in-progress | fixed (-> REG-XXX)
Notes:
```

## Known Limitations (by design, not bugs)

These are intentional v1 scope boundaries (whitepaper Section 12). The
list is here so testers do not file them as defects:

- Shape reasoning outside the affine-in-one-variable subset is rejected
  with `SemaUnresolvableShape` (RM 3.4, ARCH-006).
- No dtype system; all tensors are f32 (RM 3.23).
- No training, autograd, optimizer updates, dynamic control flow (RM 3.23).
- No grouped-query / multi-query attention contracts (whitepaper §10.4).
- No GPU backend; CPU only (RM 3.22).

## Open Defects

``` text
BUG-001
Title:         Integer literal wider than int64 throws uncaught in Lexer::tokenize()
Reported:      2026-08-18 by krishnanr
Phase:         lexer
Severity:      minor
RM/ARCH ref:   RM 3.1 (intLiteral ::= [0-9]+, no width bound stated)
Repro:         LEX-011 (tests/lexer/test_lexer.cpp); a 40-digit all-nines
               literal, e.g. Lexer::tokenize() on "9999999999999999999999999999999999999999"
Expected:      Lexer.h documents tokenize() as always producing a token
               stream (an illegal character becomes an Error token plus
               a diagnostic, and scanning continues). So no input
               should crash it.
Actual:        src/lexer/Lexer.cpp casts SC_INT lexemes with
               std::stoll(lexeme). This throws std::out_of_range for
               any literal over INT64_MAX. tokenize() does not catch
               this exception, so it escapes the function.
Status:        fixed -> REG-001
Notes:         LANG-001 fixed this (docs/language/
               05_LANGUAGE_EVOLUTION.md §5.9): intLiteral is now
               bounded to a signed 64-bit integer. An overflow is now a
               LexIntegerOverflow diagnostic (Diagnostic.h), not an
               uncaught exception; src/lexer/Lexer.cpp now uses
               std::from_chars instead of std::stoll.
```

``` text
BUG-002
Title:         AstBuilder::buildOperand segfaults on a bare int literal binop operand
Reported:      2026-08-19 by krishnanr
Phase:         parser
Severity:      major
RM/ARCH ref:   RM 3.24 (operand ::= identifier | floatLit; an int literal is not a valid operand)
Repro:         PAR-031 (tests/parser/test_parser.cpp); "graph g { a = x * 2 \n return a }"
Expected:      A syntax error diagnostic (ParseExpected), the same as
               every other malformed-input case.
Actual:        For this malformed input, ANTLR's OperandContext matches
               neither IDENTIFIER nor FLOAT_LITERAL. The rule's own
               try/catch absorbs the exception, so the enclosing
               BinaryExpression context never sees it — the same
               "nested rule catches its own error independently" hazard
               already documented in AstBuilder.cpp for buildArgValue.
               buildOperand assumed a non-matching IDENTIFIER meant
               FLOAT_LITERAL had matched. So it dereferenced a null
               antlr4::tree::TerminalNode* and crashed.
Status:        fixed -> REG-002
Notes:         buildOperand now checks FLOAT_LITERAL() for null before
               it dereferences the pointer, and falls back to a
               zero-valued ScalarExpr. This is harmless: parser
               recovery drops the enclosing statement either way.
```

``` text
BUG-003
Title:         AstBuilder::buildDimSize segfaults on an illegal axis dimSize token
Reported:      2026-08-19 by krishnanr
Phase:         parser
Severity:      major
RM/ARCH ref:   RM 3.24 (dimSize ::= intLit | identifier)
Repro:         PAR-032 (tests/parser/test_parser.cpp); "input x: Tensor[y=@]\ngraph g { return x }"
Expected:      A syntax error diagnostic (ParseExpected).
Actual:        Same root cause as BUG-002: a broken DimSizeContext
               matches neither INT_LITERAL nor IDENTIFIER, and
               buildDimSize dereferences a null IDENTIFIER() node.
Status:        fixed -> REG-003
Notes:         buildDimSize now checks IDENTIFIER() for null before it
               dereferences the pointer, and falls back to a
               zero-valued constant DimSyntax.
```
