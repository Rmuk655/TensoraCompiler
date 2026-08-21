#pragma once
#include "antlr4-runtime.h"
#include "tensora/shared/Diagnostic.h"
#include "tensora/lexer/Token.h"
#include "tensora/shared/SourceBuffer.h"

// Shared by Lexer (src/lexer/Lexer.cpp) and Parser (src/parser/Parser.cpp):
// both run the ANTLR-generated TensoraLexer and must convert its tokens
// into the project's Token the same way, so this one function is the
// single place that owns that conversion.
namespace tensora {

// Converts one already-produced ANTLR token into a project Token whose
// lexeme is a view into `src`'s own buffer (not into ANTLR's copy), and
// reports LexIllegalChar / LexIntegerOverflow into `diags` when the token
// is an illegal character or an out-of-range integer literal. `src` must
// outlive any Token built from it.
Token adaptToken(antlr4::Token& antlrToken, const SourceBuffer& src, DiagnosticEngine& diags);

}  // namespace tensora
