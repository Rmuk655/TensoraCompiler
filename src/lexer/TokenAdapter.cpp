// This file implements the ANTLR-token-to-project-Token conversion.
// The lexer and the parser both use it, so both agree on what a token
// means and on when a character is illegal.
#include "tensora/lexer/TokenAdapter.h"

#include <charconv>
#include <optional>
#include <string>

#include "TensoraLexer.h"

namespace tensora {

namespace {

TokenKind kindOf(size_t antlrType) {
  using generated::TensoraLexer;
  switch (antlrType) {
    case TensoraLexer::KW_INPUT: return TokenKind::KwInput;
    case TensoraLexer::KW_WEIGHT: return TokenKind::KwWeight;
    case TensoraLexer::KW_GRAPH: return TokenKind::KwGraph;
    case TensoraLexer::KW_RETURN: return TokenKind::KwReturn;
    case TensoraLexer::KW_TENSOR: return TokenKind::KwTensor;
    case TensoraLexer::IDENTIFIER: return TokenKind::Identifier;
    case TensoraLexer::INT_LITERAL: return TokenKind::IntLiteral;
    case TensoraLexer::FLOAT_LITERAL: return TokenKind::FloatLiteral;
    case TensoraLexer::PLUS: return TokenKind::Plus;
    case TensoraLexer::MINUS: return TokenKind::Minus;
    case TensoraLexer::STAR: return TokenKind::Star;
    case TensoraLexer::SLASH: return TokenKind::Slash;
    case TensoraLexer::AT: return TokenKind::At;
    case TensoraLexer::EQUAL: return TokenKind::Equal;
    case TensoraLexer::COLON: return TokenKind::Colon;
    case TensoraLexer::COMMA: return TokenKind::Comma;
    case TensoraLexer::LBRACKET: return TokenKind::LBracket;
    case TensoraLexer::RBRACKET: return TokenKind::RBracket;
    case TensoraLexer::LBRACE: return TokenKind::LBrace;
    case TensoraLexer::RBRACE: return TokenKind::RBrace;
    case TensoraLexer::LPAREN: return TokenKind::LParen;
    case TensoraLexer::RPAREN: return TokenKind::RParen;
    default: return TokenKind::Error;  // the ERROR rule, or anything unexpected
  }
}

// LANG-001: an intLiteral must fit in a signed 64-bit integer.
// from_chars reports overflow instead of throwing an exception, so a
// too-wide literal becomes a LexIntegerOverflow diagnostic, and
// scanning continues.
std::optional<int64_t> parseInt64(std::string_view lexeme) {
  int64_t value = 0;
  auto result = std::from_chars(lexeme.data(), lexeme.data() + lexeme.size(), value);
  if (result.ec != std::errc() || result.ptr != lexeme.data() + lexeme.size()) {
    return std::nullopt;
  }
  return value;
}

}  // namespace

Token adaptToken(antlr4::Token& antlrToken, const SourceBuffer& src, DiagnosticEngine& diags) {
  if (antlrToken.getType() == antlr4::Token::EOF) {
    // ANTLR's start/stop-index convention for the synthetic EOF token
    // varies with how far the lexer got. This anchors the EOF token to
    // the end of the source buffer.
    Token eof;
    eof.kind = TokenKind::EndOfFile;
    const uint32_t end = static_cast<uint32_t>(src.text().size());
    eof.span = SourceSpan{src.fileId(), end, end, static_cast<uint32_t>(antlrToken.getLine()),
                           static_cast<uint32_t>(antlrToken.getCharPositionInLine() + 1)};
    return eof;
  }

  const uint32_t startOffset = static_cast<uint32_t>(antlrToken.getStartIndex());
  const uint32_t endOffset = static_cast<uint32_t>(antlrToken.getStopIndex() + 1);
  std::string_view lexeme(src.text().data() + startOffset, endOffset - startOffset);

  Token t;
  t.kind = kindOf(static_cast<size_t>(antlrToken.getType()));
  t.lexeme = lexeme;
  t.span = SourceSpan{src.fileId(), startOffset, endOffset, static_cast<uint32_t>(antlrToken.getLine()),
                       static_cast<uint32_t>(antlrToken.getCharPositionInLine() + 1)};

  if (t.kind == TokenKind::IntLiteral) {
    if (auto value = parseInt64(lexeme)) {
      t.intValue = *value;
    } else {
      t.kind = TokenKind::Error;
      Diagnostic d;
      d.severity = Severity::Error;
      d.code = DiagCode::LexIntegerOverflow;
      d.span = t.span;
      d.message =
          "integer literal '" + std::string(lexeme) + "' does not fit in a signed 64-bit integer";
      diags.report(d);
    }
  } else if (t.kind == TokenKind::FloatLiteral) {
    t.floatValue = std::stod(std::string(lexeme));
  } else if (t.kind == TokenKind::Error) {
    Diagnostic d;
    d.severity = Severity::Error;
    d.code = DiagCode::LexIllegalChar;
    d.span = t.span;
    d.message = "illegal character '" + std::string(lexeme) + "'";
    diags.report(d);
  }

  return t;
}

}  // namespace tensora
