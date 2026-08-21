#pragma once
#include <cstdint>
#include <string_view>

#include "tensora/shared/SourceSpan.h"

namespace tensora {

enum class TokenKind {
  // keywords
  KwInput,
  KwWeight,
  KwGraph,
  KwReturn,
  KwTensor,
  // names and literals
  Identifier,
  IntLiteral,
  FloatLiteral,
  // operators
  Plus,
  Minus,
  Star,
  Slash,
  At,
  Equal,
  // delimiters
  Colon,
  Comma,
  LBracket,
  RBracket,
  LBrace,
  RBrace,
  LParen,
  RParen,
  // special
  EndOfFile,
  Error
};

struct Token {
  TokenKind kind = TokenKind::Error;
  std::string_view lexeme;   // view into the source buffer
  SourceSpan span;
  int64_t intValue = 0;      // set when kind == IntLiteral
  double floatValue = 0.0;   // set when kind == FloatLiteral
};

// Short human-readable name, used by --emit-tokens and by tests.
inline const char* tokenKindName(TokenKind k) {
  switch (k) {
    case TokenKind::KwInput: return "input";
    case TokenKind::KwWeight: return "weight";
    case TokenKind::KwGraph: return "graph";
    case TokenKind::KwReturn: return "return";
    case TokenKind::KwTensor: return "Tensor";
    case TokenKind::Identifier: return "identifier";
    case TokenKind::IntLiteral: return "int";
    case TokenKind::FloatLiteral: return "float";
    case TokenKind::Plus: return "+";
    case TokenKind::Minus: return "-";
    case TokenKind::Star: return "*";
    case TokenKind::Slash: return "/";
    case TokenKind::At: return "@";
    case TokenKind::Equal: return "=";
    case TokenKind::Colon: return ":";
    case TokenKind::Comma: return ",";
    case TokenKind::LBracket: return "[";
    case TokenKind::RBracket: return "]";
    case TokenKind::LBrace: return "{";
    case TokenKind::RBrace: return "}";
    case TokenKind::LParen: return "(";
    case TokenKind::RParen: return ")";
    case TokenKind::EndOfFile: return "eof";
    case TokenKind::Error: return "error";
  }
  return "?";
}

}  // namespace tensora
