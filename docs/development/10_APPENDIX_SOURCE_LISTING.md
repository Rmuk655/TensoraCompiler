# 10. Appendix: Complete Source Listing

## 10.1 Purpose

The final report requires the complete source listing of the compiler.
This appendix is generated mechanically (concatenation of the actual
repository files, not hand-transcribed) so it cannot silently diverge
from what was built and tested.

## 10.2 Repository Manifest

```text
tensora/examples/attention.tns
tensora/examples/cnn.tns
tensora/examples/matmul_large.tns
tensora/examples/mlp.tns
tensora/examples/mlp_concrete.tns
tensora/grammar/Tensora.g4
tensora/include/tensora/codegen/DialectSetup.h
tensora/include/tensora/codegen/LoopTiling.h
tensora/include/tensora/codegen/MatmulTilingPass.h
tensora/include/tensora/codegen/MlirLowerer.h
tensora/include/tensora/lexer/Lexer.h
tensora/include/tensora/lexer/Token.h
tensora/include/tensora/lexer/TokenAdapter.h
tensora/include/tensora/optimization/GraphPass.h
tensora/include/tensora/optimization/Liveness.h
tensora/include/tensora/parser/Parser.h
tensora/include/tensora/semantic/Contracts.h
tensora/include/tensora/semantic/SemanticAnalyzer.h
tensora/include/tensora/semantic/Symbol.h
tensora/include/tensora/shared/Ast.h
tensora/include/tensora/shared/Diagnostic.h
tensora/include/tensora/shared/DimSolver.h
tensora/include/tensora/shared/Ir.h
tensora/include/tensora/shared/Result.h
tensora/include/tensora/shared/SourceBuffer.h
tensora/include/tensora/shared/SourceSpan.h
tensora/include/tensora/shared/Type.h
tensora/src/codegen/DialectSetup.cpp
tensora/src/codegen/LoopTiling.cpp
tensora/src/codegen/MatmulTilingPass.cpp
tensora/src/codegen/MlirLowerer.cpp
tensora/src/driver/main.cpp
tensora/src/lexer/Lexer.cpp
tensora/src/lexer/TokenAdapter.cpp
tensora/src/optimization/GraphPasses.cpp
tensora/src/optimization/Liveness.cpp
tensora/src/parser/AstBuilder.cpp
tensora/src/parser/AstBuilder.h
tensora/src/parser/Parser.cpp
tensora/src/parser/ParserErrorListener.cpp
tensora/src/parser/ParserErrorListener.h
tensora/src/parser/ParserErrorStrategy.cpp
tensora/src/parser/ParserErrorStrategy.h
tensora/src/semantic/Contracts.cpp
tensora/src/semantic/SemanticAnalyzer.cpp
tensora/src/semantic/Symbol.cpp
tensora/src/shared/DimSolver.cpp
tensora/src/shared/Ir.cpp
tensora/src/shared/Type.cpp
tensora/tools/benchmark_matmul_tiling.cpp
tensora/tools/check-deps.sh
tensora/tools/codegen_mlp_native.cpp
tensora/tools/regen_appendix.py
tensora/tools/setup-dev.sh
```

## 10.3 Frontend Listing (`grammar/`, lexer/parser headers and sources)

### `grammar/Tensora.g4`
```antlr
// Tensora grammar (RM 3.1-3.3, 3.24). This is one ANTLR4 file for the
// lexer and the parser. One file holds the whole surface language. The
// grammar is the single source of truth for lexical and syntax rules.
//
// This grammar has no embedded C++ actions. Parsing only builds an
// ANTLR parse tree. tensora::AstBuilder (a generated
// TensoraBaseVisitor subclass, in src/parser/AstBuilder.cpp) walks
// that tree and builds the AST (ast::AstProgram, see INTERFACES.md
// Section 2). Argument count and argument kind checks for each
// operator stay a semantic-analysis job (sema/Contracts.cpp), not a
// grammar job. `arg` and `argValue` stay permissive here.
grammar Tensora;

// ---- Syntactic grammar (RM 3.2, 3.3, 3.24) ----

program
    : decl* graphDecl+ EOF
    ;

decl
    : inputDecl
    | weightDecl
    ;

inputDecl
    : KW_INPUT IDENTIFIER COLON tensorType
    ;

weightDecl
    : KW_WEIGHT IDENTIFIER COLON tensorType
    ;

tensorType
    : KW_TENSOR LBRACKET axis (COMMA axis)* RBRACKET
    ;

axis
    : IDENTIFIER EQUAL dimSize
    ;

dimSize
    : INT_LITERAL
    | IDENTIFIER
    ;

graphDecl
    : KW_GRAPH IDENTIFIER LBRACE statement* returnStmt RBRACE
    ;

statement
    : IDENTIFIER EQUAL expression
    ;

returnStmt
    : KW_RETURN IDENTIFIER
    ;

// RM 3.2: each statement has one operation. The left side of a binop
// or matmul is always a plain name, never a nested expression.
expression
    : callExpr                    # callExpression
    | IDENTIFIER binop operand    # binaryExpression
    | IDENTIFIER                  # nameExpression
    ;

operand
    : IDENTIFIER
    | FLOAT_LITERAL
    ;

binop
    : AT | PLUS | MINUS | STAR | SLASH
    ;

callExpr
    : IDENTIFIER LPAREN arg (COMMA arg)* RPAREN
    ;

// This rule parses any callee without limits. Semantic analysis checks
// argument count and argument kind for each operator (RM 3.7-3.17).
arg
    : IDENTIFIER EQUAL argValue        # keywordArg
    | IDENTIFIER                       # positionalArg
    | LBRACKET axis (COMMA axis)* RBRACKET  # axisListArg
    ;

argValue
    : INT_LITERAL                                   # intArgValue
    | IDENTIFIER                                     # nameArgValue
    | LBRACKET INT_LITERAL (COMMA INT_LITERAL)* RBRACKET  # intListArgValue
    ;

// ---- Lexical grammar (RM 3.1, 3.24) ----
//
// The keyword rules come before IDENTIFIER. ANTLR's lexer picks the
// longest match across all rules, and only uses rule order to break
// ties. So an identifier that starts with a keyword, like "inputs",
// still becomes an IDENTIFIER token (maximal munch). This needs no
// extra code here.

KW_INPUT:  'input';
KW_WEIGHT: 'weight';
KW_GRAPH:  'graph';
KW_RETURN: 'return';
KW_TENSOR: 'Tensor';

IDENTIFIER:    [A-Za-z_][A-Za-z0-9_]*;
FLOAT_LITERAL: [0-9]+ '.' [0-9]+;
INT_LITERAL:   [0-9]+;

PLUS:     '+';
MINUS:    '-';
STAR:     '*';
SLASH:    '/';
AT:       '@';
EQUAL:    '=';
COLON:    ':';
COMMA:    ',';
LBRACKET: '[';
RBRACKET: ']';
LBRACE:   '{';
RBRACE:   '}';
LPAREN:   '(';
RPAREN:   ')';

COMMENT: '#' ~[\r\n]* -> skip;
WS:      [ \t\r\n]+   -> skip;

// This is the catch-all rule. It makes exactly one Error token for each
// illegal character. It never merges two bad characters into one
// token, and it never stops the scan (RM 3.1). This rule must come
// last, so every other lexer rule gets the first try at each character.
// The rule sends the Error token to the hidden channel, like WS and
// COMMENT. This keeps the bad character out of the parser's view. So
// the bad character never also becomes a second, unneeded "unexpected
// token" syntax error, on top of its LexIllegalChar diagnostic. The
// Error token still appears when code reads the full token stream by
// type (Lexer::tokenize(), and Parser.cpp's own illegal-character scan).
ERROR: . -> channel(HIDDEN);
```

### `include/tensora/lexer/Token.h`
```cpp
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
```

### `include/tensora/lexer/TokenAdapter.h`
```cpp
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
```

### `include/tensora/lexer/Lexer.h`
```cpp
#pragma once
#include <vector>

#include "tensora/shared/Diagnostic.h"
#include "tensora/lexer/Token.h"
#include "tensora/shared/SourceBuffer.h"

namespace tensora {

// Turns source text into a list of tokens that always ends with EndOfFile.
// Whitespace and '#' comments are dropped. An illegal character produces an
// Error token plus a LexIllegalChar diagnostic, and scanning continues, so
// one run can report several problems.
class Lexer {
public:
  Lexer(const SourceBuffer& src, DiagnosticEngine& diags);

  std::vector<Token> tokenize();

private:
  const SourceBuffer& src_;
  DiagnosticEngine& diags_;
};

}  // namespace tensora
```

### `src/lexer/Lexer.cpp`
```cpp
// This file implements the lexer.
// The lexer reads source text and creates tokens.
// The parser reads these tokens (Parser.cpp lexes its own copy directly;
// this path exists for standalone lexer use: --emit-tokens and the
// lexer tests).
#include "tensora/lexer/Lexer.h"

#include <memory>

#include "TensoraLexer.h"
#include "antlr4-runtime.h"
#include "tensora/lexer/TokenAdapter.h"

namespace tensora {

Lexer::Lexer(const SourceBuffer& src, DiagnosticEngine& diags) : src_(src), diags_(diags) {}

std::vector<Token> Lexer::tokenize() {
  std::vector<Token> tokens;

  antlr4::ANTLRInputStream input(src_.text());
  generated::TensoraLexer lexer(&input);
  lexer.removeErrorListeners();  // the grammar's own ERROR rule reports every illegal char; no console noise

  for (;;) {
    std::unique_ptr<antlr4::Token> tok(lexer.nextToken());
    Token t = adaptToken(*tok, src_, diags_);
    tokens.push_back(t);
    if (t.kind == TokenKind::EndOfFile) break;
  }

  return tokens;
}

}  // namespace tensora
```

### `src/lexer/TokenAdapter.cpp`
```cpp
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
```

### `include/tensora/shared/Ast.h`
```cpp
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensora/shared/SourceSpan.h"

// AST nodes (RM 3.2-3.3, 3.24; INTERFACES.md §2). These hold syntax
// only; no types or shapes. The parser accepts argument lists
// permissively. Checking each operator's argument count and kind is a
// semantic-analysis task (RM 3.7-3.17), not a parser task.
namespace tensora::ast {

struct DimSyntax {
  enum class K { Const, Symbol } k = K::Const;
  int64_t constant = 0;
  std::string symbol;
};

struct AxisSyntax {
  std::string name;
  DimSyntax size;
  SourceSpan span;
};

struct TensorTypeSyntax {
  std::vector<AxisSyntax> axes;
  SourceSpan span;
};

struct Decl {
  enum class Kind { Input, Weight } kind = Kind::Input;
  std::string name;
  TensorTypeSyntax type;
  SourceSpan span;
};

struct Expr {
  SourceSpan span;
  virtual ~Expr() = default;
};
using ExprPtr = std::unique_ptr<Expr>;

struct NameExpr : Expr {
  std::string name;
};

struct ScalarExpr : Expr {
  double value = 0.0;
};

struct MatMulExpr : Expr {
  ExprPtr lhs, rhs;
};

struct BinaryExpr : Expr {
  enum class Op { Add, Sub, Mul, Div } op = Op::Add;
  ExprPtr lhs, rhs;
};

struct Arg {
  std::optional<std::string> keyword;  // e.g. "stride", "axis", "perm", or none
  enum class Kind { Name, Int, Float, IntList, AxisList } kind = Kind::Name;
  std::string name;                      // Name / axis reference
  int64_t intVal = 0;
  double floatVal = 0.0;
  std::vector<int64_t> intList;          // perm=[...]
  std::vector<AxisSyntax> axisList;      // reshape target [batch=B, head=12, dim=64]
  SourceSpan span;
};

struct CallExpr : Expr {
  std::string callee;
  std::vector<Arg> args;
};

struct Stmt {
  std::string name;
  ExprPtr rhs;
  SourceSpan span;
};

struct GraphDecl {
  std::string name;
  std::vector<Stmt> stmts;
  std::string returnName;
  SourceSpan span;
};

struct AstProgram {
  std::vector<Decl> decls;
  std::vector<GraphDecl> graphs;
};

}  // namespace tensora::ast
```

### `include/tensora/parser/Parser.h`
```cpp
#pragma once
#include <string>

#include "tensora/shared/Ast.h"
#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/Result.h"
#include "tensora/shared/SourceBuffer.h"

namespace tensora {

// Runs the ANTLR-generated TensoraLexer and TensoraParser (grammar/
// Tensora.g4) directly over source text, then walks the parse tree to
// build the AST below (see src/parser/AstBuilder.h). No project Token
// vector sits between the lexer and the parser here (Lexer/Token exist
// only for the lexer's own standalone demo and tests; see the RM
// migration doc, Section 5). This class builds a complete, best-effort
// AST and reports syntax errors into the DiagnosticEngine. On an error,
// it recovers at the next statement or top-level declaration/graph, so
// one file can report several syntax errors in one pass. It never
// inspects shapes or types.
class Parser {
 public:
  Parser(const SourceBuffer& src, DiagnosticEngine& diags);

  Result<ast::AstProgram> parseProgram();

 private:
  const SourceBuffer& src_;
  DiagnosticEngine& diags_;
};

// For debugging and demos only (--emit-parse-tree): a parenthesized
// (LISP-style) dump of the raw ANTLR parse tree, before AstBuilder
// turns it into the AST above. No other compiler phase reads this
// dump. It exists so a student can see what ANTLR itself produced,
// separate from the project's own AST.
std::string dumpParseTree(const SourceBuffer& src);

}  // namespace tensora
```

### `src/parser/Parser.cpp`
```cpp
// This file implements the parser.
// The parser reads source text, runs the ANTLR lexer and parser, and
// creates the AST. Semantic analysis reads the AST next.
#include "tensora/parser/Parser.h"

#include "AstBuilder.h"
#include "ParserErrorListener.h"
#include "ParserErrorStrategy.h"
#include "TensoraLexer.h"
#include "TensoraParser.h"
#include "antlr4-runtime.h"
#include "tensora/lexer/TokenAdapter.h"

namespace tensora {

Parser::Parser(const SourceBuffer& src, DiagnosticEngine& diags) : src_(src), diags_(diags) {}

Result<ast::AstProgram> Parser::parseProgram() {
  antlr4::ANTLRInputStream input(src_.text());
  generated::TensoraLexer lexer(&input);
  lexer.removeErrorListeners();  // illegal characters are diagnosed below, once, for the whole file

  antlr4::CommonTokenStream tokens(&lexer);
  // The grammar routes illegal characters to a hidden ERRORS channel
  // (see grammar/Tensora.g4). So the parser below never sees them as
  // spurious "unexpected token" syntax errors. This code reports them
  // here instead, exactly once per character, matching
  // Lexer::tokenize()'s behavior for the standalone lexer path.
  // TokenAdapter.cpp is the one place both agree on what "illegal"
  // means.
  tokens.fill();
  for (antlr4::Token* tok : tokens.getTokens()) {
    if (static_cast<size_t>(tok->getType()) == generated::TensoraParser::ERROR) {
      adaptToken(*tok, src_, diags_);
    }
  }

  generated::TensoraParser parser(&tokens);
  parser.removeErrorListeners();
  ParserErrorListener errorListener(src_.fileId(), diags_);
  parser.addErrorListener(&errorListener);
  parser.setErrorHandler(std::make_shared<ParserErrorStrategy>());

  generated::TensoraParser::ProgramContext* tree = parser.program();

  AstBuilder builder(src_.fileId(), diags_);
  ast::AstProgram program = builder.build(tree);

  bool ok = !diags_.hasErrors();
  return Result<ast::AstProgram>::make(std::move(program), ok, diags_.all());
}

std::string dumpParseTree(const SourceBuffer& src) {
  antlr4::ANTLRInputStream input(src.text());
  generated::TensoraLexer lexer(&input);
  lexer.removeErrorListeners();
  antlr4::CommonTokenStream tokens(&lexer);
  generated::TensoraParser parser(&tokens);
  parser.removeErrorListeners();  // this dump is best-effort; diagnostics come from parseProgram()
  generated::TensoraParser::ProgramContext* tree = parser.program();
  return tree->toStringTree(&parser);
}

}  // namespace tensora
```

### `src/parser/AstBuilder.h`
```cpp
#pragma once
#include "TensoraParser.h"
#include "antlr4-runtime.h"
#include "tensora/shared/Ast.h"
#include "tensora/shared/Diagnostic.h"

// Walks the ANTLR parse tree (grammar/Tensora.g4) and builds the
// project AST (tensora/shared/Ast.h) (INTERFACES.md Section 2). This
// is plain recursive descent
// over context pointers. It uses dynamic_cast to the labeled-
// alternative subclasses ANTLR generates for `expression`, `arg`, and
// `argValue`, instead of the generated Visitor base: the AST is built
// from move-only std::unique_ptr nodes, and the Visitor's std::any-
// based dispatch cannot carry those. This approach keeps the
// ANTLR/AST boundary as one small, ordinary C++ class, keeping AST
// construction in project-owned code.
//
// A context whose `exception` field is set means ANTLR's
// DefaultErrorStrategy caught a syntax error while parsing it, and
// recovered. This code silently skips that subtree; it does not build
// it, even partially.
namespace tensora {

class AstBuilder {
 public:
  AstBuilder(uint32_t fileId, DiagnosticEngine& diags) : fileId_(fileId), diags_(diags) {}

  ast::AstProgram build(generated::TensoraParser::ProgramContext* ctx);

 private:
  SourceSpan spanOf(antlr4::ParserRuleContext* ctx) const;
  SourceSpan spanOf(antlr4::tree::TerminalNode* node) const;
  int64_t parseIntLiteral(antlr4::tree::TerminalNode* node);

  ast::TensorTypeSyntax buildTensorType(generated::TensoraParser::TensorTypeContext* ctx);
  ast::AxisSyntax buildAxis(generated::TensoraParser::AxisContext* ctx);
  ast::DimSyntax buildDimSize(generated::TensoraParser::DimSizeContext* ctx);
  ast::ExprPtr buildExpression(generated::TensoraParser::ExpressionContext* ctx);
  ast::ExprPtr buildOperand(generated::TensoraParser::OperandContext* ctx);
  ast::ExprPtr buildCallExpr(generated::TensoraParser::CallExprContext* ctx);
  ast::Arg buildArg(generated::TensoraParser::ArgContext* ctx);
  ast::Arg buildArgValue(generated::TensoraParser::ArgValueContext* ctx);

  uint32_t fileId_;
  DiagnosticEngine& diags_;
};

}  // namespace tensora
```

### `src/parser/AstBuilder.cpp`
```cpp
// This file implements the parse-tree-to-AST conversion.
// It walks the ANTLR parse tree and creates the AST. Semantic analysis
// reads the AST next.
#include "AstBuilder.h"

#include <charconv>
#include <optional>
#include <string>

namespace tensora {

namespace {

std::optional<int64_t> tryParseInt64(std::string_view text) {
  int64_t value = 0;
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

ast::ExprPtr makeName(std::string name, SourceSpan span) {
  auto e = std::make_unique<ast::NameExpr>();
  e->name = std::move(name);
  e->span = span;
  return e;
}

ast::ExprPtr makeScalar(double value, SourceSpan span) {
  auto e = std::make_unique<ast::ScalarExpr>();
  e->value = value;
  e->span = span;
  return e;
}

}  // namespace

SourceSpan AstBuilder::spanOf(antlr4::ParserRuleContext* ctx) const {
  antlr4::Token* start = ctx->getStart();
  antlr4::Token* stop = ctx->getStop() != nullptr ? ctx->getStop() : start;
  SourceSpan s;
  s.fileId = fileId_;
  s.beginOffset = static_cast<uint32_t>(start->getStartIndex());
  s.endOffset = static_cast<uint32_t>(stop->getStopIndex() + 1);
  s.line = static_cast<uint32_t>(start->getLine());
  s.col = static_cast<uint32_t>(start->getCharPositionInLine() + 1);
  return s;
}

SourceSpan AstBuilder::spanOf(antlr4::tree::TerminalNode* node) const {
  antlr4::Token* tok = node->getSymbol();
  SourceSpan s;
  s.fileId = fileId_;
  s.beginOffset = static_cast<uint32_t>(tok->getStartIndex());
  s.endOffset = static_cast<uint32_t>(tok->getStopIndex() + 1);
  s.line = static_cast<uint32_t>(tok->getLine());
  s.col = static_cast<uint32_t>(tok->getCharPositionInLine() + 1);
  return s;
}

// LANG-001: an INT_LITERAL must fit in a signed 64-bit integer, the
// same rule the lexer's own token-adaptation path uses
// (TokenAdapter.cpp). The parser reads INT_LITERAL text directly from
// the ANTLR token stream, not through that path (Parser.h). So it
// needs its own overflow check, to report the same LexIntegerOverflow
// diagnostic instead of silently misparsing the value.
int64_t AstBuilder::parseIntLiteral(antlr4::tree::TerminalNode* node) {
  std::string text = node->getText();
  if (auto value = tryParseInt64(text)) return *value;
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = DiagCode::LexIntegerOverflow;
  d.span = spanOf(node);
  d.message = "integer literal '" + text + "' does not fit in a signed 64-bit integer";
  diags_.report(d);
  return 0;
}

ast::TensorTypeSyntax AstBuilder::buildTensorType(generated::TensoraParser::TensorTypeContext* ctx) {
  ast::TensorTypeSyntax t;
  for (auto* axisCtx : ctx->axis()) {
    if (axisCtx->exception) continue;
    t.axes.push_back(buildAxis(axisCtx));
  }
  t.span = spanOf(ctx);
  return t;
}

ast::AxisSyntax AstBuilder::buildAxis(generated::TensoraParser::AxisContext* ctx) {
  return ast::AxisSyntax{ctx->IDENTIFIER()->getText(), buildDimSize(ctx->dimSize()), spanOf(ctx)};
}

ast::DimSyntax AstBuilder::buildDimSize(generated::TensoraParser::DimSizeContext* ctx) {
  if (auto* intTok = ctx->INT_LITERAL()) {
    return ast::DimSyntax{ast::DimSyntax::K::Const, parseIntLiteral(intTok), {}};
  }
  // A syntax error here (for example "y=@") can be caught and
  // recovered by this rule's own call, without the enclosing axis
  // context ever seeing an exception (each rule's try/catch block is
  // local; see buildArgValue's note above). So this checks explicitly,
  // instead of assuming IDENTIFIER() is non-null.
  if (auto* id = ctx->IDENTIFIER()) return ast::DimSyntax{ast::DimSyntax::K::Symbol, 0, id->getText()};
  return ast::DimSyntax{ast::DimSyntax::K::Const, 0, {}};
}

ast::ExprPtr AstBuilder::buildOperand(generated::TensoraParser::OperandContext* ctx) {
  if (auto* id = ctx->IDENTIFIER()) return makeName(id->getText(), spanOf(ctx));
  // A syntax error here (for example a bare integer literal, which this
  // rule does not accept, since RM 3.24 requires a float literal) can
  // be caught and recovered by this rule's own call, without the
  // enclosing BinaryExpression context ever seeing an exception (each
  // rule's try/catch block is local; see the matching note on
  // buildArgValue above). So this checks explicitly, instead of
  // assuming FLOAT_LITERAL() is non-null.
  if (auto* f = ctx->FLOAT_LITERAL()) return makeScalar(std::stod(f->getText()), spanOf(ctx));
  return makeScalar(0.0, spanOf(ctx));
}

ast::ExprPtr AstBuilder::buildCallExpr(generated::TensoraParser::CallExprContext* ctx) {
  auto call = std::make_unique<ast::CallExpr>();
  call->callee = ctx->IDENTIFIER()->getText();
  for (auto* argCtx : ctx->arg()) {
    if (argCtx->exception) continue;
    call->args.push_back(buildArg(argCtx));
  }
  call->span = spanOf(ctx);
  return call;
}

ast::ExprPtr AstBuilder::buildExpression(generated::TensoraParser::ExpressionContext* ctx) {
  using namespace generated;
  if (ctx == nullptr || ctx->exception) return nullptr;

  if (auto* c = dynamic_cast<TensoraParser::CallExpressionContext*>(ctx)) {
    return buildCallExpr(c->callExpr());
  }
  if (auto* b = dynamic_cast<TensoraParser::BinaryExpressionContext*>(ctx)) {
    ast::ExprPtr lhs = makeName(b->IDENTIFIER()->getText(), spanOf(b->IDENTIFIER()));
    ast::ExprPtr rhs = buildOperand(b->operand());
    SourceSpan span = spanOf(b);
    size_t tokenType = b->binop()->getStart()->getType();
    if (tokenType == TensoraParser::AT) {
      auto e = std::make_unique<ast::MatMulExpr>();
      e->lhs = std::move(lhs);
      e->rhs = std::move(rhs);
      e->span = span;
      return e;
    }
    ast::BinaryExpr::Op op = ast::BinaryExpr::Op::Add;
    switch (tokenType) {
      case TensoraParser::PLUS: op = ast::BinaryExpr::Op::Add; break;
      case TensoraParser::MINUS: op = ast::BinaryExpr::Op::Sub; break;
      case TensoraParser::STAR: op = ast::BinaryExpr::Op::Mul; break;
      case TensoraParser::SLASH: op = ast::BinaryExpr::Op::Div; break;
      default: break;
    }
    auto e = std::make_unique<ast::BinaryExpr>();
    e->op = op;
    e->lhs = std::move(lhs);
    e->rhs = std::move(rhs);
    e->span = span;
    return e;
  }
  if (auto* n = dynamic_cast<TensoraParser::NameExpressionContext*>(ctx)) {
    return makeName(n->IDENTIFIER()->getText(), spanOf(ctx));
  }
  return nullptr;
}

ast::Arg AstBuilder::buildArgValue(generated::TensoraParser::ArgValueContext* ctx) {
  using namespace generated;
  ast::Arg a;
  a.span = spanOf(ctx);
  // A syntax error inside argValue (for example "stride=)") can be
  // caught and recovered by this rule's own call, without the
  // enclosing `arg` context ever seeing an exception (each rule's
  // try/catch block is local). So `ctx` here may be the plain
  // ArgValueContext base, not one of its three labeled subclasses.
  // This checks explicitly, instead of assuming one of the
  // dynamic_casts below succeeds.
  if (ctx->exception) return a;
  if (auto* iv = dynamic_cast<TensoraParser::IntArgValueContext*>(ctx)) {
    a.kind = ast::Arg::Kind::Int;
    a.intVal = parseIntLiteral(iv->INT_LITERAL());
    return a;
  }
  if (auto* nv = dynamic_cast<TensoraParser::NameArgValueContext*>(ctx)) {
    a.kind = ast::Arg::Kind::Name;
    a.name = nv->IDENTIFIER()->getText();
    return a;
  }
  if (auto* lv = dynamic_cast<TensoraParser::IntListArgValueContext*>(ctx)) {
    a.kind = ast::Arg::Kind::IntList;
    for (auto* tok : lv->INT_LITERAL()) {
      a.intList.push_back(parseIntLiteral(tok));
    }
  }
  return a;
}

ast::Arg AstBuilder::buildArg(generated::TensoraParser::ArgContext* ctx) {
  using namespace generated;
  ast::Arg fallback;
  fallback.span = spanOf(ctx);
  if (ctx->exception) return fallback;  // same reasoning as buildArgValue above
  if (auto* k = dynamic_cast<TensoraParser::KeywordArgContext*>(ctx)) {
    ast::Arg a = buildArgValue(k->argValue());
    a.keyword = k->IDENTIFIER()->getText();
    a.span = spanOf(ctx);
    return a;
  }
  if (auto* p = dynamic_cast<TensoraParser::PositionalArgContext*>(ctx)) {
    ast::Arg a;
    a.kind = ast::Arg::Kind::Name;
    a.name = p->IDENTIFIER()->getText();
    a.span = spanOf(ctx);
    return a;
  }
  if (auto* axisListCtx = dynamic_cast<TensoraParser::AxisListArgContext*>(ctx)) {
    ast::Arg a;
    a.kind = ast::Arg::Kind::AxisList;
    for (auto* axisCtx : axisListCtx->axis()) {
      if (axisCtx->exception) continue;
      a.axisList.push_back(buildAxis(axisCtx));
    }
    a.span = spanOf(ctx);
    return a;
  }
  return fallback;
}

ast::AstProgram AstBuilder::build(generated::TensoraParser::ProgramContext* ctx) {
  ast::AstProgram program;

  for (auto* declCtx : ctx->decl()) {
    if (declCtx->exception) continue;
    if (auto* inputCtx = declCtx->inputDecl()) {
      if (inputCtx->exception) continue;
      ast::Decl d;
      d.kind = ast::Decl::Kind::Input;
      d.name = inputCtx->IDENTIFIER()->getText();
      d.type = buildTensorType(inputCtx->tensorType());
      d.span = spanOf(declCtx);
      program.decls.push_back(std::move(d));
    } else if (auto* weightCtx = declCtx->weightDecl()) {
      if (weightCtx->exception) continue;
      ast::Decl d;
      d.kind = ast::Decl::Kind::Weight;
      d.name = weightCtx->IDENTIFIER()->getText();
      d.type = buildTensorType(weightCtx->tensorType());
      d.span = spanOf(declCtx);
      program.decls.push_back(std::move(d));
    }
  }

  for (auto* graphCtx : ctx->graphDecl()) {
    if (graphCtx->exception) continue;
    ast::GraphDecl g;
    g.name = graphCtx->IDENTIFIER()->getText();
    for (auto* stmtCtx : graphCtx->statement()) {
      if (stmtCtx->exception) continue;
      ast::Stmt s;
      s.name = stmtCtx->IDENTIFIER()->getText();
      s.rhs = buildExpression(stmtCtx->expression());
      s.span = spanOf(stmtCtx);
      g.stmts.push_back(std::move(s));
    }
    auto* retCtx = graphCtx->returnStmt();
    if (retCtx != nullptr && retCtx->exception == nullptr) {
      g.returnName = retCtx->IDENTIFIER()->getText();
    }
    g.span = spanOf(graphCtx);
    program.graphs.push_back(std::move(g));
  }

  return program;
}

}  // namespace tensora
```

### `src/parser/ParserErrorListener.h`
```cpp
#pragma once
#include "antlr4-runtime.h"
#include "tensora/shared/Diagnostic.h"

// Converts ANTLR's syntax-error callback into a project Diagnostic.
//
// ANTLR reports two kinds of syntax error here. A mismatched-token
// error (it found token X, but expected token Y) names one specific
// expected token, so this becomes DiagCode::ParseExpected. A "no
// viable alternative" error (antlr4::NoViableAltException) happens
// when the current token cannot start any rule the parser could try
// next. There is no single expected token to name, so this becomes
// DiagCode::ParseUnexpectedToken instead.
//
// Recovery itself uses ParserErrorStrategy (panic-mode resync to the
// enclosing rule's follow set).
namespace tensora {

class ParserErrorListener : public antlr4::BaseErrorListener {
 public:
  ParserErrorListener(uint32_t fileId, DiagnosticEngine& diags) : fileId_(fileId), diags_(diags) {}

  void syntaxError(antlr4::Recognizer* recognizer, antlr4::Token* offendingSymbol, size_t line,
                    size_t charPositionInLine, const std::string& msg,
                    std::exception_ptr e) override;

 private:
  uint32_t fileId_;
  DiagnosticEngine& diags_;
};

}  // namespace tensora
```

### `src/parser/ParserErrorListener.cpp`
```cpp
#include "ParserErrorListener.h"

namespace tensora {

namespace {

// See the class comment above: a "no viable alternative" error has no
// single expected token to name, so it gets its own DiagCode.
DiagCode classify(std::exception_ptr e) {
  if (!e) return DiagCode::ParseExpected;
  try {
    std::rethrow_exception(e);
  } catch (const antlr4::NoViableAltException&) {
    return DiagCode::ParseUnexpectedToken;
  } catch (...) {
    return DiagCode::ParseExpected;
  }
}

}  // namespace

void ParserErrorListener::syntaxError(antlr4::Recognizer*, antlr4::Token* offendingSymbol, size_t line,
                                       size_t charPositionInLine, const std::string& msg,
                                       std::exception_ptr e) {
  SourceSpan span;
  span.fileId = fileId_;
  span.line = static_cast<uint32_t>(line);
  span.col = static_cast<uint32_t>(charPositionInLine + 1);
  if (offendingSymbol != nullptr) {
    span.beginOffset = static_cast<uint32_t>(offendingSymbol->getStartIndex());
    span.endOffset = static_cast<uint32_t>(offendingSymbol->getStopIndex() + 1);
  } else {
    span.beginOffset = span.endOffset = 0;
  }

  Diagnostic d;
  d.severity = Severity::Error;
  d.code = classify(e);
  d.span = span;
  d.message = msg;
  diags_.report(d);
}

}  // namespace tensora
```

### `src/parser/ParserErrorStrategy.h`
```cpp
#pragma once
#include "antlr4-runtime.h"

// This file defines the parser's error-recovery strategy.
// The parser uses this strategy when it finds a syntax error.
//
// ANTLR's default strategy repairs some errors by inserting or deleting
// one token (for example, it can silently add a missing "="). That repair
// hides the error instead of dropping the bad statement. This strategy
// turns that repair off, so every syntax error falls back to panic-mode
// recovery: the parser drops the bad construct and resyncs at the next
// safe point (the next statement or declaration).
//
// Turning off single-token repair has one side effect: after the parser
// drops a bad construct, it can resync at a point that itself looks like
// the start of a new (still broken) construct, causing a second, spurious
// diagnostic from the same root cause. The base class only requires one
// clean token match to stop suppressing new errors; that is too little
// here. This class instead requires a few consecutive clean matches
// (`kCleanShiftsRequired`) before it reports a new error.
namespace tensora {

class ParserErrorStrategy : public antlr4::DefaultErrorStrategy {
 public:
  bool singleTokenInsertion(antlr4::Parser* recognizer) override;
  antlr4::Token* singleTokenDeletion(antlr4::Parser* recognizer) override;
  void reportError(antlr4::Parser* recognizer, const antlr4::RecognitionException& e) override;
  void reportMatch(antlr4::Parser* recognizer) override;

 private:
  static constexpr int kCleanShiftsRequired = 3;
  int cleanShiftsSinceError_ = kCleanShiftsRequired;
};

}  // namespace tensora
```

### `src/parser/ParserErrorStrategy.cpp`
```cpp
#include "ParserErrorStrategy.h"

namespace tensora {

// Always refuse the single-token repair. The base class's recoverInline()
// then throws instead of silently inserting a token, so the enclosing
// rule's own error handling drops the bad construct (panic-mode recovery).
bool ParserErrorStrategy::singleTokenInsertion(antlr4::Parser*) { return false; }

// Always refuse the single-token repair. Returning nullptr tells the base
// class no deletion is possible, so it falls back to panic-mode recovery
// the same way singleTokenInsertion above does.
antlr4::Token* ParserErrorStrategy::singleTokenDeletion(antlr4::Parser*) { return nullptr; }

// Report the error only after enough clean token matches since the last
// one. A cascade error (the same root cause, reported again right after
// recovery) has too few clean matches in between and is dropped here
// instead of reaching the DiagnosticEngine.
void ParserErrorStrategy::reportError(antlr4::Parser* recognizer, const antlr4::RecognitionException& e) {
  if (cleanShiftsSinceError_ < kCleanShiftsRequired) return;
  cleanShiftsSinceError_ = 0;
  antlr4::DefaultErrorStrategy::reportError(recognizer, e);
}

void ParserErrorStrategy::reportMatch(antlr4::Parser* recognizer) {
  if (cleanShiftsSinceError_ < kCleanShiftsRequired) ++cleanShiftsSinceError_;
  antlr4::DefaultErrorStrategy::reportMatch(recognizer);
}

}  // namespace tensora
```

### `include/tensora/shared/SourceSpan.h`
```cpp
#pragma once
#include <cstdint>

namespace tensora {

// A half-open byte range [beginOffset, endOffset) in one source file,
// plus the line and column where it starts. Line and column are 1-based.
struct SourceSpan {
  uint32_t fileId = 0;
  uint32_t beginOffset = 0;
  uint32_t endOffset = 0;
  uint32_t line = 1;
  uint32_t col = 1;
};

}  // namespace tensora
```

### `include/tensora/shared/SourceBuffer.h`
```cpp
#pragma once
#include <cstdint>
#include <string>
#include <utility>

namespace tensora {

// Owns the text of one source file for the whole compilation.
// Tokens keep string_views into this text, so a SourceBuffer must outlive
// the tokens taken from it.
class SourceBuffer {
public:
  SourceBuffer(std::string name, std::string text, uint32_t fileId = 0)
      : name_(std::move(name)), text_(std::move(text)), fileId_(fileId) {}

  const std::string& text() const { return text_; }
  const std::string& name() const { return name_; }
  uint32_t fileId() const { return fileId_; }

private:
  std::string name_;
  std::string text_;
  uint32_t fileId_;
};

}  // namespace tensora
```

### `include/tensora/shared/Result.h`
```cpp
#pragma once
#include <optional>
#include <utility>

#include "tensora/shared/Diagnostic.h"

namespace tensora {

// A phase's outcome: a best-effort value (built even after recovery
// from an error), plus every diagnostic reported while building it.
// ok() is false when any Error-severity diagnostic was reported, but
// value() still holds the built value, so a caller or a test can look
// at the partial result.
template <class T>
class Result {
 public:
  static Result make(T value, bool ok, DiagnosticList errors) {
    Result r;
    r.value_.emplace(std::move(value));  // construction, not assignment: T need not be move-assignable
    r.ok_ = ok;
    r.errors_ = std::move(errors);
    return r;
  }

  bool ok() const { return ok_; }
  T& value() { return *value_; }
  const T& value() const { return *value_; }
  const DiagnosticList& errors() const { return errors_; }

 private:
  std::optional<T> value_;
  bool ok_ = false;
  DiagnosticList errors_;
};

}  // namespace tensora
```

### `include/tensora/shared/Diagnostic.h`
```cpp
#pragma once
#include <string>
#include <vector>

#include "tensora/shared/SourceBuffer.h"
#include "tensora/shared/SourceSpan.h"

namespace tensora {

enum class Severity { Error, Warning, Note };

// These codes stay stable, so tests can check the error category
// instead of the exact wording. The parser and semantic phases share
// this one set of codes.
enum class DiagCode {
  LexIllegalChar,
  LexIntegerOverflow,
  ParseUnexpectedToken,
  ParseExpected,
  SemaUndefinedName,
  SemaRedeclaration,
  SemaArgCount,
  SemaArgKind,
  SemaTypeMismatch,
  SemaRankMismatch,
  SemaShapeMismatch,
  SemaAxisNameMismatch,
  SemaBroadcast,
  SemaReshapeCount,
  SemaTransposePerm,
  SemaAxisNotFound,
  SemaConvChannel,
  SemaConvArithmetic,
  SemaUnresolvableShape,
  // RM 3.3: an axis size must be a concrete positive integer, or a
  // symbolic dimension. The compiler never checks a symbolic dimension
  // for a positive value statically.
  SemaNonPositiveDimension,
  // RM 3.3: axis names must be unique within one tensor type (Stage 9
  // decision, 2026-08-19 -- see docs/architecture/DECISIONS.md ARCH-013).
  SemaDuplicateAxisName,
  // IR.md §6: GraphVerifier re-checks TCG invariants (SSA, acyclic and
  // closed, single output), no matter how the graph was built. A
  // violation here is a compiler-internal defect (a buggy pass or a bad
  // hand-built fixture), not a user-program error. Every Sema* code
  // above is instead about the *source program*'s meaning.
  IrInvariantViolation
};

struct Diagnostic {
  Severity severity = Severity::Error;
  DiagCode code = DiagCode::LexIllegalChar;
  SourceSpan span;
  std::string message;
};

// INTERFACES.md §0's DiagnosticList alias.
using DiagnosticList = std::vector<Diagnostic>;

// Collects diagnostics from every phase. One engine per compilation run.
class DiagnosticEngine {
public:
  void report(const Diagnostic& d) { items_.push_back(d); }

  bool hasErrors() const {
    for (const Diagnostic& d : items_) {
      if (d.severity == Severity::Error) return true;
    }
    return false;
  }

  const DiagnosticList& all() const { return items_; }

  // One line per diagnostic: "file:line:col: error: message".
  std::string render(const SourceBuffer& src) const {
    std::string out;
    for (const Diagnostic& d : items_) {
      out += src.name();
      out += ":" + std::to_string(d.span.line);
      out += ":" + std::to_string(d.span.col);
      out += (d.severity == Severity::Error) ? ": error: " : ": note: ";
      out += d.message;
      out += "\n";
    }
    return out;
  }

private:
  DiagnosticList items_;
};

}  // namespace tensora
```

## 10.4 Semantic Listing (symbol table, type system, contracts, `SemanticAnalyzer`)

### `include/tensora/semantic/Symbol.h`
```cpp
#pragma once
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "tensora/shared/SourceSpan.h"
#include "tensora/shared/Type.h"

// Name resolution and scope (RM 3.6). There are only two scope levels,
// matching the grammar: a flat global scope (input/weight declarations,
// shared by every graph) and one flat local scope per graph (value
// bindings from statements). The local scope resets at the start of
// each graph.
namespace tensora {

struct Symbol {
  enum class Kind { Input, Weight, Value };
  std::string name;
  Kind kind = Kind::Value;
  const Type* type = nullptr;  // verified decl type, or the RHS's inferred type
  SourceSpan span;             // declaration/binding site
};

class SymbolTable {
 public:
  // Returns false, and does not insert, if `name` is already declared
  // globally -- RM 3.6 says redeclaring a top-level name is rejected.
  bool declareGlobal(const Symbol& sym);
  const Symbol* lookupGlobal(std::string_view name) const;

  // Resets the local (per-graph) scope. Call this once at the start of
  // each graph.
  void enterGraph();

  enum class BindOutcome {
    Ok,           // A fresh local bind, or a reassignment of an existing local Value (shadows, RM 3.6).
    KindConflict  // `name` already exists globally as Input/Weight (see the note below).
  };
  // RM 3.6 says a rebind, in the same graph, of a name already bound to
  // a different declaration kind is rejected. It also says reassigning
  // an already-bound value name within a graph is allowed and shadows
  // the old value. The only way a name can have two different *kinds*
  // is Input/Weight (global) vs. Value (local); a purely local rebind
  // is always Value -> Value, which the second rule allows. So this
  // class treats a local assignment that shadows a GLOBAL input/weight
  // name as the kind conflict the first rule forbids. A local
  // assignment that only shadows an earlier LOCAL binding is always
  // Ok. (The RM text does not force this reading; see the semantic
  // implementation report's Specification/Ambiguity Issues section.)
  BindOutcome bindLocal(const std::string& name, Symbol sym);

  // Resolves a name for use inside the current graph. It checks the
  // most recent local binding first, then the global scope. Returns
  // nullptr if the name is undefined (RM 3.6).
  const Symbol* resolve(std::string_view name) const;

  // Declaration-order views of the two scopes. These hold the same
  // Symbol objects `resolve` reads, in the order they were first
  // accepted. (A rebind that shadows an earlier local does not add a
  // second entry; it updates the existing one in place, the same way
  // `locals_` does.) --emit-symbols (main.cpp) renders this data, so
  // its output shows the real resolution state, not a second, separate
  // rebuild of it. `orderedLocals` reflects the *current* graph's
  // scope. The next call to `enterGraph` clears it.
  const std::vector<const Symbol*>& orderedGlobals() const { return globalOrder_; }
  const std::vector<const Symbol*>& orderedLocals() const { return localOrder_; }

 private:
  std::unordered_map<std::string, Symbol> globals_;
  std::unordered_map<std::string, Symbol> locals_;
  std::vector<const Symbol*> globalOrder_;
  std::vector<const Symbol*> localOrder_;
};

}  // namespace tensora
```

### `src/semantic/Symbol.cpp`
```cpp
// This file implements the symbol table.
// Semantic analysis uses it to resolve names and enforce scope rules
// (RM 3.6).
#include "tensora/semantic/Symbol.h"

namespace tensora {

bool SymbolTable::declareGlobal(const Symbol& sym) {
  auto [it, inserted] = globals_.emplace(sym.name, sym);
  if (inserted) globalOrder_.push_back(&it->second);
  return inserted;
}

const Symbol* SymbolTable::lookupGlobal(std::string_view name) const {
  auto it = globals_.find(std::string(name));
  return it == globals_.end() ? nullptr : &it->second;
}

void SymbolTable::enterGraph() {
  locals_.clear();
  localOrder_.clear();
}

SymbolTable::BindOutcome SymbolTable::bindLocal(const std::string& name, Symbol sym) {
  if (locals_.find(name) == locals_.end() && lookupGlobal(name) != nullptr) {
    return BindOutcome::KindConflict;
  }
  auto it = locals_.find(name);
  if (it == locals_.end()) {
    auto [newIt, inserted] = locals_.emplace(name, std::move(sym));
    localOrder_.push_back(&newIt->second);
  } else {
    it->second = std::move(sym);  // shadows the earlier local in place, same entry
  }
  return BindOutcome::Ok;
}

const Symbol* SymbolTable::resolve(std::string_view name) const {
  auto it = locals_.find(std::string(name));
  if (it != locals_.end()) return &it->second;
  return lookupGlobal(name);
}

}  // namespace tensora
```

### `include/tensora/shared/Type.h`
```cpp
#pragma once
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// The Tensora type system (RM 3.3-3.5; IR.md §2; ARCH-005/006).
// TypeContext interns and owns every type. Code refers to a type by
// raw const pointer (TypePtr = const Type*), for the life of one
// TypedProgram. The element type is always f32 in v1; there is no
// dtype field (RM 3.23).
namespace tensora {

// An interned symbolic-dimension name. The name -> id table is shared
// across the whole program (one TypeContext), only so the same
// spelling always gets the same id. This does NOT unify the symbol
// across graphs. Each graph gets its own fresh DimSolver
// (SemanticAnalyzer::analyzeGraph). So a symbol named "B" in one graph
// and "B" in another graph are independent, matching RM 3.4 ("within
// one graph").
using SymbolId = uint32_t;

struct Dimension {
  enum class Kind { Const, Symbol };
  Kind kind = Kind::Const;
  int64_t constant = 0;  // valid iff kind == Const (> 0, RM 3.3 "concrete positive integer")
  SymbolId symbol = 0;   // valid iff kind == Symbol

  static Dimension makeConst(int64_t v) {
    Dimension d;
    d.kind = Kind::Const;
    d.constant = v;
    return d;
  }
  static Dimension makeSymbol(SymbolId s) {
    Dimension d;
    d.kind = Kind::Symbol;
    d.symbol = s;
    return d;
  }
};

struct Axis {
  std::string name;  // e.g. "batch", "out", "kseq" (RM 3.5: axis names, not a reserved namespace)
  Dimension size;
};

class Type {
 public:
  enum class Kind { Tensor, Scalar, Invalid };
  explicit Type(Kind k) : kind_(k) {}
  virtual ~Type() = default;
  Kind kind() const { return kind_; }

 private:
  Kind kind_;
};

// kind() == Tensor. The axes are ordered and never empty (RM 3.3); the rank equals axes().size().
class TensorType : public Type {
 public:
  explicit TensorType(std::vector<Axis> axes) : Type(Kind::Tensor), axes_(std::move(axes)) {}
  const std::vector<Axis>& axes() const { return axes_; }
  size_t rank() const { return axes_.size(); }

 private:
  std::vector<Axis> axes_;
};

// kind() == Scalar: an f32 scalar (a bare floatLiteral operand, RM 3.8).
class ScalarType : public Type {
 public:
  ScalarType() : Type(Kind::Scalar) {}
};

// kind() == Invalid: only used as semantic analysis's error-recovery
// poison value (ARCH-007). This never appears on a Value in a graph
// that GraphVerifier accepts as clean.
class InvalidType : public Type {
 public:
  InvalidType() : Type(Kind::Invalid) {}
};

// Owns every Type object, and the SymbolId <-> name table, for one
// compilation unit. Symbol interning applies across the whole program.
// (RM 3.4's "within one graph" scoping applies to DimSolver's
// unification state, which starts fresh for each graph -- see
// DimSolver.h -- not to naming.)
class TypeContext {
 public:
  const TensorType* tensor(std::vector<Axis> axes) {
    tensors_.emplace_back(std::move(axes));
    return &tensors_.back();
  }
  const ScalarType* scalar() { return &scalar_; }
  const InvalidType* invalid() { return &invalid_; }

  SymbolId internSymbol(std::string_view name) {
    auto it = symbolIds_.find(std::string(name));
    if (it != symbolIds_.end()) return it->second;
    SymbolId id = static_cast<SymbolId>(symbolNames_.size());
    symbolNames_.emplace_back(name);
    symbolIds_.emplace(std::string(name), id);
    return id;
  }
  std::string_view symbolName(SymbolId id) const { return symbolNames_.at(id); }

 private:
  std::deque<TensorType> tensors_;  // Keeps stable addresses, so a TypePtr stays valid as long as this context does.
  ScalarType scalar_;
  InvalidType invalid_;
  std::vector<std::string> symbolNames_;
  std::unordered_map<std::string, SymbolId> symbolIds_;
};

// Checks type equality by value: both names AND sizes must match
// exactly. Used, for example, by relu's rule that the result type must
// equal the input type exactly (RM 3.10).
bool dimensionsEqual(const Dimension& a, const Dimension& b);
bool typesEqual(const Type* a, const Type* b);

// Renders a type or dimension for a diagnostic message, for example
// "Tensor[batch=B, feat=784]", "784", or "B".
std::string describeDimension(const Dimension& d, const TypeContext& ctx);
std::string describeType(const Type* t, const TypeContext& ctx);

}  // namespace tensora
```

### `src/shared/Type.cpp`
```cpp
// This file implements the tensor type representation.
// Semantic analysis, optimization, and codegen all read and create
// these types, so this file lives in shared/.
#include "tensora/shared/Type.h"

#include <sstream>

namespace tensora {

bool dimensionsEqual(const Dimension& a, const Dimension& b) {
  if (a.kind != b.kind) return false;
  if (a.kind == Dimension::Kind::Const) return a.constant == b.constant;
  return a.symbol == b.symbol;
}

bool typesEqual(const Type* a, const Type* b) {
  if (a->kind() != b->kind()) return false;
  if (a->kind() == Type::Kind::Tensor) {
    const auto* ta = static_cast<const TensorType*>(a);
    const auto* tb = static_cast<const TensorType*>(b);
    if (ta->rank() != tb->rank()) return false;
    for (size_t i = 0; i < ta->rank(); ++i) {
      if (ta->axes()[i].name != tb->axes()[i].name) return false;
      if (!dimensionsEqual(ta->axes()[i].size, tb->axes()[i].size)) return false;
    }
    return true;
  }
  return true;  // Scalar == Scalar, Invalid == Invalid (kind already matched)
}

std::string describeDimension(const Dimension& d, const TypeContext& ctx) {
  if (d.kind == Dimension::Kind::Const) return std::to_string(d.constant);
  return std::string(ctx.symbolName(d.symbol));
}

std::string describeType(const Type* t, const TypeContext& ctx) {
  switch (t->kind()) {
    case Type::Kind::Scalar:
      return "scalar";
    case Type::Kind::Invalid:
      return "<invalid>";
    case Type::Kind::Tensor: {
      const auto* tt = static_cast<const TensorType*>(t);
      std::ostringstream out;
      out << "Tensor[";
      for (size_t i = 0; i < tt->rank(); ++i) {
        if (i) out << ", ";
        out << tt->axes()[i].name << "=" << describeDimension(tt->axes()[i].size, ctx);
      }
      out << "]";
      return out.str();
    }
  }
  return "?";
}

}  // namespace tensora
```

### `include/tensora/shared/DimSolver.h`
```cpp
#pragma once
#include <unordered_map>
#include <vector>

#include "tensora/shared/Type.h"

// The bounded dimension solver (RM 3.4; ARCH-006). It uses union-find
// over SymbolId, plus integer arithmetic. It handles exactly three
// cases: a concrete integer, a symbolic variable, or unifying the two.
// It has no SMT solver and no general nonlinear solving. Each graph
// gets its own DimSolver (RM 3.4 scopes dimension reasoning to "within
// one graph"), even though TypeContext interns SymbolId names across
// the whole program.
namespace tensora {

// INTERFACES.md sketches sameElementCount as returning bool. Here it
// returns an enum instead, because semantic analysis must tell two
// cases apart: "the counts are decidable and differ" (RM 3.12's
// SemaReshapeCount) versus "this needs reasoning outside the bounded
// subset" (RM 3.4's SemaUnresolvableShape). These are different
// diagnostics for different problems, not the same failure.
struct ElementCountResult {
  enum class Status {
    Equal,          // The symbolic factors match, or there are none, and the constant products are equal.
    Mismatch,       // The symbolic factors match, or there are none, and the constant products differ.
    Unresolvable    // The symbolic factors differ in identity or in multiplicity.
  } status;
  int64_t lhsConstProduct = 1;
  int64_t rhsConstProduct = 1;
};

class DimSolver {
 public:
  // Const == Const: checked, not assumed. Symbol ~ Symbol: always
  // unifies; the two become synonyms. Symbol := Const: binds the
  // symbol. Returns false only when both sides already resolve to
  // different constants.
  bool unify(Dimension a, Dimension b);

  // Follows the union-find chain, with path compression, to the
  // current best-known representative: an unbound Symbol, or the Const
  // it was bound to, directly or through a chain.
  Dimension resolve(Dimension d) const;

  // RM 3.4 / RM 3.12: checks product-of-sizes equality within the
  // bounded subset. It cancels a symbolic factor only when that factor
  // appears with the same multiplicity on both sides. Otherwise the
  // comparison falls outside the subset (Unresolvable).
  ElementCountResult sameElementCount(const std::vector<Dimension>& lhs,
                                      const std::vector<Dimension>& rhs) const;

 private:
  mutable std::unordered_map<SymbolId, Dimension> rep_;  // Path-compressed each time resolve() runs.
};

}  // namespace tensora
```

### `src/shared/DimSolver.cpp`
```cpp
// This file implements the bounded symbolic-dimension solver.
// Semantic analysis uses it to unify axis sizes and check element-count
// equality (RM 3.4), without a general SMT solver.
#include "tensora/shared/DimSolver.h"

#include <algorithm>

namespace tensora {

Dimension DimSolver::resolve(Dimension d) const {
  while (d.kind == Dimension::Kind::Symbol) {
    auto it = rep_.find(d.symbol);
    if (it == rep_.end()) break;  // unbound: this symbol is its own representative
    if (dimensionsEqual(it->second, d)) break;  // already a fixed point
    d = it->second;
  }
  return d;
}

bool DimSolver::unify(Dimension a, Dimension b) {
  Dimension ra = resolve(a);
  Dimension rb = resolve(b);

  if (ra.kind == Dimension::Kind::Const && rb.kind == Dimension::Kind::Const) {
    return ra.constant == rb.constant;
  }
  if (ra.kind == Dimension::Kind::Symbol && rb.kind == Dimension::Kind::Symbol) {
    if (ra.symbol == rb.symbol) return true;
    rep_[ra.symbol] = rb;  // ra becomes a synonym of rb
    return true;
  }
  // Exactly one side is a free Symbol; bind it to the other's resolved value.
  if (ra.kind == Dimension::Kind::Symbol) {
    rep_[ra.symbol] = rb;
  } else {
    rep_[rb.symbol] = ra;
  }
  return true;
}

ElementCountResult DimSolver::sameElementCount(const std::vector<Dimension>& lhs,
                                                const std::vector<Dimension>& rhs) const {
  auto split = [this](const std::vector<Dimension>& dims, int64_t& constProduct,
                       std::vector<SymbolId>& symbols) {
    constProduct = 1;
    for (const Dimension& raw : dims) {
      Dimension d = resolve(raw);
      if (d.kind == Dimension::Kind::Const) {
        constProduct *= d.constant;
      } else {
        symbols.push_back(d.symbol);
      }
    }
    std::sort(symbols.begin(), symbols.end());
  };

  ElementCountResult result;
  std::vector<SymbolId> lhsSymbols, rhsSymbols;
  split(lhs, result.lhsConstProduct, lhsSymbols);
  split(rhs, result.rhsConstProduct, rhsSymbols);

  if (lhsSymbols != rhsSymbols) {
    result.status = ElementCountResult::Status::Unresolvable;
    return result;
  }
  result.status = (result.lhsConstProduct == result.rhsConstProduct)
                       ? ElementCountResult::Status::Equal
                       : ElementCountResult::Status::Mismatch;
  return result;
}

}  // namespace tensora
```

### `include/tensora/semantic/Contracts.h`
```cpp
#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

#include "tensora/shared/Ast.h"
#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/Ir.h"
#include "tensora/shared/DimSolver.h"
#include "tensora/shared/Type.h"

// One contract checker per operator (RM 3.7-3.17; ARCHITECTURE.md §6.2).
// Each is an attribute-grammar rule: it builds the result TensorType
// from the operand types, or it reports a diagnostic and returns
// ctx.types.invalid() (ARCH-007 poisoning). SemanticAnalyzer's dispatch
// layer checks argument *count* and *kind* (positional vs. keyword,
// which keyword names are valid) before it calls a contract. So each
// contract always gets arguments with the right count and kind, and it
// only needs to check shapes and types.
namespace tensora {

struct ContractCtx {
  TypeContext& types;
  DimSolver& solver;
  DiagnosticEngine& diags;
  SourceSpan span;
};

const Type* checkMatMul(const Type* a, const Type* b, ContractCtx& ctx);
const Type* checkLinear(const Type* x, const Type* w, const Type* b, ContractCtx& ctx);
const Type* checkElementwise(ast::BinaryExpr::Op op, const Type* a, const Type* bOrScalar, ContractCtx& ctx);
const Type* checkRelu(const Type* x, ContractCtx& ctx);
const Type* checkSoftmax(const Type* x, std::string_view axis, ContractCtx& ctx);
const Type* checkReshape(const Type* x, const std::vector<Axis>& target, ContractCtx& ctx);
const Type* checkFlatten(const Type* x, ContractCtx& ctx);
const Type* checkTranspose(const Type* x, const std::vector<uint32_t>& perm, ContractCtx& ctx);
// sumOrMean must be OpKind::Sum or OpKind::Mean.
const Type* checkReduce(OpKind sumOrMean, const Type* x, std::string_view axis, ContractCtx& ctx);
const Type* checkConv2D(const Type* x, const Type* w, const Type* b, int64_t stride, int64_t padding,
                         ContractCtx& ctx);
// The caller already resolves `stride` before this call (RM 3.17: it
// defaults to `size` when the call omits it), so this contract always
// gets a concrete value.
const Type* checkMaxPool(const Type* x, int64_t size, int64_t stride, ContractCtx& ctx);

}  // namespace tensora
```

### `src/semantic/Contracts.cpp`
```cpp
// This file implements the operator contract checkers.
// Each function checks one Tensora operator (matmul, linear, conv2d,
// and so on) and returns its verified result type. SemanticAnalyzer
// calls these functions while it checks a graph.
#include "tensora/semantic/Contracts.h"

#include <algorithm>
#include <optional>
#include <sstream>

namespace tensora {

namespace {

const Type* fail(ContractCtx& ctx, DiagCode code, std::string message) {
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = code;
  d.span = ctx.span;
  d.message = std::move(message);
  ctx.diags.report(d);
  return ctx.types.invalid();
}

bool isInvalid(const Type* t) { return t->kind() == Type::Kind::Invalid; }

// Returns the operand as a TensorType. If it is not one, this reports
// SemaTypeMismatch and returns nullptr. It never reports on an already-
// Invalid operand, since that error was already reported once
// (poisoning).
const TensorType* requireTensor(const Type* t, std::string_view what, ContractCtx& ctx) {
  if (isInvalid(t)) return nullptr;
  if (t->kind() != Type::Kind::Tensor) {
    fail(ctx, DiagCode::SemaTypeMismatch,
         std::string(what) + " requires a tensor operand; found " + describeType(t, ctx.types));
    return nullptr;
  }
  return static_cast<const TensorType*>(t);
}

std::string axesToString(const std::vector<Axis>& axes, ContractCtx& ctx) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i) out << ", ";
    out << axes[i].name << "=" << describeDimension(axes[i].size, ctx.types);
  }
  out << "]";
  return out.str();
}

// RM 3.8's trailing-align broadcast rule. checkElementwise uses this
// on the full axis lists, and checkMatMul uses it on just the
// leading (batch) axes. On an incompatibility, this reports a
// diagnostic and returns nullopt.
std::optional<std::vector<Axis>> broadcastAxes(const std::vector<Axis>& a, const std::vector<Axis>& b,
                                                ContractCtx& ctx) {
  size_t ra = a.size(), rb = b.size();
  size_t n = std::max(ra, rb);
  std::vector<Axis> result(n);

  for (size_t i = 0; i < n; ++i) {  // i counts positions from the right: 0 = last axis
    bool hasA = i < ra, hasB = i < rb;
    const Axis* axA = hasA ? &a[ra - 1 - i] : nullptr;
    const Axis* axB = hasB ? &b[rb - 1 - i] : nullptr;

    if (axA && !axB) {
      result[n - 1 - i] = *axA;  // shorter side is implicitly padded (size 1, name from the longer side)
      continue;
    }
    if (axB && !axA) {
      result[n - 1 - i] = *axB;
      continue;
    }

    Dimension da = ctx.solver.resolve(axA->size);
    Dimension db = ctx.solver.resolve(axB->size);
    bool aIsOne = da.kind == Dimension::Kind::Const && da.constant == 1;
    bool bIsOne = db.kind == Dimension::Kind::Const && db.constant == 1;

    if (!aIsOne && !bIsOne && axA->name != axB->name) {
      fail(ctx, DiagCode::SemaAxisNameMismatch,
           "axis " + std::to_string(n - 1 - i) + " is '" + axA->name + "' on one operand but '" + axB->name +
               "' on the other, and neither side is a size-1 broadcast axis");
      return std::nullopt;
    }

    if (aIsOne) {
      result[n - 1 - i] = *axB;
    } else if (bIsOne) {
      result[n - 1 - i] = *axA;
    } else if (ctx.solver.unify(axA->size, axB->size)) {
      result[n - 1 - i] = *axA;
    } else {
      fail(ctx, DiagCode::SemaBroadcast,
           "incompatible sizes for axis '" + axA->name + "': " + describeDimension(ctx.solver.resolve(axA->size), ctx.types) +
               " != " + describeDimension(ctx.solver.resolve(axB->size), ctx.types));
      return std::nullopt;
    }
  }
  return result;
}

}  // namespace

// --- RM 3.7: matrix multiplication (@) ---
const Type* checkMatMul(const Type* aType, const Type* bType, ContractCtx& ctx) {
  if (isInvalid(aType) || isInvalid(bType)) return ctx.types.invalid();
  const TensorType* a = requireTensor(aType, "matmul (@)", ctx);
  const TensorType* b = requireTensor(bType, "matmul (@)", ctx);
  if (!a || !b) return ctx.types.invalid();

  if (a->rank() < 2 || b->rank() < 2) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "matmul (@) requires rank >= 2 on both operands; found rank " + std::to_string(a->rank()) +
                    " and rank " + std::to_string(b->rank()));
  }

  const Axis& leftM = a->axes()[a->rank() - 2];
  const Axis& leftK = a->axes()[a->rank() - 1];
  const Axis& rightK = b->axes()[b->rank() - 2];
  const Axis& rightN = b->axes()[b->rank() - 1];

  if (!ctx.solver.unify(leftK.size, rightK.size)) {
    return fail(ctx, DiagCode::SemaShapeMismatch,
                "cannot multiply " + describeType(a, ctx.types) + " by " + describeType(b, ctx.types) +
                    ": contraction dims " + describeDimension(ctx.solver.resolve(leftK.size), ctx.types) +
                    " != " + describeDimension(ctx.solver.resolve(rightK.size), ctx.types));
  }

  std::vector<Axis> aLead(a->axes().begin(), a->axes().end() - 2);
  std::vector<Axis> bLead(b->axes().begin(), b->axes().end() - 2);
  auto lead = broadcastAxes(aLead, bLead, ctx);
  if (!lead) return ctx.types.invalid();

  std::vector<Axis> result = std::move(*lead);
  result.push_back(leftM);
  result.push_back(rightN);
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.9: linear ---
const Type* checkLinear(const Type* xType, const Type* wType, const Type* bType, ContractCtx& ctx) {
  if (isInvalid(xType) || isInvalid(wType) || isInvalid(bType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "linear", ctx);
  const TensorType* w = requireTensor(wType, "linear", ctx);
  const TensorType* bias = requireTensor(bType, "linear", ctx);
  if (!x || !w || !bias) return ctx.types.invalid();

  if (x->rank() < 1) {
    return fail(ctx, DiagCode::SemaRankMismatch, "linear's input must have rank >= 1");
  }
  if (w->rank() != 2) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "linear's weight must be rank 2 (Tensor[in, out]); found " + describeType(w, ctx.types));
  }
  if (bias->rank() != 1) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "linear's bias must be rank 1 (Tensor[out]); found " + describeType(bias, ctx.types));
  }

  const Axis& xK = x->axes().back();
  const Axis& wIn = w->axes()[0];
  const Axis& wOut = w->axes()[1];
  const Axis& biasAxis = bias->axes()[0];

  if (!ctx.solver.unify(xK.size, wIn.size)) {
    return fail(ctx, DiagCode::SemaShapeMismatch,
                "linear: input's trailing dim " + describeDimension(ctx.solver.resolve(xK.size), ctx.types) +
                    " != weight's in dim " + describeDimension(ctx.solver.resolve(wIn.size), ctx.types));
  }
  if (biasAxis.name != wOut.name) {
    return fail(ctx, DiagCode::SemaAxisNameMismatch,
                "linear: bias axis '" + biasAxis.name + "' does not match weight's out axis '" + wOut.name + "'");
  }
  if (!ctx.solver.unify(biasAxis.size, wOut.size)) {
    return fail(ctx, DiagCode::SemaShapeMismatch,
                "linear: bias size " + describeDimension(ctx.solver.resolve(biasAxis.size), ctx.types) +
                    " != weight's out size " + describeDimension(ctx.solver.resolve(wOut.size), ctx.types));
  }

  std::vector<Axis> result(x->axes().begin(), x->axes().end() - 1);
  result.push_back(Axis{wOut.name, wOut.size});
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.8: elementwise arithmetic and broadcasting ---
const Type* checkElementwise(ast::BinaryExpr::Op /*op*/, const Type* aType, const Type* bOrScalarType,
                              ContractCtx& ctx) {
  if (isInvalid(aType) || isInvalid(bOrScalarType)) return ctx.types.invalid();
  const TensorType* a = requireTensor(aType, "elementwise operator", ctx);
  if (!a) return ctx.types.invalid();

  if (bOrScalarType->kind() == Type::Kind::Scalar) {
    return a;  // tensor-scalar: applies elementwise, no name/size constraint (RM 3.8)
  }
  const TensorType* b = requireTensor(bOrScalarType, "elementwise operator", ctx);
  if (!b) return ctx.types.invalid();

  auto result = broadcastAxes(a->axes(), b->axes(), ctx);
  if (!result) return ctx.types.invalid();
  return ctx.types.tensor(std::move(*result));
}

// --- RM 3.10: relu ---
const Type* checkRelu(const Type* xType, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "relu", ctx);
  if (!x) return ctx.types.invalid();
  return x;  // result type equals input type exactly
}

// --- RM 3.11: softmax ---
const Type* checkSoftmax(const Type* xType, std::string_view axis, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "softmax", ctx);
  if (!x) return ctx.types.invalid();

  bool found = std::any_of(x->axes().begin(), x->axes().end(),
                            [&](const Axis& ax) { return ax.name == axis; });
  if (!found) {
    return fail(ctx, DiagCode::SemaAxisNotFound,
                "softmax: axis '" + std::string(axis) + "' not found in " + describeType(x, ctx.types));
  }
  return x;  // result type equals input type exactly (RM 3.11)
}

// --- RM 3.12: reshape ---
const Type* checkReshape(const Type* xType, const std::vector<Axis>& target, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "reshape", ctx);
  if (!x) return ctx.types.invalid();

  std::vector<Dimension> srcDims, dstDims;
  for (const Axis& ax : x->axes()) srcDims.push_back(ax.size);
  for (const Axis& ax : target) dstDims.push_back(ax.size);

  ElementCountResult r = ctx.solver.sameElementCount(srcDims, dstDims);
  if (r.status == ElementCountResult::Status::Unresolvable) {
    return fail(ctx, DiagCode::SemaUnresolvableShape,
                "reshape: cannot statically resolve whether " + describeType(x, ctx.types) + " and " +
                    axesToString(target, ctx) + " have the same element count (symbolic factors do not match)");
  }
  if (r.status == ElementCountResult::Status::Mismatch) {
    return fail(ctx, DiagCode::SemaReshapeCount,
                "reshape: element count " + std::to_string(r.rhsConstProduct) + " != " +
                    std::to_string(r.lhsConstProduct) + " (source " + describeType(x, ctx.types) + " -> target " +
                    axesToString(target, ctx) + ")");
  }
  return ctx.types.tensor(target);
}

// --- RM 3.13: flatten ---
const Type* checkFlatten(const Type* xType, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "flatten", ctx);
  if (!x) return ctx.types.invalid();

  if (x->rank() < 2) {
    return fail(ctx, DiagCode::SemaRankMismatch, "flatten requires rank >= 2; found " + describeType(x, ctx.types));
  }

  int64_t product = 1;
  for (size_t i = 1; i < x->rank(); ++i) {
    Dimension d = ctx.solver.resolve(x->axes()[i].size);
    if (d.kind != Dimension::Kind::Const) {
      return fail(ctx, DiagCode::SemaUnresolvableShape,
                  "flatten: trailing axis '" + x->axes()[i].name + "' must have a concrete size; found symbolic '" +
                      describeDimension(d, ctx.types) + "'");
    }
    product *= d.constant;
  }

  std::vector<Axis> result;
  result.push_back(x->axes()[0]);
  result.push_back(Axis{"flat", Dimension::makeConst(product)});
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.14: transpose ---
const Type* checkTranspose(const Type* xType, const std::vector<uint32_t>& perm, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "transpose", ctx);
  if (!x) return ctx.types.invalid();

  size_t rank = x->rank();
  if (perm.size() != rank) {
    return fail(ctx, DiagCode::SemaTransposePerm,
                "transpose: perm has " + std::to_string(perm.size()) + " entries but the input has rank " +
                    std::to_string(rank));
  }
  std::vector<bool> seen(rank, false);
  for (uint32_t p : perm) {
    if (p >= rank) {
      return fail(ctx, DiagCode::SemaTransposePerm,
                  "transpose: perm index " + std::to_string(p) + " is out of range for rank " +
                      std::to_string(rank));
    }
    if (seen[p]) {
      return fail(ctx, DiagCode::SemaTransposePerm, "transpose: perm index " + std::to_string(p) + " repeated");
    }
    seen[p] = true;
  }

  std::vector<Axis> result;
  result.reserve(rank);
  for (uint32_t p : perm) result.push_back(x->axes()[p]);
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.15: sum / mean ---
const Type* checkReduce(OpKind sumOrMean, const Type* xType, std::string_view axis, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(
      xType, sumOrMean == OpKind::Sum ? "sum" : "mean", ctx);
  if (!x) return ctx.types.invalid();

  auto it = std::find_if(x->axes().begin(), x->axes().end(), [&](const Axis& ax) { return ax.name == axis; });
  if (it == x->axes().end()) {
    return fail(ctx, DiagCode::SemaAxisNotFound,
                std::string(sumOrMean == OpKind::Sum ? "sum" : "mean") + ": axis '" + std::string(axis) +
                    "' not found in " + describeType(x, ctx.types));
  }
  if (x->rank() == 1) {
    // RM 3.3 says a TensorType's axis list is never empty. Reducing
    // the only axis of a rank-1 tensor would need a rank-0 result. A
    // TensorType cannot represent that at all, not even as an empty
    // result.
    return fail(ctx, DiagCode::SemaRankMismatch,
                std::string(sumOrMean == OpKind::Sum ? "sum" : "mean") +
                    ": cannot reduce the only axis of a rank-1 tensor (result would have rank 0)");
  }

  std::vector<Axis> result;
  result.reserve(x->rank() - 1);
  for (auto axIt = x->axes().begin(); axIt != x->axes().end(); ++axIt) {
    if (axIt != it) result.push_back(*axIt);
  }
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.16: conv2d ---
const Type* checkConv2D(const Type* xType, const Type* wType, const Type* bType, int64_t stride, int64_t padding,
                         ContractCtx& ctx) {
  if (isInvalid(xType) || isInvalid(wType) || isInvalid(bType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "conv2d", ctx);
  const TensorType* w = requireTensor(wType, "conv2d", ctx);
  const TensorType* bias = requireTensor(bType, "conv2d", ctx);
  if (!x || !w || !bias) return ctx.types.invalid();

  if (x->rank() != 4) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "conv2d's input must be rank 4 (Tensor[batch,channel,height,width]); found " +
                    describeType(x, ctx.types));
  }
  if (w->rank() != 4) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "conv2d's weight must be rank 4 (Tensor[out_channel,in_channel,kh,kw]); found " +
                    describeType(w, ctx.types));
  }
  if (bias->rank() != 1) {
    return fail(ctx, DiagCode::SemaRankMismatch, "conv2d's bias must be rank 1; found " + describeType(bias, ctx.types));
  }

  const Axis& xBatch = x->axes()[0];
  const Axis& xChan = x->axes()[1];
  const Axis& xH = x->axes()[2];
  const Axis& xW = x->axes()[3];
  const Axis& wOutC = w->axes()[0];
  const Axis& wInC = w->axes()[1];
  const Axis& wKh = w->axes()[2];
  const Axis& wKw = w->axes()[3];

  if (!ctx.solver.unify(xChan.size, wInC.size)) {
    return fail(ctx, DiagCode::SemaConvChannel,
                "conv2d: input channel " + describeDimension(ctx.solver.resolve(xChan.size), ctx.types) +
                    " != weight in_channel " + describeDimension(ctx.solver.resolve(wInC.size), ctx.types));
  }
  if (!ctx.solver.unify(bias->axes()[0].size, wOutC.size)) {
    return fail(ctx, DiagCode::SemaShapeMismatch,
                "conv2d: bias size " + describeDimension(ctx.solver.resolve(bias->axes()[0].size), ctx.types) +
                    " != weight out_channel " + describeDimension(ctx.solver.resolve(wOutC.size), ctx.types));
  }

  Dimension h = ctx.solver.resolve(xH.size);
  Dimension wd = ctx.solver.resolve(xW.size);
  Dimension kh = ctx.solver.resolve(wKh.size);
  Dimension kw = ctx.solver.resolve(wKw.size);
  if (h.kind != Dimension::Kind::Const || wd.kind != Dimension::Kind::Const ||
      kh.kind != Dimension::Kind::Const || kw.kind != Dimension::Kind::Const) {
    return fail(ctx, DiagCode::SemaUnresolvableShape,
                "conv2d: spatial arithmetic requires concrete height/width/kernel sizes; found a symbolic dimension");
  }
  if (stride <= 0) {
    return fail(ctx, DiagCode::SemaConvArithmetic, "conv2d: stride must be a positive integer; found " +
                                                        std::to_string(stride));
  }

  int64_t hNum = h.constant + 2 * padding - kh.constant;
  int64_t wNum = wd.constant + 2 * padding - kw.constant;
  if (hNum < 0 || hNum % stride != 0 || wNum < 0 || wNum % stride != 0) {
    return fail(ctx, DiagCode::SemaConvArithmetic,
                "conv2d: non-dividing spatial arithmetic for height=" + std::to_string(h.constant) +
                    ", width=" + std::to_string(wd.constant) + ", kernel=(" + std::to_string(kh.constant) + "," +
                    std::to_string(kw.constant) + "), stride=" + std::to_string(stride) +
                    ", padding=" + std::to_string(padding));
  }
  int64_t hout = hNum / stride + 1;
  int64_t wout = wNum / stride + 1;

  // RM 3.16's schema reuses the *input's* axis name at each position
  // (batch=B on both sides, channel=Cin -> channel=Cout, and so on).
  // The weight only supplies the new Cout *size*, not a new axis
  // name.
  std::vector<Axis> result = {
      Axis{xBatch.name, xBatch.size},
      Axis{xChan.name, wOutC.size},
      Axis{xH.name, Dimension::makeConst(hout)},
      Axis{xW.name, Dimension::makeConst(wout)},
  };
  return ctx.types.tensor(std::move(result));
}

// --- RM 3.17: maxpool ---
const Type* checkMaxPool(const Type* xType, int64_t size, int64_t stride, ContractCtx& ctx) {
  if (isInvalid(xType)) return ctx.types.invalid();
  const TensorType* x = requireTensor(xType, "maxpool", ctx);
  if (!x) return ctx.types.invalid();

  if (x->rank() != 4) {
    return fail(ctx, DiagCode::SemaRankMismatch,
                "maxpool's input must be rank 4 (Tensor[batch,channel,height,width]); found " +
                    describeType(x, ctx.types));
  }

  const Axis& xBatch = x->axes()[0];
  const Axis& xChan = x->axes()[1];
  const Axis& xH = x->axes()[2];
  const Axis& xW = x->axes()[3];

  Dimension h = ctx.solver.resolve(xH.size);
  Dimension wd = ctx.solver.resolve(xW.size);
  if (h.kind != Dimension::Kind::Const || wd.kind != Dimension::Kind::Const) {
    return fail(ctx, DiagCode::SemaUnresolvableShape,
                "maxpool: spatial arithmetic requires concrete height/width; found a symbolic dimension");
  }
  if (stride <= 0) {
    return fail(ctx, DiagCode::SemaConvArithmetic, "maxpool: stride must be a positive integer; found " +
                                                        std::to_string(stride));
  }

  int64_t hNum = h.constant - size;   // maxpool == conv2d's formula with padding=0
  int64_t wNum = wd.constant - size;
  if (hNum < 0 || hNum % stride != 0 || wNum < 0 || wNum % stride != 0) {
    return fail(ctx, DiagCode::SemaConvArithmetic,
                "maxpool: non-dividing spatial arithmetic for height=" + std::to_string(h.constant) +
                    ", width=" + std::to_string(wd.constant) + ", size=" + std::to_string(size) +
                    ", stride=" + std::to_string(stride));
  }
  int64_t hout = hNum / stride + 1;
  int64_t wout = wNum / stride + 1;

  std::vector<Axis> result = {
      Axis{xBatch.name, xBatch.size},
      Axis{xChan.name, xChan.size},
      Axis{xH.name, Dimension::makeConst(hout)},
      Axis{xW.name, Dimension::makeConst(wout)},
  };
  return ctx.types.tensor(std::move(result));
}

}  // namespace tensora
```

### `include/tensora/semantic/SemanticAnalyzer.h`
```cpp
#pragma once
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "tensora/shared/Ast.h"
#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/Ir.h"
#include "tensora/shared/Result.h"
#include "tensora/semantic/Symbol.h"
#include "tensora/shared/DimSolver.h"
#include "tensora/shared/Type.h"

// Drives semantic analysis (ARCHITECTURE.md §6). It resolves names,
// infers types and shapes through the operator contracts
// (semantic/Contracts.h), and builds a Typed Computation Graph. It
// never depends on MLIR or LLVM.
namespace tensora {

// A declaration-order snapshot of the entries SymbolTable accepted,
// captured while `analyze()` runs (--emit-symbols in main.cpp renders
// this; see SymbolTable::orderedGlobals/orderedLocals for why this is a
// snapshot of the real resolution state, not a second, separate
// rebuild of it). There is one GraphScope per graph, in program order.
// `locals` is that graph's final local scope: a rebind that shadows an
// earlier local updates the same entry in place, so only the last
// value stays, matching what name resolution itself would see at the
// end of that graph.
struct SymbolTableReport {
  std::vector<Symbol> globals;
  struct GraphScope {
    std::string graphName;
    std::vector<Symbol> locals;
    // One entry per distinct name referenced in this graph's body, in
    // first-reference order: (referenced name, "global::name" /
    // "local::name" / "undefined"). This is a real log of what
    // SymbolTable::resolve() returned for that reference, not a
    // recomputed value.
    std::vector<std::pair<std::string, std::string>> resolutions;
  };
  std::vector<GraphScope> graphs;
};

class SemanticAnalyzer {
 public:
  SemanticAnalyzer(TypeContext& types, DiagnosticEngine& diags) : types_(types), diags_(diags) {}

  Result<TypedProgram> analyze(const ast::AstProgram& program);

  // Valid after analyze() returns. It reflects exactly the symbols
  // `analyze()` inserted into `symbols_` (it skips rejected
  // redeclarations, since those never entered the table).
  const SymbolTableReport& symbolTable() const { return symbolReport_; }

 private:
  // Per-graph state: a fresh GraphBuilder and DimSolver for each graph
  // (RM 3.4 scopes dimension reasoning to "within one graph"), plus the
  // name -> ValueId maps for graph-local bindings and lazily-added
  // global params.
  struct GraphCtx {
    GraphBuilder builder;
    DimSolver solver;
    std::unordered_map<std::string, ValueId> localValues;
    std::unordered_map<std::string, ValueId> globalValues;
    // First-reference-order resolution log for this graph (see
    // SymbolTableReport::GraphScope::resolutions). loggedNames removes
    // duplicates, so a name used on several lines gets one entry, not
    // one per use.
    std::vector<std::pair<std::string, std::string>> resolutionLog;
    std::unordered_set<std::string> loggedNames;
  };

  TypedGraph analyzeGraph(const ast::GraphDecl& g);

  // Resolves a plain identifier operand. If it is undefined, reports
  // SemaUndefinedName and returns an InvalidType poison value. Adds a
  // global input/weight as this graph's param the first time the graph
  // references it.
  std::pair<ValueId, const Type*> resolveOperand(GraphCtx& gc, const std::string& name, SourceSpan span);
  // RM 3.24 `operand ::= identifier | floatLit`.
  std::pair<ValueId, const Type*> resolveExprOperand(GraphCtx& gc, const ast::Expr& e);

  std::pair<ValueId, const Type*> analyzeCall(GraphCtx& gc, const ast::CallExpr& call);
  std::pair<ValueId, const Type*> analyzeMatMul(GraphCtx& gc, const ast::MatMulExpr& mm);
  std::pair<ValueId, const Type*> analyzeBinary(GraphCtx& gc, const ast::BinaryExpr& bin);

  TypeContext& types_;
  DiagnosticEngine& diags_;
  SymbolTable symbols_;
  SymbolTableReport symbolReport_;
};

}  // namespace tensora
```

### `src/semantic/SemanticAnalyzer.cpp`
```cpp
// This file implements semantic analysis.
// It checks the AST for name, scope, type, and shape errors, and
// creates the verified Typed Computation Graph (TCG). Optimization
// reads the TCG next.
#include "tensora/semantic/SemanticAnalyzer.h"

#include <initializer_list>
#include <string_view>
#include <unordered_set>

#include "tensora/semantic/Contracts.h"

namespace tensora {

namespace {

void reportSimple(DiagnosticEngine& diags, DiagCode code, SourceSpan span, std::string msg) {
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = code;
  d.span = span;
  d.message = std::move(msg);
  diags.report(d);
}

// Checks the two tensor-type rules RM 3.3 requires of any axis list:
// every concrete axis size must be positive, and axis names must be
// unique within the list. This reports one diagnostic per violation,
// and returns false so the caller can poison the result. This never
// checks a symbolic axis size for a positive value here, since its
// concrete value, if it has one, is not known until DimSolver
// resolves it later.
bool validateAxes(const std::vector<Axis>& axes, DiagnosticEngine& diags, SourceSpan span) {
  bool ok = true;
  std::unordered_set<std::string> seen;
  for (const Axis& ax : axes) {
    if (ax.size.kind == Dimension::Kind::Const && ax.size.constant <= 0) {
      reportSimple(diags, DiagCode::SemaNonPositiveDimension, span,
                   "axis '" + ax.name + "' has size " + std::to_string(ax.size.constant) +
                       "; RM 3.3 requires a concrete positive integer");
      ok = false;
    }
    if (!seen.insert(ax.name).second) {
      reportSimple(diags, DiagCode::SemaDuplicateAxisName, span,
                   "duplicate axis name '" + ax.name + "' within one tensor type");
      ok = false;
    }
  }
  return ok;
}

const Type* convertTensorType(const ast::TensorTypeSyntax& syn, TypeContext& types, DiagnosticEngine& diags) {
  std::vector<Axis> axes;
  axes.reserve(syn.axes.size());
  for (const ast::AxisSyntax& a : syn.axes) {
    Dimension d = a.size.k == ast::DimSyntax::K::Const ? Dimension::makeConst(a.size.constant)
                                                        : Dimension::makeSymbol(types.internSymbol(a.size.symbol));
    axes.push_back(Axis{a.name, d});
  }
  if (!validateAxes(axes, diags, syn.span)) return types.invalid();
  return types.tensor(std::move(axes));
}

std::vector<Axis> convertAxisList(const std::vector<ast::AxisSyntax>& syn, TypeContext& types) {
  std::vector<Axis> axes;
  axes.reserve(syn.size());
  for (const ast::AxisSyntax& a : syn) {
    Dimension d = a.size.k == ast::DimSyntax::K::Const ? Dimension::makeConst(a.size.constant)
                                                        : Dimension::makeSymbol(types.internSymbol(a.size.symbol));
    axes.push_back(Axis{a.name, d});
  }
  return axes;
}

// Checks that `args`'s first `positionalCount` entries are all
// unkeyworded Kind::Name (plain identifiers), and that no more
// positional entries follow. Every call-style operator uses this,
// except reshape, whose second argument is a positional axis list,
// not a name.
bool takePositionalNames(const std::vector<ast::Arg>& args, size_t positionalCount, std::string_view opName,
                         DiagnosticEngine& diags, SourceSpan span, std::vector<const ast::Arg*>& positional) {
  size_t have = 0;
  while (have < args.size() && !args[have].keyword.has_value()) ++have;
  if (have != positionalCount) {
    reportSimple(diags, DiagCode::SemaArgCount, span,
                 std::string(opName) + " expects " + std::to_string(positionalCount) +
                     " positional argument(s); found " + std::to_string(have));
    return false;
  }
  for (size_t i = 0; i < have; ++i) {
    if (args[i].kind != ast::Arg::Kind::Name) {
      reportSimple(diags, DiagCode::SemaArgKind, span,
                   std::string(opName) + "'s positional argument " + std::to_string(i) +
                       " must be a plain identifier");
      return false;
    }
    positional.push_back(&args[i]);
  }
  return true;
}

// Finds a keyword argument by name; order does not matter. Reports
// SemaArgCount if `required` is true and the argument is absent, or
// SemaArgKind if it is present with the wrong Arg::Kind. If the
// caller supplies `found`, this function sets it either way, correct
// kind or not. This lets an optional keyword's caller tell "absent"
// apart from "present but invalid" (the invalid case is already
// reported).
const ast::Arg* findKeyword(const std::vector<ast::Arg>& args, std::string_view key, ast::Arg::Kind expectedKind,
                            bool required, std::string_view opName, DiagnosticEngine& diags, SourceSpan span,
                            bool* found = nullptr) {
  for (const ast::Arg& a : args) {
    if (a.keyword.has_value() && *a.keyword == key) {
      if (found) *found = true;
      if (a.kind != expectedKind) {
        reportSimple(diags, DiagCode::SemaArgKind, span,
                     std::string(opName) + "'s '" + std::string(key) + "' argument has the wrong kind");
        return nullptr;
      }
      return &a;
    }
  }
  if (found) *found = false;
  if (required) {
    reportSimple(diags, DiagCode::SemaArgCount, span,
                 std::string(opName) + " requires a '" + std::string(key) + "' keyword argument");
  }
  return nullptr;
}

bool rejectUnknownKeywords(const std::vector<ast::Arg>& args, std::initializer_list<std::string_view> allowed,
                           std::string_view opName, DiagnosticEngine& diags, SourceSpan span) {
  for (const ast::Arg& a : args) {
    if (!a.keyword.has_value()) continue;
    bool ok = false;
    for (std::string_view k : allowed) {
      if (*a.keyword == k) {
        ok = true;
        break;
      }
    }
    if (!ok) {
      reportSimple(diags, DiagCode::SemaArgKind, span,
                   std::string(opName) + ": unknown keyword argument '" + *a.keyword + "'");
      return false;
    }
  }
  return true;
}

}  // namespace

std::pair<ValueId, const Type*> SemanticAnalyzer::resolveOperand(GraphCtx& gc, const std::string& name,
                                                                  SourceSpan span) {
  const Symbol* sym = symbols_.resolve(name);
  if (gc.loggedNames.insert(name).second) {
    std::string resolvedAs = !sym                                  ? "undefined"
                              : sym->kind == Symbol::Kind::Value ? "local::" + name
                                                                  : "global::" + name;
    gc.resolutionLog.emplace_back(name, std::move(resolvedAs));
  }
  if (!sym) {
    reportSimple(diags_, DiagCode::SemaUndefinedName, span, "use of undefined name '" + name + "'");
    return {0, types_.invalid()};
  }
  if (sym->kind == Symbol::Kind::Value) {
    auto it = gc.localValues.find(name);
    return {it != gc.localValues.end() ? it->second : 0, sym->type};
  }
  auto git = gc.globalValues.find(name);
  if (git != gc.globalValues.end()) return {git->second, sym->type};
  if (sym->type->kind() == Type::Kind::Invalid) {
    // The declaration itself failed RM 3.3 validation. That error was
    // already reported at the declaration site, so this use is
    // poisoned, not a new error.
    return {0, sym->type};
  }
  const auto* tt = static_cast<const TensorType*>(sym->type);  // globals are always Tensor once past the invalid check
  ValueId vid = sym->kind == Symbol::Kind::Input ? gc.builder.addInput(name, tt, sym->span)
                                                  : gc.builder.addWeight(name, tt, sym->span);
  gc.globalValues.emplace(name, vid);
  return {vid, sym->type};
}

std::pair<ValueId, const Type*> SemanticAnalyzer::resolveExprOperand(GraphCtx& gc, const ast::Expr& e) {
  if (const auto* name = dynamic_cast<const ast::NameExpr*>(&e)) {
    return resolveOperand(gc, name->name, name->span);
  }
  if (dynamic_cast<const ast::ScalarExpr*>(&e)) {
    return {0, types_.scalar()};
  }
  return {0, types_.invalid()};  // unreachable given RM 3.24's `operand ::= identifier | floatLit`
}

std::pair<ValueId, const Type*> SemanticAnalyzer::analyzeMatMul(GraphCtx& gc, const ast::MatMulExpr& mm) {
  const auto* lhsName = static_cast<const ast::NameExpr*>(mm.lhs.get());  // grammar: lhs is always a NameExpr
  auto [lhsId, lhsType] = resolveOperand(gc, lhsName->name, lhsName->span);
  auto [rhsId, rhsType] = resolveExprOperand(gc, *mm.rhs);

  ContractCtx ctx{types_, gc.solver, diags_, mm.span};
  const Type* result = checkMatMul(lhsType, rhsType, ctx);
  if (result->kind() == Type::Kind::Invalid) return {0, result};

  ValueId out =
      gc.builder.emit(OpKind::MatMul, {lhsId, rhsId}, {}, static_cast<const TensorType*>(result), mm.span, "");
  return {out, result};
}

std::pair<ValueId, const Type*> SemanticAnalyzer::analyzeBinary(GraphCtx& gc, const ast::BinaryExpr& bin) {
  const auto* lhsName = static_cast<const ast::NameExpr*>(bin.lhs.get());
  auto [lhsId, lhsType] = resolveOperand(gc, lhsName->name, lhsName->span);
  auto [rhsId, rhsType] = resolveExprOperand(gc, *bin.rhs);

  ContractCtx ctx{types_, gc.solver, diags_, bin.span};
  const Type* result = checkElementwise(bin.op, lhsType, rhsType, ctx);
  if (result->kind() == Type::Kind::Invalid) return {0, result};

  NodeAttrs attrs;
  std::vector<ValueId> inputs = {lhsId};
  if (rhsType->kind() == Type::Kind::Scalar) {
    attrs.scalar = static_cast<const ast::ScalarExpr*>(bin.rhs.get())->value;
  } else {
    inputs.push_back(rhsId);
  }
  OpKind op = bin.op == ast::BinaryExpr::Op::Add   ? OpKind::Add
              : bin.op == ast::BinaryExpr::Op::Sub ? OpKind::Sub
              : bin.op == ast::BinaryExpr::Op::Mul ? OpKind::Mul
                                                    : OpKind::Div;
  ValueId out = gc.builder.emit(op, std::move(inputs), attrs, static_cast<const TensorType*>(result), bin.span, "");
  return {out, result};
}

std::pair<ValueId, const Type*> SemanticAnalyzer::analyzeCall(GraphCtx& gc, const ast::CallExpr& call) {
  const std::string& op = call.callee;
  const SourceSpan span = call.span;
  const std::vector<ast::Arg>& args = call.args;
  auto poison = [&]() -> std::pair<ValueId, const Type*> { return {0, types_.invalid()}; };

  if (op == "linear") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 3, "linear", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {}, "linear", diags_, span)) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    auto [wId, wType] = resolveOperand(gc, pos[1]->name, pos[1]->span);
    auto [bId, bType] = resolveOperand(gc, pos[2]->name, pos[2]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkLinear(xType, wType, bType, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    ValueId out =
        gc.builder.emit(OpKind::Linear, {xId, wId, bId}, {}, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "relu") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "relu", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {}, "relu", diags_, span)) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkRelu(xType, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    ValueId out = gc.builder.emit(OpKind::Relu, {xId}, {}, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "softmax") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "softmax", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"axis"}, "softmax", diags_, span)) return poison();
    const ast::Arg* axisArg =
        findKeyword(args, "axis", ast::Arg::Kind::Name, true, "softmax", diags_, span);
    if (!axisArg) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkSoftmax(xType, axisArg->name, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.axisName = axisArg->name;
    ValueId out = gc.builder.emit(OpKind::Softmax, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "reshape") {
    if (args.size() != 2) {
      reportSimple(diags_, DiagCode::SemaArgCount, span,
                   "reshape expects exactly 2 arguments (identifier, axis-list); found " +
                       std::to_string(args.size()));
      return poison();
    }
    if (args[0].keyword.has_value() || args[0].kind != ast::Arg::Kind::Name) {
      reportSimple(diags_, DiagCode::SemaArgKind, span, "reshape's first argument must be a plain identifier");
      return poison();
    }
    if (args[1].keyword.has_value() || args[1].kind != ast::Arg::Kind::AxisList) {
      reportSimple(diags_, DiagCode::SemaArgKind, span,
                   "reshape's second argument must be an axis list [name=size, ...]");
      return poison();
    }
    auto [xId, xType] = resolveOperand(gc, args[0].name, args[0].span);
    std::vector<Axis> target = convertAxisList(args[1].axisList, types_);
    if (!validateAxes(target, diags_, span)) return poison();
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkReshape(xType, target, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.reshapeTarget = target;
    ValueId out = gc.builder.emit(OpKind::Reshape, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "flatten") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "flatten", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {}, "flatten", diags_, span)) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkFlatten(xType, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    ValueId out = gc.builder.emit(OpKind::Flatten, {xId}, {}, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "transpose") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "transpose", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"perm"}, "transpose", diags_, span)) return poison();
    const ast::Arg* permArg =
        findKeyword(args, "perm", ast::Arg::Kind::IntList, true, "transpose", diags_, span);
    if (!permArg) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    std::vector<uint32_t> perm;
    perm.reserve(permArg->intList.size());
    for (int64_t v : permArg->intList) perm.push_back(static_cast<uint32_t>(v));
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkTranspose(xType, perm, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.perm = perm;
    ValueId out = gc.builder.emit(OpKind::Transpose, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "sum" || op == "mean") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, op, diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"axis"}, op, diags_, span)) return poison();
    const ast::Arg* axisArg = findKeyword(args, "axis", ast::Arg::Kind::Name, true, op, diags_, span);
    if (!axisArg) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    OpKind kind = op == "sum" ? OpKind::Sum : OpKind::Mean;
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkReduce(kind, xType, axisArg->name, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.axisName = axisArg->name;
    ValueId out = gc.builder.emit(kind, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "conv2d") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 3, "conv2d", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"stride", "padding"}, "conv2d", diags_, span)) return poison();
    const ast::Arg* strideArg =
        findKeyword(args, "stride", ast::Arg::Kind::Int, true, "conv2d", diags_, span);
    const ast::Arg* paddingArg =
        findKeyword(args, "padding", ast::Arg::Kind::Int, true, "conv2d", diags_, span);
    if (!strideArg || !paddingArg) return poison();
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    auto [wId, wType] = resolveOperand(gc, pos[1]->name, pos[1]->span);
    auto [bId, bType] = resolveOperand(gc, pos[2]->name, pos[2]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkConv2D(xType, wType, bType, strideArg->intVal, paddingArg->intVal, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.stride = strideArg->intVal;
    attrs.padding = paddingArg->intVal;
    ValueId out =
        gc.builder.emit(OpKind::Conv2D, {xId, wId, bId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }
  if (op == "maxpool") {
    std::vector<const ast::Arg*> pos;
    if (!takePositionalNames(args, 1, "maxpool", diags_, span, pos)) return poison();
    if (!rejectUnknownKeywords(args, {"size", "stride"}, "maxpool", diags_, span)) return poison();
    const ast::Arg* sizeArg = findKeyword(args, "size", ast::Arg::Kind::Int, true, "maxpool", diags_, span);
    if (!sizeArg) return poison();
    bool strideFound = false;
    const ast::Arg* strideArg =
        findKeyword(args, "stride", ast::Arg::Kind::Int, false, "maxpool", diags_, span, &strideFound);
    if (strideFound && !strideArg) return poison();  // present but wrong kind (already reported)
    int64_t stride = strideArg ? strideArg->intVal : sizeArg->intVal;  // RM 3.17: defaults to size
    auto [xId, xType] = resolveOperand(gc, pos[0]->name, pos[0]->span);
    ContractCtx ctx{types_, gc.solver, diags_, span};
    const Type* result = checkMaxPool(xType, sizeArg->intVal, stride, ctx);
    if (result->kind() == Type::Kind::Invalid) return {0, result};
    NodeAttrs attrs;
    attrs.poolSize = sizeArg->intVal;
    attrs.stride = stride;
    ValueId out = gc.builder.emit(OpKind::MaxPool, {xId}, attrs, static_cast<const TensorType*>(result), span, "");
    return {out, result};
  }

  reportSimple(diags_, DiagCode::SemaUndefinedName, span, "undefined operator '" + op + "'");
  return poison();
}

TypedGraph SemanticAnalyzer::analyzeGraph(const ast::GraphDecl& g) {
  symbols_.enterGraph();
  GraphCtx gc{GraphBuilder(g.name), DimSolver(), {}, {}, {}, {}};
  const size_t diagsBeforeGraph = diags_.all().size();

  for (const ast::Stmt& stmt : g.stmts) {
    ValueId resultValue = 0;
    const Type* resultType = types_.invalid();

    if (const auto* call = dynamic_cast<const ast::CallExpr*>(stmt.rhs.get())) {
      auto r = analyzeCall(gc, *call);
      resultValue = r.first;
      resultType = r.second;
    } else if (const auto* mm = dynamic_cast<const ast::MatMulExpr*>(stmt.rhs.get())) {
      auto r = analyzeMatMul(gc, *mm);
      resultValue = r.first;
      resultType = r.second;
    } else if (const auto* bin = dynamic_cast<const ast::BinaryExpr*>(stmt.rhs.get())) {
      auto r = analyzeBinary(gc, *bin);
      resultValue = r.first;
      resultType = r.second;
    } else if (const auto* name = dynamic_cast<const ast::NameExpr*>(stmt.rhs.get())) {
      auto r = resolveOperand(gc, name->name, name->span);
      resultValue = r.first;
      resultType = r.second;
    }

    if (resultType->kind() != Type::Kind::Invalid) gc.builder.setDebugName(resultValue, stmt.name);

    Symbol sym;
    sym.name = stmt.name;
    sym.kind = Symbol::Kind::Value;
    sym.type = resultType;
    sym.span = stmt.span;
    SymbolTable::BindOutcome outcome = symbols_.bindLocal(stmt.name, sym);
    if (outcome == SymbolTable::BindOutcome::KindConflict) {
      reportSimple(diags_, DiagCode::SemaRedeclaration, stmt.span,
                   "'" + stmt.name + "' is already declared as a global input/weight; a graph cannot rebind it");
    } else {
      gc.localValues[stmt.name] = resultValue;
    }
  }

  auto [retId, retType] = resolveOperand(gc, g.returnName, g.span);
  gc.builder.setOutput(retType->kind() != Type::Kind::Invalid ? retId : 0);

  SymbolTableReport::GraphScope scope;
  scope.graphName = g.name;
  for (const Symbol* s : symbols_.orderedLocals()) scope.locals.push_back(*s);
  scope.resolutions = std::move(gc.resolutionLog);
  symbolReport_.graphs.push_back(std::move(scope));

  TypedGraph graph = gc.builder.finish();
  // GraphVerifier catches bugs from a broken pass or a bad hand-built
  // fixture (IR.md §6), on a graph that is otherwise clean. It should
  // not add a second, redundant diagnostic to a graph semantic
  // analysis already knows is broken (for example, `return` of an
  // undefined name leaves an empty graph with no output, which is
  // *expected*, since the error was already reported).
  if (diags_.all().size() == diagsBeforeGraph) {
    GraphVerifier verifier;
    for (Diagnostic& d : verifier.verify(graph)) diags_.report(d);
  }
  return graph;
}

Result<TypedProgram> SemanticAnalyzer::analyze(const ast::AstProgram& program) {
  TypedProgram output(types_);

  for (const ast::Decl& decl : program.decls) {
    const Type* type = convertTensorType(decl.type, types_, diags_);
    Symbol sym;
    sym.name = decl.name;
    sym.kind = decl.kind == ast::Decl::Kind::Input ? Symbol::Kind::Input : Symbol::Kind::Weight;
    sym.type = type;
    sym.span = decl.span;
    if (!symbols_.declareGlobal(sym)) {
      reportSimple(diags_, DiagCode::SemaRedeclaration, decl.span,
                   "redeclaration of top-level name '" + decl.name + "'");
    }
  }

  for (const Symbol* s : symbols_.orderedGlobals()) symbolReport_.globals.push_back(*s);

  std::unordered_set<std::string> graphNames;
  for (const ast::GraphDecl& g : program.graphs) {
    if (!graphNames.insert(g.name).second) {
      reportSimple(diags_, DiagCode::SemaRedeclaration, g.span, "duplicate graph name '" + g.name + "'");
      continue;
    }
    output.addGraph(analyzeGraph(g));
  }

  bool ok = !diags_.hasErrors();
  return Result<TypedProgram>::make(std::move(output), ok, diags_.all());
}

}  // namespace tensora
```

## 10.5 Typed Graph Listing (node/value definitions, builder, verifier)

### `include/tensora/shared/Ir.h`
```cpp
#pragma once
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/SourceSpan.h"
#include "tensora/shared/Type.h"

// The Typed Computation Graph (TCG), RM 3.20; IR.md. This is the
// boundary between semantic analysis and the backend: a verified, SSA,
// arena-owned graph. Later phases (passes and lowering) read this
// graph without re-checking its semantics. See IR.md for the full
// reasoning.
namespace tensora {

using ValueId = uint32_t;
using NodeId = uint32_t;
constexpr NodeId kNoProducer = static_cast<NodeId>(-1);

struct Value {
  ValueId id = 0;
  const Type* type = nullptr;      // verified; TensorType for every graph value in v1
  NodeId producer = kNoProducer;   // producing node, or kNoProducer for a parameter
  SourceSpan span;                 // where it was declared/bound in the source
  std::string debugName;           // original source name (best effort, non-unique)

  enum class Origin { Input, Weight, Internal };
  Origin origin = Origin::Internal;
};

enum class OpKind {
  Add, Sub, Mul, Div,
  MatMul, Linear,
  Relu, Softmax,
  Reshape, Flatten, Transpose,
  Sum, Mean,
  Conv2D, MaxPool
};

const char* opKindName(OpKind op);

// Free-form facts kept for later passes and for `explain` (whitepaper
// §6.1) -- for example "symbolic:batch". A pass only ever adds facts,
// never removes them. This class is deliberately simple: a set of
// strings.
class FactSet {
 public:
  void add(std::string fact) { facts_.insert(std::move(fact)); }
  bool has(std::string_view fact) const { return facts_.count(std::string(fact)) != 0; }
  const std::set<std::string>& all() const { return facts_; }

 private:
  std::set<std::string> facts_;
};

// Operator attributes; only the fields relevant to a given OpKind are used.
struct NodeAttrs {
  std::optional<double> scalar;               // Mul/Div: e.g. `scores * 0.125`
  std::optional<std::string> axisName;        // Softmax/Sum/Mean: axis referenced by name
  std::vector<uint32_t> perm;                 // Transpose: permutation of 0..rank-1
  std::vector<Axis> reshapeTarget;             // Reshape: target axes (names + sizes)
  int64_t stride = 0, padding = 0, poolSize = 0;  // Conv2D: stride/padding; MaxPool: size (stride reuses `stride`)
};

struct GraphNode {
  NodeId id = 0;
  OpKind op = OpKind::Add;
  std::vector<ValueId> inputs;   // operand order matches the surface operator
  ValueId output = 0;            // single result value (v1: no multi-result ops)
  NodeAttrs attrs;
  SourceSpan span;
  FactSet facts;
};

class GraphBuilder;

class TypedGraph {
 public:
  std::string name;
  std::vector<ValueId> params;  // inputs + weights, declaration order
  ValueId output = 0;           // the single returned value

  const std::vector<Value>& values() const { return values_; }
  const std::vector<GraphNode>& nodes() const { return nodes_; }

  const Value& value(ValueId id) const { return values_.at(id); }
  const GraphNode& node(NodeId id) const { return nodes_.at(id); }

  NodeId producerOf(ValueId id) const { return values_.at(id).producer; }
  const std::vector<NodeId>& consumersOf(ValueId id) const { return consumers_.at(id); }

  // Adds a proven property to an already-built node (IR.md §4: a place
  // to record proven properties without re-deriving them). A pass uses
  // this, for example, to mark a fusion group it found, without
  // changing the graph's structure. This never changes SSA order,
  // topological order, or any other GraphVerifier invariant.
  void addFact(NodeId id, std::string fact) { nodes_.at(id).facts.add(std::move(fact)); }

 private:
  friend class GraphBuilder;
  std::vector<Value> values_;
  std::vector<GraphNode> nodes_;
  std::vector<std::vector<NodeId>> consumers_;  // indexed by ValueId
};

class TypedProgram {
 public:
  explicit TypedProgram(TypeContext& types) : types_(types) {}

  TypeContext& types() { return types_; }
  const std::vector<TypedGraph>& graphs() const { return graphs_; }
  const TypedGraph* graph(std::string_view name) const {
    for (const TypedGraph& g : graphs_) {
      if (g.name == name) return &g;
    }
    return nullptr;
  }
  void addGraph(TypedGraph g) { graphs_.push_back(std::move(g)); }

 private:
  TypeContext& types_;
  std::vector<TypedGraph> graphs_;
};

// The one allowed way to create nodes and values (ARCH-008). It keeps
// SSA order, topological order, and the producer/consumer index
// correct. Semantic analysis uses it, and so do test fixtures and
// downstream mocks.
class GraphBuilder {
 public:
  explicit GraphBuilder(std::string graphName) { graph_.name = std::move(graphName); }

  ValueId addInput(std::string name, const TensorType* type, SourceSpan span) {
    return addParam(std::move(name), type, span, Value::Origin::Input);
  }
  ValueId addWeight(std::string name, const TensorType* type, SourceSpan span) {
    return addParam(std::move(name), type, span, Value::Origin::Weight);
  }

  // Adds a node and returns its one output value. The caller supplies
  // the result type (semantic analysis found it through the operator
  // contract).
  //
  // A correct caller (SemanticAnalyzer) only ever passes ValueIds it
  // already got from this same builder, so `inputs` is always in
  // range. This function does *not* throw or assert on an out-of-range
  // or forward-referencing ValueId. Instead, it bounds-checks the
  // consumer-index update rather than using .at(). This lets
  // GraphVerifier do its real job: catch bugs from a broken pass or a
  // bad hand-built fixture (IR.md §6). Without this, a hand-built test
  // could not build a dangling or forward-referencing operand to
  // check that GraphVerifier catches it.
  ValueId emit(OpKind op, std::vector<ValueId> inputs, NodeAttrs attrs, const TensorType* resultType,
               SourceSpan span, std::string debugName) {
    ValueId out = pushValue(resultType, span, std::move(debugName), Value::Origin::Internal);
    NodeId nid = static_cast<NodeId>(graph_.nodes_.size());
    GraphNode node;
    node.id = nid;
    node.op = op;
    node.inputs = std::move(inputs);
    node.output = out;
    node.attrs = std::move(attrs);
    node.span = span;
    for (ValueId in : node.inputs) {
      if (in < graph_.consumers_.size()) graph_.consumers_[in].push_back(nid);
    }
    graph_.nodes_.push_back(std::move(node));
    graph_.values_[out].producer = nid;
    return out;
  }

  void setOutput(ValueId v) { graph_.output = v; }

  // Sets a cosmetic `debugName`, "best effort, non-unique" as Value's
  // own comment says. Diagnostics and driver dumps (--emit-graph,
  // --explain) use this name. SSA, verification, and lowering never
  // read it. This lets a caller, such as SemanticAnalyzer, attach the
  // source-level name after the fact, once it knows it (SemanticAnalyzer
  // only learns the statement's bound name after `emit` already picked
  // the OpKind-specific helper and returned). This avoids passing that
  // name through every call site.
  void setDebugName(ValueId id, std::string name) { graph_.values_.at(id).debugName = std::move(name); }

  TypedGraph finish() { return std::move(graph_); }

 private:
  ValueId addParam(std::string name, const TensorType* type, SourceSpan span, Value::Origin origin) {
    ValueId id = pushValue(type, span, std::move(name), origin);
    graph_.params.push_back(id);
    return id;
  }
  ValueId pushValue(const Type* type, SourceSpan span, std::string debugName, Value::Origin origin) {
    ValueId id = static_cast<ValueId>(graph_.values_.size());
    Value v;
    v.id = id;
    v.type = type;
    v.span = span;
    v.debugName = std::move(debugName);
    v.origin = origin;
    graph_.values_.push_back(std::move(v));
    graph_.consumers_.emplace_back();
    return id;
  }

  TypedGraph graph_;
};

// Re-checks IR.md §1's invariants, no matter how the graph was built.
// This catches bugs from a broken pass or a bad hand-built fixture. An
// empty result means the graph is valid. Every diagnostic here uses
// DiagCode::IrInvariantViolation.
class GraphVerifier {
 public:
  DiagnosticList verify(const TypedGraph& g) const;
};

}  // namespace tensora
```

### `src/shared/Ir.cpp`
```cpp
// This file implements the Typed Computation Graph (TCG).
// Semantic analysis creates the TCG. Optimization and codegen read it.
#include "tensora/shared/Ir.h"

namespace tensora {

const char* opKindName(OpKind op) {
  switch (op) {
    case OpKind::Add: return "add";
    case OpKind::Sub: return "sub";
    case OpKind::Mul: return "mul";
    case OpKind::Div: return "div";
    case OpKind::MatMul: return "matmul";
    case OpKind::Linear: return "linear";
    case OpKind::Relu: return "relu";
    case OpKind::Softmax: return "softmax";
    case OpKind::Reshape: return "reshape";
    case OpKind::Flatten: return "flatten";
    case OpKind::Transpose: return "transpose";
    case OpKind::Sum: return "sum";
    case OpKind::Mean: return "mean";
    case OpKind::Conv2D: return "conv2d";
    case OpKind::MaxPool: return "maxpool";
  }
  return "?";
}

namespace {
void report(DiagnosticList& diags, const SourceSpan& span, std::string msg) {
  Diagnostic d;
  d.severity = Severity::Error;
  d.code = DiagCode::IrInvariantViolation;
  d.span = span;
  d.message = std::move(msg);
  diags.push_back(std::move(d));
}
}  // namespace

DiagnosticList GraphVerifier::verify(const TypedGraph& g) const {
  DiagnosticList diags;
  const size_t numValues = g.values().size();
  const size_t numNodes = g.nodes().size();

  // GraphBuilder always sets node.id to that node's position in the
  // node list. A hand-built fixture might not, so this checks that
  // fact directly before the rest of this function relies on it.
  for (size_t i = 0; i < numNodes; ++i) {
    if (g.nodes()[i].id != i) {
      report(diags, g.nodes()[i].span, "node id does not match its position in the node list");
    }
  }

  std::vector<bool> isParam(numValues, false);
  for (ValueId p : g.params) {
    if (p < numValues) isParam[p] = true;
  }

  // Every value's producer must match its origin: a param has no
  // producer, and every other value is produced by exactly the node
  // that records it as an output (SSA, IR.md §1.1/§1.2).
  for (size_t vid = 0; vid < numValues; ++vid) {
    const Value& v = g.values()[vid];
    if (v.origin != Value::Origin::Internal) {
      if (v.producer != kNoProducer) {
        report(diags, v.span, "parameter value '" + v.debugName + "' unexpectedly has a producer node");
      }
      if (!isParam[vid]) {
        report(diags, v.span, "input/weight value '" + v.debugName + "' is not listed in the graph's params");
      }
    } else if (v.producer == kNoProducer || v.producer >= numNodes) {
      report(diags, v.span, "internal value '" + v.debugName + "' has no valid producer (dangling value)");
    } else if (g.nodes()[v.producer].output != vid) {
      report(diags, v.span,
             "internal value '" + v.debugName + "' is not the recorded output of its producer node");
    }
  }

  // Checks that the graph is closed, acyclic, and in topological
  // order: every node's operands are either a parameter, or produced
  // by a strictly earlier node (IR.md §1.3/§1.4).
  for (size_t i = 0; i < numNodes; ++i) {
    const GraphNode& node = g.nodes()[i];
    for (ValueId in : node.inputs) {
      if (in >= numValues) {
        report(diags, node.span, std::string("node '") + opKindName(node.op) + "' references a dangling operand");
        continue;
      }
      const Value& operand = g.values()[in];
      if (operand.producer == kNoProducer) {
        if (!isParam[in]) {
          report(diags, node.span,
                 std::string("node '") + opKindName(node.op) +
                     "' operand is neither a parameter nor produced by any node");
        }
        continue;
      }
      if (!(operand.producer < i)) {
        report(diags, node.span,
               std::string("node '") + opKindName(node.op) +
                   "' operand is not produced by a strictly earlier node (cycle or bad order)");
      }
    }
  }

  // Checks for a single, valid output (IR.md §1.5). Reachability from
  // params already follows from the closed and topological checks
  // above: every node's operands trace back to a param or an earlier
  // node, transitively. So only a bounds check is needed here.
  if (numValues == 0 || g.output >= numValues) {
    report(diags, SourceSpan{}, "graph '" + g.name + "' has no valid output value");
  }

  return diags;
}

}  // namespace tensora
```

## 10.6 Optimization Listing (Level-1 graph passes, liveness analysis)

### `include/tensora/optimization/GraphPass.h`
```cpp
#pragma once
#include <memory>
#include <string_view>
#include <vector>

#include "tensora/shared/Diagnostic.h"
#include "tensora/shared/Ir.h"
#include "tensora/shared/Type.h"

// Level-1 graph optimization (RM 3.21; ARCHITECTURE.md §6a): the six
// frozen v1 passes over the SSA Typed Computation Graph. Every pass
// must preserve the observable `return` value of the graph, for all
// inputs that satisfy the checked program's semantics (RM 3.21). See
// each pass's own precondition comment in optimization/GraphPasses.cpp
// for why its transformation is legal.
namespace tensora {

struct PassStats {
  size_t numChanges = 0;  // e.g. nodes removed/fused/folded; pass-specific meaning, always 0 == no-op
  size_t iterations = 0;  // set by PassManager::runAll: how many full sweeps over the pass list ran
};

struct PassResult {
  bool changed = false;
  DiagnosticList diagnostics;
  PassStats stats;
};

struct PassContext {
  TypeContext& types;
  DiagnosticEngine& diags;
};

class GraphPass {
 public:
  virtual ~GraphPass() = default;
  virtual std::string_view name() const = 0;
  // Structural passes rebuild `graph` in place (via a fresh GraphBuilder
  // internally, then move-assign) rather than mutating it directly,
  // since GraphBuilder remains "the only sanctioned way to create
  // nodes/values" (ARCH-008) even during optimization. Pure annotation
  // passes (tagging via TypedGraph::addFact) mutate in place directly.
  virtual PassResult run(TypedGraph& graph, PassContext& ctx) = 0;
};

// Runs the fixed sequence of passes to a fixed point: one "sweep" runs
// every pass once, in order; runAll repeats the sweep as long as the
// last sweep changed the graph, up to kMaxSweeps. This is what lets one
// pass's output create a new opportunity for an earlier pass -- e.g.
// DeadOpElimination dropping a value can bring a Linear node's
// consumer count down to one, which is exactly what
// LinearBiasActivationFusion requires but did not see in the sweep
// before. Every pass here is monotonic (a structural pass only removes
// or merges nodes; a tagging pass only adds a fact, and never re-tags a
// node it already tagged -- see GraphPasses.cpp), so a sweep with no
// change means every pass has actually reached its own fixed point, not
// just paused. kMaxSweeps is a defensive cap, not something a correct
// pass set should ever hit; it exists so a future pass bug (one that
// reports `changed` without making real progress) fails safe -- it
// stops, instead of hanging the compiler.
class PassManager {
 public:
  void add(std::unique_ptr<GraphPass> pass) { passes_.push_back(std::move(pass)); }

  PassResult runAll(TypedGraph& graph, PassContext& ctx) {
    constexpr size_t kMaxSweeps = 10;
    PassResult combined;
    for (; combined.stats.iterations < kMaxSweeps; ++combined.stats.iterations) {
      bool sweepChanged = false;
      for (auto& pass : passes_) {
        PassResult r = pass->run(graph, ctx);
        sweepChanged = sweepChanged || r.changed;
        combined.changed = combined.changed || r.changed;
        combined.stats.numChanges += r.stats.numChanges;
        combined.diagnostics.insert(combined.diagnostics.end(), r.diagnostics.begin(), r.diagnostics.end());
      }
      if (!sweepChanged) {
        ++combined.stats.iterations;  // count the final, no-change confirmation sweep too
        break;
      }
    }
    return combined;
  }

  const std::vector<std::unique_ptr<GraphPass>>& passes() const { return passes_; }

 private:
  std::vector<std::unique_ptr<GraphPass>> passes_;
};

// The six frozen v1 passes (RM 3.21). Constant folding was retired: its
// same-kind scalar-chain combining cannot be proven to preserve observable
// output under IEEE-754 f32 non-associativity, which RM 3.21 requires of
// every enabled pass.
std::unique_ptr<GraphPass> createShapeFoldingPass();
std::unique_ptr<GraphPass> createDeadOpEliminationPass();
std::unique_ptr<GraphPass> createReshapeCancellationPass();
std::unique_ptr<GraphPass> createTransposeCancellationPass();
std::unique_ptr<GraphPass> createElementwiseFusionPass();
std::unique_ptr<GraphPass> createLinearBiasActivationFusionPass();

}  // namespace tensora
```

### `src/optimization/GraphPasses.cpp`
```cpp
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "tensora/optimization/GraphPass.h"

// The six frozen v1 graph passes (RM 3.21). Every *structural* pass
// (Dead, Reshape, Transpose, ShapeFolding) builds a fresh graph with
// GraphBuilder (ARCH-008: GraphBuilder is the only allowed way to
// create nodes and values), instead of changing TypedGraph's internals
// directly. It then move-assigns the result back into the caller's
// reference. This satisfies the INTERFACES.md signature
// `run(TypedGraph&, PassContext&)`, which looks like true in-place
// mutation to the caller, without breaking that invariant. The two
// *fusion* passes only add annotations (TypedGraph::addFact) and
// change the graph in place directly, since tagging changes no
// structural invariant.
namespace tensora {

namespace {

bool isElementwise(OpKind op) {
  return op == OpKind::Add || op == OpKind::Sub || op == OpKind::Mul || op == OpKind::Div;
}

// True if `node` already carries a fact starting with `prefix`. Both
// fusion passes below tag a node once and must never tag it again: since
// PassManager::runAll re-sweeps the pass list to a fixed point (e.g. a
// later DeadOpElimination sweep can drop a node's other consumer, which
// then makes an earlier fusion pass see this same pair as eligible
// again), a fusion pass that re-tagged an already-tagged pair every
// sweep would report `changed` forever and never reach a fixed point.
bool hasFactPrefix(const GraphNode& node, std::string_view prefix) {
  for (const std::string& fact : node.facts.all()) {
    if (fact.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

// Copies every param 1:1 into a fresh builder for the same graph name.
// This records the old -> new ValueId mapping. Here, that mapping is
// the identity, since params are added in the same order first. This
// code still computes it explicitly, instead of assuming it.
GraphBuilder startRebuild(const TypedGraph& graph, std::unordered_map<ValueId, ValueId>& oldToNew) {
  GraphBuilder builder(graph.name);
  for (ValueId p : graph.params) {
    const Value& v = graph.value(p);
    const auto* tt = static_cast<const TensorType*>(v.type);
    ValueId newId = (v.origin == Value::Origin::Input) ? builder.addInput(v.debugName, tt, v.span)
                                                        : builder.addWeight(v.debugName, tt, v.span);
    oldToNew[p] = newId;
  }
  return builder;
}

}  // namespace

// --- DeadOpElimination ---
// Precondition: none. This pass is legal because a node whose output
// `return` never uses, even transitively, has no observable effect at
// all. Removing it cannot change the graph's return value (RM 3.21).
class DeadOpEliminationPass : public GraphPass {
 public:
  std::string_view name() const override { return "DeadOpElimination"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    const size_t numValues = graph.values().size();
    std::vector<bool> reachableNode(numNodes, false);
    std::vector<bool> visitedValue(numValues, false);
    std::vector<ValueId> worklist{graph.output};

    while (!worklist.empty()) {
      ValueId v = worklist.back();
      worklist.pop_back();
      if (v >= numValues || visitedValue[v]) continue;
      visitedValue[v] = true;
      NodeId prod = graph.producerOf(v);
      if (prod == kNoProducer || prod >= numNodes || reachableNode[prod]) continue;
      reachableNode[prod] = true;
      for (ValueId in : graph.node(prod).inputs) worklist.push_back(in);
    }

    size_t deadCount = 0;
    for (bool r : reachableNode) {
      if (!r) ++deadCount;
    }
    if (deadCount == 0) return result;

    std::unordered_map<ValueId, ValueId> oldToNew;
    GraphBuilder builder = startRebuild(graph, oldToNew);
    for (size_t i = 0; i < numNodes; ++i) {
      if (!reachableNode[i]) continue;
      const GraphNode& node = graph.node(i);
      std::vector<ValueId> newInputs;
      newInputs.reserve(node.inputs.size());
      for (ValueId in : node.inputs) newInputs.push_back(oldToNew.at(in));
      const auto* resultType = static_cast<const TensorType*>(graph.value(node.output).type);
      ValueId newOut = builder.emit(node.op, std::move(newInputs), node.attrs, resultType, node.span,
                                     graph.value(node.output).debugName);
      oldToNew[node.output] = newOut;
    }
    builder.setOutput(oldToNew.at(graph.output));
    graph = builder.finish();

    result.changed = true;
    result.stats.numChanges = deadCount;
    return result;
  }
};

// --- ReshapeCancellation ---
// Pattern: reshape(reshape(x, A), B), where only the outer reshape
// uses the inner one. Precondition: reshape is a pure re-labeling
// operation. It needs no name relationship between the source and
// result axes (RM 3.12). So composing two re-labelings, each already
// checked on its own by semantic analysis, is the same as one reshape
// straight to the final target B.
class ReshapeCancellationPass : public GraphPass {
 public:
  std::string_view name() const override { return "ReshapeCancellation"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    std::unordered_set<NodeId> skip;
    std::unordered_map<ValueId, ValueId> bypass;  // skipped node's output -> its own input

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& outer = graph.node(i);
      if (outer.op != OpKind::Reshape) continue;
      NodeId innerId = graph.producerOf(outer.inputs[0]);
      if (innerId == kNoProducer) continue;
      const GraphNode& inner = graph.node(innerId);
      if (inner.op != OpKind::Reshape) continue;
      if (graph.consumersOf(inner.output).size() != 1) continue;  // inner used elsewhere: cannot drop it
      skip.insert(innerId);
      bypass[inner.output] = inner.inputs[0];
    }
    if (skip.empty()) return result;

    std::unordered_map<ValueId, ValueId> oldToNew;
    GraphBuilder builder = startRebuild(graph, oldToNew);
    auto resolve = [&](ValueId v) {
      while (bypass.count(v)) v = bypass.at(v);
      return oldToNew.at(v);
    };
    for (size_t i = 0; i < numNodes; ++i) {
      if (skip.count(static_cast<NodeId>(i))) continue;
      const GraphNode& node = graph.node(i);
      std::vector<ValueId> newInputs;
      newInputs.reserve(node.inputs.size());
      for (ValueId in : node.inputs) newInputs.push_back(resolve(in));
      const auto* resultType = static_cast<const TensorType*>(graph.value(node.output).type);
      ValueId newOut = builder.emit(node.op, std::move(newInputs), node.attrs, resultType, node.span,
                                     graph.value(node.output).debugName);
      oldToNew[node.output] = newOut;
    }
    builder.setOutput(resolve(graph.output));
    graph = builder.finish();

    result.changed = true;
    result.stats.numChanges = skip.size();
    return result;
  }
};

// --- TransposeCancellation ---
// Pattern: transpose(transpose(x, P1), P2), where only the outer
// transpose uses the inner one. Precondition: composing two
// permutations and applying the result once gives the same axis order
// as applying each permutation in sequence (permutation composition is
// associative). If the composed permutation is the identity, the two
// transposes cancel out completely.
class TransposeCancellationPass : public GraphPass {
 public:
  std::string_view name() const override { return "TransposeCancellation"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    std::unordered_set<NodeId> skip;
    std::unordered_map<ValueId, ValueId> bypass;
    std::unordered_map<NodeId, std::vector<uint32_t>> overridePerm;

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& outer = graph.node(i);
      if (outer.op != OpKind::Transpose) continue;
      NodeId innerId = graph.producerOf(outer.inputs[0]);
      if (innerId == kNoProducer) continue;
      const GraphNode& inner = graph.node(innerId);
      if (inner.op != OpKind::Transpose) continue;
      if (graph.consumersOf(inner.output).size() != 1) continue;
      if (inner.attrs.perm.size() != outer.attrs.perm.size()) continue;  // defensive; sema guarantees this anyway

      // z[i] = y[P2[i]] = x[P1[P2[i]]]  (P1 = inner.perm, P2 = outer.perm)
      std::vector<uint32_t> composed(outer.attrs.perm.size());
      bool isIdentity = true;
      for (size_t k = 0; k < composed.size(); ++k) {
        composed[k] = inner.attrs.perm[outer.attrs.perm[k]];
        if (composed[k] != k) isIdentity = false;
      }

      skip.insert(innerId);
      bypass[inner.output] = inner.inputs[0];
      if (isIdentity) {
        skip.insert(static_cast<NodeId>(i));
        bypass[outer.output] = inner.inputs[0];
      } else {
        overridePerm[static_cast<NodeId>(i)] = std::move(composed);
      }
    }
    if (skip.empty()) return result;

    std::unordered_map<ValueId, ValueId> oldToNew;
    GraphBuilder builder = startRebuild(graph, oldToNew);
    auto resolve = [&](ValueId v) {
      while (bypass.count(v)) v = bypass.at(v);
      return oldToNew.at(v);
    };
    for (size_t i = 0; i < numNodes; ++i) {
      NodeId nid = static_cast<NodeId>(i);
      if (skip.count(nid)) continue;
      const GraphNode& node = graph.node(i);
      std::vector<ValueId> newInputs;
      newInputs.reserve(node.inputs.size());
      for (ValueId in : node.inputs) newInputs.push_back(resolve(in));

      NodeAttrs attrs = node.attrs;
      auto it = overridePerm.find(nid);
      if (it != overridePerm.end()) attrs.perm = it->second;

      const auto* resultType = static_cast<const TensorType*>(graph.value(node.output).type);
      ValueId newOut = builder.emit(node.op, std::move(newInputs), attrs, resultType, node.span,
                                     graph.value(node.output).debugName);
      oldToNew[node.output] = newOut;
    }
    builder.setOutput(resolve(graph.output));
    graph = builder.finish();

    result.changed = true;
    result.stats.numChanges = skip.size();
    return result;
  }
};

// --- ShapeFolding ---
// Pattern: a Reshape or Flatten node whose result type exactly matches
// its own input type. Precondition: such a node is a pure no-op
// re-labeling back to the same names and sizes it started with. So its
// consumers can use its input directly (RM 3.12/3.13 place no
// requirement that a reshape or flatten actually change anything).
class ShapeFoldingPass : public GraphPass {
 public:
  std::string_view name() const override { return "ShapeFolding"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    std::unordered_set<NodeId> skip;
    std::unordered_map<ValueId, ValueId> bypass;

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& node = graph.node(i);
      if (node.op != OpKind::Reshape && node.op != OpKind::Flatten) continue;
      const Type* inType = graph.value(node.inputs[0]).type;
      const Type* outType = graph.value(node.output).type;
      if (!typesEqual(inType, outType)) continue;
      skip.insert(static_cast<NodeId>(i));
      bypass[node.output] = node.inputs[0];
    }
    if (skip.empty()) return result;

    std::unordered_map<ValueId, ValueId> oldToNew;
    GraphBuilder builder = startRebuild(graph, oldToNew);
    auto resolve = [&](ValueId v) {
      while (bypass.count(v)) v = bypass.at(v);
      return oldToNew.at(v);
    };
    for (size_t i = 0; i < numNodes; ++i) {
      if (skip.count(static_cast<NodeId>(i))) continue;
      const GraphNode& node = graph.node(i);
      std::vector<ValueId> newInputs;
      newInputs.reserve(node.inputs.size());
      for (ValueId in : node.inputs) newInputs.push_back(resolve(in));
      const auto* resultType = static_cast<const TensorType*>(graph.value(node.output).type);
      ValueId newOut = builder.emit(node.op, std::move(newInputs), node.attrs, resultType, node.span,
                                     graph.value(node.output).debugName);
      oldToNew[node.output] = newOut;
    }
    builder.setOutput(resolve(graph.output));
    graph = builder.finish();

    result.changed = true;
    result.stats.numChanges = skip.size();
    return result;
  }
};

// --- ElementwiseFusion ---
// Pattern: the longest chain of two or more elementwise ops (Add, Sub,
// Mul, Div), where the next op in the chain is the only consumer of
// each intermediate result. Precondition: none is needed just to *tag*
// the chain, since this pass only records a fact; it does not change
// the graph's structure or result. Lowering time decides whether it is
// legal to actually fuse a tagged chain's codegen into one
// linalg.generic, and lowering is where it reads the tag (IR.md §4:
// facts let a later phase use a proven property "without re-deriving
// them").
class ElementwiseFusionPass : public GraphPass {
 public:
  std::string_view name() const override { return "ElementwiseFusion"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    size_t groupCounter = 0;

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& node = graph.node(i);
      if (!isElementwise(node.op)) continue;
      if (hasFactPrefix(node, "fusion_group_")) continue;  // already tagged in an earlier sweep

      NodeId pred = graph.producerOf(node.inputs[0]);
      bool isChainStart = !(pred != kNoProducer && isElementwise(graph.node(pred).op) &&
                            graph.consumersOf(node.inputs[0]).size() == 1);
      if (!isChainStart) continue;

      std::vector<NodeId> chain{static_cast<NodeId>(i)};
      NodeId cur = static_cast<NodeId>(i);
      while (graph.consumersOf(graph.node(cur).output).size() == 1) {
        NodeId next = graph.consumersOf(graph.node(cur).output)[0];
        if (!isElementwise(graph.node(next).op)) break;
        chain.push_back(next);
        cur = next;
      }
      if (chain.size() < 2) continue;

      std::string tag = "fusion_group_" + std::to_string(groupCounter++);
      for (NodeId n : chain) graph.addFact(n, tag);
      result.changed = true;
      ++result.stats.numChanges;
    }
    return result;
  }
};

// --- LinearBiasActivationFusion ---
// Pattern: a Relu node whose one operand is a Linear node's output,
// where that Relu is the Linear's only consumer. Precondition: the
// same as ElementwiseFusion -- this pass only tags the pair (RM 3.9's
// Linear already includes the bias-add, so "Linear+Bias+Activation"
// means exactly "Linear immediately followed by an activation"). A
// later lowering pass fuses the codegen.
class LinearBiasActivationFusionPass : public GraphPass {
 public:
  std::string_view name() const override { return "LinearBiasActivationFusion"; }

  PassResult run(TypedGraph& graph, PassContext&) override {
    PassResult result;
    const size_t numNodes = graph.nodes().size();
    size_t groupCounter = 0;

    for (size_t i = 0; i < numNodes; ++i) {
      const GraphNode& relu = graph.node(i);
      if (relu.op != OpKind::Relu) continue;
      NodeId pred = graph.producerOf(relu.inputs[0]);
      if (pred == kNoProducer) continue;
      const GraphNode& linear = graph.node(pred);
      if (linear.op != OpKind::Linear) continue;
      if (graph.consumersOf(linear.output).size() != 1) continue;
      if (hasFactPrefix(linear, "fused_linear_relu_")) continue;  // already tagged in an earlier sweep

      std::string tag = "fused_linear_relu_" + std::to_string(groupCounter++);
      graph.addFact(pred, tag);
      graph.addFact(static_cast<NodeId>(i), tag);
      result.changed = true;
      ++result.stats.numChanges;
    }
    return result;
  }
};

std::unique_ptr<GraphPass> createShapeFoldingPass() { return std::make_unique<ShapeFoldingPass>(); }
std::unique_ptr<GraphPass> createDeadOpEliminationPass() { return std::make_unique<DeadOpEliminationPass>(); }
std::unique_ptr<GraphPass> createReshapeCancellationPass() { return std::make_unique<ReshapeCancellationPass>(); }
std::unique_ptr<GraphPass> createTransposeCancellationPass() {
  return std::make_unique<TransposeCancellationPass>();
}
std::unique_ptr<GraphPass> createElementwiseFusionPass() { return std::make_unique<ElementwiseFusionPass>(); }
std::unique_ptr<GraphPass> createLinearBiasActivationFusionPass() {
  return std::make_unique<LinearBiasActivationFusionPass>();
}

}  // namespace tensora
```

### `include/tensora/optimization/Liveness.h`
```cpp
#pragma once
#include <cstdint>
#include <unordered_map>

#include "tensora/shared/Ir.h"

// Tensor liveness (ARCH-009; INTERFACES.md §4a): a real data-flow
// analysis over the SSA Typed Computation Graph. It is read-only, and
// its result feeds real optimization/codegen decisions (e.g.
// codegen/LoopTiling.h), not just a report number.
namespace tensora {

// f32 byte size of a tensor type (RM 3.23: no dtype, all f32). An unbound
// symbolic dimension is conservatively treated as size 1 -- callers that
// know a concrete size (e.g. a benchmark's chosen batch size) should
// supply it via `symbolBindings`.
int64_t tensorByteSize(const TensorType* t, const std::unordered_map<SymbolId, int64_t>& symbolBindings = {});

struct Interval {
  // kNoProducer for a graph parameter (live from graph entry, i.e.
  // before any node).
  NodeId firstDef = kNoProducer;
  // The NodeId of the last node that consumes this value, or
  // graph.nodes().size() (one past the last real node) if the value is
  // (also) the graph's `return`ed output.
  NodeId lastUse = 0;
  int64_t bytes = 0;
};

struct LivenessResult {
  std::unordered_map<ValueId, Interval> intervals;
  int64_t peakLiveBytes = 0;  // max total size of simultaneously-live tensors, anywhere in the graph
};

class LivenessAnalysis {
 public:
  LivenessResult run(const TypedGraph& graph,
                      const std::unordered_map<SymbolId, int64_t>& symbolBindings = {}) const;
};

// A minimal real "decision that consumes the estimate" (ARCH-009,
// INTERFACES.md §4a): would a schedule with this peak live-memory
// footprint exceed a given budget? Used directly by TilingDecision
// (codegen/LoopTiling.h) and independently unit-tested (LIVE-004).
bool exceedsMemoryBudget(const LivenessResult& result, int64_t budgetBytes);

}  // namespace tensora
```

### `src/optimization/Liveness.cpp`
```cpp
// This file implements the tensor-liveness analysis.
// It computes each value's live interval and a peak-live-byte estimate
// from the verified Typed Computation Graph. Codegen's Level-2 tiling
// decision reads this estimate (see codegen/LoopTiling.cpp).
#include "tensora/optimization/Liveness.h"

#include <algorithm>
#include <vector>

namespace tensora {

int64_t tensorByteSize(const TensorType* t, const std::unordered_map<SymbolId, int64_t>& symbolBindings) {
  int64_t product = 1;
  for (const Axis& ax : t->axes()) {
    if (ax.size.kind == Dimension::Kind::Const) {
      product *= ax.size.constant;
    } else {
      auto it = symbolBindings.find(ax.size.symbol);
      product *= (it != symbolBindings.end()) ? it->second : 1;
    }
  }
  return product * 4;  // f32 (RM 3.23): 4 bytes/element
}

LivenessResult LivenessAnalysis::run(const TypedGraph& graph,
                                      const std::unordered_map<SymbolId, int64_t>& symbolBindings) const {
  LivenessResult result;
  const auto numNodes = static_cast<long>(graph.nodes().size());

  std::vector<ValueId> allValues;
  allValues.reserve(graph.params.size() + graph.nodes().size());
  for (ValueId p : graph.params) allValues.push_back(p);
  for (const GraphNode& n : graph.nodes()) allValues.push_back(n.output);

  for (ValueId v : allValues) {
    NodeId producer = graph.producerOf(v);
    long firstPos = (producer == kNoProducer) ? -1 : static_cast<long>(producer);
    long lastPos = firstPos;
    for (NodeId c : graph.consumersOf(v)) lastPos = std::max(lastPos, static_cast<long>(c));
    if (v == graph.output) lastPos = std::max(lastPos, numNodes);  // `return` is a use

    Interval interval;
    interval.firstDef = producer;
    interval.lastUse = static_cast<NodeId>(lastPos);
    interval.bytes = tensorByteSize(static_cast<const TensorType*>(graph.value(v).type), symbolBindings);
    result.intervals[v] = interval;
  }

  // A sweep line over "position" (a node index, or -1 for params, or
  // numNodes for `return`). This tracks the net byte change at each
  // position. The max of the running sum is the peak footprint of all
  // values live at the same time.
  std::unordered_map<long, int64_t> delta;
  for (ValueId v : allValues) {
    const Interval& iv = result.intervals.at(v);
    long firstPos = (iv.firstDef == kNoProducer) ? -1 : static_cast<long>(iv.firstDef);
    long lastPos = static_cast<long>(iv.lastUse);
    delta[firstPos] += iv.bytes;
    delta[lastPos + 1] -= iv.bytes;
  }
  std::vector<long> positions;
  positions.reserve(delta.size());
  for (const auto& [pos, d] : delta) positions.push_back(pos);
  std::sort(positions.begin(), positions.end());

  int64_t running = 0, peak = 0;
  for (long pos : positions) {
    running += delta.at(pos);
    peak = std::max(peak, running);
  }
  result.peakLiveBytes = peak;
  return result;
}

bool exceedsMemoryBudget(const LivenessResult& result, int64_t budgetBytes) {
  return result.peakLiveBytes > budgetBytes;
}

}  // namespace tensora
```

## 10.7 Codegen Listing (Tensora-to-MLIR lowering, Level-2 tiling pass, backend)

### `include/tensora/codegen/DialectSetup.h`
```cpp
#pragma once
#include "mlir/IR/MLIRContext.h"

// Registers the shared MLIR dialects and bufferization interfaces
// (INTEG-003: LLVM/MLIR 21). MlirLowerer and the backend/execution path
// both use this function. Every tool in lowering/ builds an MLIRContext
// the same way.
namespace tensora {

mlir::DialectRegistry makeTensoraDialectRegistry();

}  // namespace tensora
```

### `src/codegen/DialectSetup.cpp`
```cpp
// This file builds the shared MLIR dialect registry.
// Codegen uses this registry to lower the Typed Computation Graph to MLIR.
#include "tensora/codegen/DialectSetup.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

namespace tensora {

mlir::DialectRegistry makeTensoraDialectRegistry() {
  using namespace mlir;
  DialectRegistry registry;
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect,
                  scf::SCFDialect, math::MathDialect, LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerTilingInterfaceExternalModels(registry);
  return registry;
}

}  // namespace tensora
```

### `include/tensora/codegen/MlirLowerer.h`
```cpp
#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "tensora/codegen/LoopTiling.h"
#include "tensora/shared/Ir.h"

// Lowers a Typed Computation Graph to MLIR (ARCHITECTURE.md §7.1;
// INTERFACES.md §5). Each TypedGraph becomes a func.func, using the
// func/arith/tensor/linalg dialects. Axis names are erased here
// (ARCH-005): a concrete dim becomes a static tensor dimension, and a
// symbolic dim becomes dynamic (`?`). This class reads the verified
// TypedGraph as-is. It never re-derives a shape or re-checks an
// operator rule that semantic analysis already checked.
namespace tensora {

class MlirLowerer {
 public:
  // `tilingPolicy` must match the TilingPolicy given to
  // createMatmulTilingPass (MatmulTilingPass.h) for the same build.
  // This is what makes TilingDecision the single source of the tile
  // sizes MatmulTilingPass applies, instead of the two computing their
  // own policy that only happens to share default values.
  explicit MlirLowerer(mlir::MLIRContext& ctx, TilingPolicy tilingPolicy = {})
      : ctx_(ctx), tilingDecision_(tilingPolicy) {}

  // One mlir::ModuleOp containing one func.func per graph in `program`,
  // each marked for JIT/native invocation (llvm.emit_c_interface).
  mlir::ModuleOp lower(const TypedProgram& program);

 private:
  mlir::MLIRContext& ctx_;
  TilingDecision tilingDecision_;
};

}  // namespace tensora
```

### `src/codegen/MlirLowerer.cpp`
```cpp
// This file implements the codegen phase's lowering step.
// It converts a verified, optimized Typed Computation Graph to MLIR
// (func/linalg/tensor/arith/scf). Standard MLIR/LLVM passes take the
// MLIR this file produces the rest of the way to native code.
#include "tensora/codegen/MlirLowerer.h"

#include <unordered_map>
#include <unordered_set>

#include <limits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace tensora {

using namespace mlir;
// tensora::Value and tensora::Type (shared/Ir.h, shared/Type.h) share
// their names with mlir::Value and mlir::Type. The enclosing
// `namespace tensora` always wins over `using namespace mlir` for a
// bare name. So every MLIR value or type below must use these aliases
// instead of the bare names.
using MlirValue = mlir::Value;
using MlirType = mlir::Type;

namespace {

// ARCH-005: concrete dim -> static; symbolic dim -> dynamic (?).
RankedTensorType lowerTensorType(const TensorType* t, OpBuilder& b) {
  SmallVector<int64_t> shape;
  shape.reserve(t->rank());
  for (const Axis& ax : t->axes()) {
    shape.push_back(ax.size.kind == Dimension::Kind::Const ? ax.size.constant : ShapedType::kDynamic);
  }
  return RankedTensorType::get(shape, b.getF32Type());
}

MlirValue zeroFilledAccumulator(OpBuilder& b, Location loc, RankedTensorType type, ValueRange dynSizes) {
  MlirValue empty = tensor::EmptyOp::create(b, loc, type, dynSizes);
  MlirValue zero = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
  return linalg::FillOp::create(b, loc, TypeRange{type}, ValueRange{zero}, ValueRange{empty}).getResult(0);
}

// Right-aligns `operandType` to `outRank` (RM 3.8 trailing-axis
// alignment). Operand axis i maps to output axis (i + rankDiff). If
// semantic analysis marked an operand axis as the size-1 broadcast
// side, that axis maps to a constant-0 affine expression instead of a
// real dimension. This matches exactly which side semantic analysis
// verified as broadcastable. This function does not re-check
// broadcast legality; it only replays what semantic analysis already
// proved.
AffineMap broadcastOperandMap(MLIRContext* ctx, const TensorType* operandType, size_t outRank) {
  size_t inRank = operandType->rank();
  size_t rankDiff = outRank - inRank;
  SmallVector<AffineExpr> exprs(inRank);
  for (size_t i = 0; i < inRank; ++i) {
    const Dimension& d = operandType->axes()[i].size;
    bool isBroadcastOne = (d.kind == Dimension::Kind::Const && d.constant == 1);
    exprs[i] = isBroadcastOne ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(i + rankDiff, ctx);
  }
  return AffineMap::get(outRank, 0, exprs, ctx);
}

SmallVector<utils::IteratorType> parallelIterators(size_t rank) {
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

MlirValue elementwiseBinaryOp(OpBuilder& b, Location loc, OpKind op, MlirValue lhs, MlirValue rhs) {
  switch (op) {
    case OpKind::Add: return arith::AddFOp::create(b, loc, lhs, rhs);
    case OpKind::Sub: return arith::SubFOp::create(b, loc, lhs, rhs);
    case OpKind::Mul: return arith::MulFOp::create(b, loc, lhs, rhs);
    case OpKind::Div: return arith::DivFOp::create(b, loc, lhs, rhs);
    default: return lhs;
  }
}

// The same shape as `t`, except axis `axisIdx` has size 1. This is the
// "keep-dims" shape of a reduction along that one axis (RM 3.10/3.15:
// softmax, sum, and mean each reduce exactly one named axis).
RankedTensorType withAxisSizeOne(RankedTensorType t, size_t axisIdx) {
  SmallVector<int64_t> shape(t.getShape().begin(), t.getShape().end());
  shape[axisIdx] = 1;
  return RankedTensorType::get(shape, t.getElementType());
}

// An identity map on every axis except `axisIdx`. That one axis reads
// and writes its one (size-1) slot, instead of a per-iteration index.
// This lets a keep-dims reduction result broadcast back against the
// full-rank tensor it was reduced from, using the same generic-op
// broadcast mechanism as `broadcastOperandMap` above. This is that
// mechanism's reduction-axis case.
AffineMap keepDimMap(MLIRContext* ctx, size_t rank, size_t axisIdx) {
  SmallVector<AffineExpr> exprs(rank);
  for (size_t i = 0; i < rank; ++i) {
    exprs[i] = (i == axisIdx) ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(i, ctx);
  }
  return AffineMap::get(rank, 0, exprs, ctx);
}

// Reduces `x` (rank R, shape `xType`) along `axisIdx` to a keep-dims
// (size-1-at-axisIdx) result, by taking the max or the sum. These are
// the two reductions softmax needs (RM 3.11's numerically-stable
// method: subtract the max, then exponentiate, then normalize by the
// sum).
MlirValue reduceAlongAxis(OpBuilder& b, Location loc, MlirValue x, RankedTensorType xType, size_t axisIdx,
                          bool isMax) {
  size_t rank = xType.getRank();
  RankedTensorType redType = withAxisSizeOne(xType, axisIdx);
  MlirValue init;
  if (isMax) {
    MlirValue negInf =
        arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    MlirValue empty = tensor::EmptyOp::create(b, loc, redType, ValueRange{});
    init = linalg::FillOp::create(b, loc, TypeRange{redType}, ValueRange{negInf}, ValueRange{empty}).getResult(0);
  } else {
    init = zeroFilledAccumulator(b, loc, redType, {});
  }
  AffineMap inMap = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
  AffineMap outMap = keepDimMap(b.getContext(), rank, axisIdx);
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  iters[axisIdx] = utils::IteratorType::reduction;
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{redType}, ValueRange{x}, ValueRange{init}, ArrayRef<AffineMap>{inMap, outMap}, iters,
      [&](OpBuilder& body, Location l, ValueRange args) {
        MlirValue result = isMax ? MlirValue(arith::MaximumFOp::create(body, l, args[0], args[1]))
                                  : MlirValue(arith::AddFOp::create(body, l, args[0], args[1]));
        linalg::YieldOp::create(body, l, ValueRange{result});
      });
  return generic.getResult(0);
}

// Applies a binary scalar op between full-rank `x` and a keep-dims
// (size-1-at-axisIdx) `reduced`, and broadcasts `reduced` back out
// along that axis. This is softmax's "subtract the max" and "divide by
// the sum" step.
MlirValue broadcastAxisBinary(OpBuilder& b, Location loc, MlirValue x, MlirValue reduced, RankedTensorType xType,
                              size_t axisIdx, OpKind op) {
  size_t rank = xType.getRank();
  MlirValue init = tensor::EmptyOp::create(b, loc, xType, ValueRange{});
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
  AffineMap redMap = keepDimMap(b.getContext(), rank, axisIdx);
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{xType}, ValueRange{x, reduced}, ValueRange{init}, ArrayRef<AffineMap>{idMap, redMap, idMap},
      parallelIterators(rank), [&](OpBuilder& body, Location l, ValueRange args) {
        MlirValue result = elementwiseBinaryOp(body, l, op, args[0], args[1]);
        linalg::YieldOp::create(body, l, ValueRange{result});
      });
  return generic.getResult(0);
}

MlirValue lowerExp(OpBuilder& b, Location loc, MlirValue x, RankedTensorType xType) {
  size_t rank = xType.getRank();
  MlirValue init = tensor::EmptyOp::create(b, loc, xType, ValueRange{});
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
  auto generic = linalg::GenericOp::create(
      b, loc, TypeRange{xType}, ValueRange{x}, ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap},
      parallelIterators(rank), [&](OpBuilder& body, Location l, ValueRange args) {
        MlirValue e = math::ExpOp::create(body, l, args[0]);
        linalg::YieldOp::create(body, l, ValueRange{e});
      });
  return generic.getResult(0);
}

size_t findAxisIndex(const TensorType* t, const std::string& name) {
  for (size_t i = 0; i < t->rank(); ++i) {
    if (t->axes()[i].name == name) return i;
  }
  llvm_unreachable("MlirLowerer: axis name not found (sema's checkSoftmax/checkReduce already verified it exists)");
}

// Softmax's keep-dims reduction keeps the same rank, with the axis
// size set to 1. RM 3.15's sum/mean instead *removes* the reduced axis
// entirely (rank - 1). The output affine map's domain still spans the
// full input rank, since linalg.generic iterates over the input's
// index space, but its range lists only the surviving dims, in order.
AffineMap dropAxisMap(MLIRContext* ctx, size_t inRank, size_t axisIdx) {
  SmallVector<AffineExpr> exprs;
  exprs.reserve(inRank - 1);
  for (size_t i = 0; i < inRank; ++i) {
    if (i != axisIdx) exprs.push_back(getAffineDimExpr(i, ctx));
  }
  return AffineMap::get(inRank, 0, exprs, ctx);
}

class LowererImpl {
 public:
  explicit LowererImpl(MLIRContext& ctx, const TilingDecision& tilingDecision)
      : ctx_(ctx), tilingDecision_(tilingDecision) {}

  ModuleOp lower(const TypedProgram& program) {
    OpBuilder builder(&ctx_);
    Location loc = builder.getUnknownLoc();
    ModuleOp module = ModuleOp::create(builder, loc);
    builder.setInsertionPointToEnd(module.getBody());
    for (const TypedGraph& graph : program.graphs()) lowerGraph(graph, builder, loc);
    return module;
  }

 private:
  void lowerGraph(const TypedGraph& graph, OpBuilder& moduleBuilder, Location loc) {
    // Computed once per graph and read by tilingDecision_.decide() below --
    // this is what lets the Level-2 tiling decision actually see peak
    // live-memory pressure (ARCH-009), instead of always taking the
    // liveness=nullptr default.
    currentLiveness_ = LivenessAnalysis().run(graph);

    SmallVector<MlirType> argTypes;
    for (ValueId p : graph.params) {
      argTypes.push_back(lowerTensorType(static_cast<const TensorType*>(graph.value(p).type), moduleBuilder));
    }
    MlirType resultType =
        lowerTensorType(static_cast<const TensorType*>(graph.value(graph.output).type), moduleBuilder);
    auto funcType = moduleBuilder.getFunctionType(argTypes, {resultType});
    auto func = func::FuncOp::create(moduleBuilder, loc, graph.name, funcType);
    func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(), moduleBuilder.getUnitAttr());

    Block* entry = func.addEntryBlock();
    OpBuilder b(entry, entry->begin());

    std::unordered_map<ValueId, MlirValue> values;
    for (size_t i = 0; i < graph.params.size(); ++i) values[graph.params[i]] = entry->getArgument(i);

    std::unordered_set<NodeId> alreadyLowered;  // e.g. a Relu consumed by fusion above
    for (size_t i = 0; i < graph.nodes().size(); ++i) {
      auto nid = static_cast<NodeId>(i);
      if (alreadyLowered.count(nid)) continue;
      const GraphNode& node = graph.node(nid);
      values[node.output] = lowerNode(graph, node, values, b, loc, alreadyLowered);
    }

    func::ReturnOp::create(b, loc, ValueRange{values.at(graph.output)});
  }

  MlirValue lowerNode(const TypedGraph& graph, const GraphNode& node, std::unordered_map<ValueId, MlirValue>& values,
                      OpBuilder& b, Location loc, std::unordered_set<NodeId>& alreadyLowered) {
    const auto* outType = static_cast<const TensorType*>(graph.value(node.output).type);
    RankedTensorType mlirOutType = lowerTensorType(outType, b);

    switch (node.op) {
      case OpKind::Add:
      case OpKind::Sub:
      case OpKind::Mul:
      case OpKind::Div:
        return lowerElementwise(graph, node, values, b, loc, mlirOutType);
      case OpKind::Relu:
        return lowerRelu(values.at(node.inputs[0]), b, loc, mlirOutType);
      case OpKind::MatMul: {
        const auto* aType = static_cast<const TensorType*>(graph.value(node.inputs[0]).type);
        const auto* bType = static_cast<const TensorType*>(graph.value(node.inputs[1]).type);
        MlirValue a = values.at(node.inputs[0]);
        MlirValue rhsVal = values.at(node.inputs[1]);
        if (aType->rank() == 2 && bType->rank() == 2) {
          TilingPlan plan = tilingDecision_.decide(node, graph, &currentLiveness_);
          return lowerMatMul(a, rhsVal, b, loc, mlirOutType, plan);
        }
        return lowerBatchedMatMul(a, rhsVal, aType, bType, b, loc, mlirOutType);
      }
      case OpKind::Linear:
        return lowerLinear(graph, node, values, b, loc, mlirOutType, alreadyLowered);
      case OpKind::Softmax:
        return lowerSoftmax(graph, node, values.at(node.inputs[0]), b, loc, mlirOutType);
      case OpKind::Flatten:
        return lowerFlatten(values.at(node.inputs[0]), b, loc, mlirOutType);
      case OpKind::Transpose:
        return lowerTranspose(values.at(node.inputs[0]), node.attrs.perm, b, loc, mlirOutType);
      case OpKind::Conv2D:
        return lowerConv2D(node, values.at(node.inputs[0]), values.at(node.inputs[1]), values.at(node.inputs[2]), b,
                            loc, mlirOutType);
      case OpKind::Sum:
      case OpKind::Mean:
        return lowerReduce(graph, node, values.at(node.inputs[0]), b, loc, mlirOutType);
      case OpKind::MaxPool:
        return lowerMaxPool(node, values.at(node.inputs[0]), b, loc, mlirOutType);
      default:
        // This op is not lowered yet. See docs/testing/TEST_MATRIX.md
        // LOW-* and the final codegen report's Known Limitations.
        llvm_unreachable("MlirLowerer: unsupported OpKind for this milestone");
    }
  }

  // RM 3.8: a tensor-scalar op has no shape constraint, so this lowers
  // it as one uniform map over the tensor. A tensor-tensor op uses the
  // broadcast affine maps above, which replay the alignment semantic
  // analysis already verified.
  MlirValue lowerElementwise(const TypedGraph& graph, const GraphNode& node,
                             std::unordered_map<ValueId, MlirValue>& values, OpBuilder& b, Location loc,
                             RankedTensorType mlirOutType) {
    MlirValue lhs = values.at(node.inputs[0]);
    size_t outRank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});

    if (node.attrs.scalar.has_value()) {
      MlirValue scalar = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(*node.attrs.scalar));
      AffineMap idMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
      auto generic = linalg::GenericOp::create(
          b, loc, TypeRange{mlirOutType}, ValueRange{lhs}, ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap},
          parallelIterators(outRank), [&](OpBuilder& body, Location l, ValueRange args) {
            MlirValue result = elementwiseBinaryOp(body, l, node.op, args[0], scalar);
            linalg::YieldOp::create(body, l, ValueRange{result});
          });
      return generic.getResult(0);
    }

    MlirValue rhs = values.at(node.inputs[1]);
    const auto* lhsType = static_cast<const TensorType*>(graph.value(node.inputs[0]).type);
    const auto* rhsType = static_cast<const TensorType*>(graph.value(node.inputs[1]).type);
    AffineMap lhsMap = broadcastOperandMap(b.getContext(), lhsType, outRank);
    AffineMap rhsMap = broadcastOperandMap(b.getContext(), rhsType, outRank);
    AffineMap outMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{lhs, rhs}, ValueRange{init},
        ArrayRef<AffineMap>{lhsMap, rhsMap, outMap}, parallelIterators(outRank),
        [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue result = elementwiseBinaryOp(body, l, node.op, args[0], args[1]);
          linalg::YieldOp::create(body, l, ValueRange{result});
        });
    return generic.getResult(0);
  }

  MlirValue lowerRelu(MlirValue input, OpBuilder& b, Location loc, RankedTensorType mlirOutType) {
    size_t rank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, b.getContext());
    MlirValue zero = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{input}, ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap},
        parallelIterators(rank), [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue result = arith::MaximumFOp::create(body, l, args[0], zero);
          linalg::YieldOp::create(body, l, ValueRange{result});
        });
    return generic.getResult(0);
  }

  // The dedicated rank-2 path (RM 3.7). This is the one path
  // MatmulTilingPass tiles, so it stays a plain linalg.matmul, instead
  // of going through the general batched-contraction lowering below.
  // `plan` comes from tilingDecision_, computed at the TCG level
  // before any MLIR exists. This function stamps `plan` onto the op as
  // an attribute, which MatmulTilingPass reads directly, instead of
  // computing its own tile sizes from raw MLIR shapes.
  MlirValue lowerMatMul(MlirValue lhs, MlirValue rhs, OpBuilder& b, Location loc, RankedTensorType mlirOutType,
                        const TilingPlan& plan) {
    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    auto matmul = linalg::MatmulOp::create(b, loc, TypeRange{mlirOutType}, ValueRange{lhs, rhs}, ValueRange{acc});
    if (plan.tile) {
      matmul->setAttr("tensora.tile_sizes",
                       b.getDenseI64ArrayAttr({plan.tileSizes[0], plan.tileSizes[1], plan.tileSizes[2]}));
    }
    return matmul.getResult(0);
  }

  // RM 3.7's general case: rank >= 2 with broadcastable leading (batch)
  // axes. This mirrors elementwise broadcasting (broadcastOperandMap
  // above), but with M and N as the two trailing parallel dims and K
  // as the reduction dim, instead of one shared identity map. This is
  // correct for any rank, but MatmulTilingPass does not tile it (only
  // the plain rank-2 path above gets tiled). Batched matmul has no
  // performance goal in v1, only a correctness goal.
  MlirValue lowerBatchedMatMul(MlirValue lhs, MlirValue rhs, const TensorType* lhsType, const TensorType* rhsType,
                               OpBuilder& b, Location loc, RankedTensorType mlirOutType) {
    MLIRContext* ctx = b.getContext();
    size_t outRank = mlirOutType.getRank();
    size_t batchRank = outRank - 2;
    size_t mDim = batchRank, nDim = batchRank + 1, kDim = batchRank + 2;
    size_t numLoopDims = batchRank + 3;

    auto batchExprs = [&](const TensorType* t) {
      size_t tBatchRank = t->rank() - 2;
      size_t rankDiff = batchRank - tBatchRank;
      SmallVector<AffineExpr> exprs;
      for (size_t i = 0; i < tBatchRank; ++i) {
        const Dimension& d = t->axes()[i].size;
        bool isBroadcastOne = (d.kind == Dimension::Kind::Const && d.constant == 1);
        exprs.push_back(isBroadcastOne ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(i + rankDiff, ctx));
      }
      return exprs;
    };

    SmallVector<AffineExpr> lhsExprs = batchExprs(lhsType);
    lhsExprs.push_back(getAffineDimExpr(mDim, ctx));
    lhsExprs.push_back(getAffineDimExpr(kDim, ctx));
    AffineMap lhsMap = AffineMap::get(numLoopDims, 0, lhsExprs, ctx);

    SmallVector<AffineExpr> rhsExprs = batchExprs(rhsType);
    rhsExprs.push_back(getAffineDimExpr(kDim, ctx));
    rhsExprs.push_back(getAffineDimExpr(nDim, ctx));
    AffineMap rhsMap = AffineMap::get(numLoopDims, 0, rhsExprs, ctx);

    SmallVector<AffineExpr> outExprs;
    for (size_t i = 0; i < batchRank; ++i) outExprs.push_back(getAffineDimExpr(i, ctx));
    outExprs.push_back(getAffineDimExpr(mDim, ctx));
    outExprs.push_back(getAffineDimExpr(nDim, ctx));
    AffineMap outMap = AffineMap::get(numLoopDims, 0, outExprs, ctx);

    SmallVector<utils::IteratorType> iters(numLoopDims, utils::IteratorType::parallel);
    iters[kDim] = utils::IteratorType::reduction;

    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{lhs, rhs}, ValueRange{acc},
        ArrayRef<AffineMap>{lhsMap, rhsMap, outMap}, iters, [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue prod = arith::MulFOp::create(body, l, args[0], args[1]);
          MlirValue sum = arith::AddFOp::create(body, l, args[2], prod);
          linalg::YieldOp::create(body, l, ValueRange{sum});
        });
    return generic.getResult(0);
  }

  // RM 3.9's linear, for rank(x) == 1 or rank(x) >= 3. Every leading
  // axis of x passes straight through to the output. Unlike `@`, this
  // has no separate M axis. So this is a batched vector-matrix product
  // against the one shared (K, N) weight, not a batched matmul. Rank 2
  // stays on the dedicated linalg.matmul path (lowerMatMul), since
  // that is the one path MatmulTilingPass tiles.
  MlirValue lowerLinearContraction(MlirValue x, MlirValue w, OpBuilder& b, Location loc,
                                    RankedTensorType mlirOutType) {
    MLIRContext* ctx = b.getContext();
    size_t outRank = mlirOutType.getRank();
    size_t batchRank = outRank - 1;  // == rank(x) - 1
    size_t nDim = batchRank, kDim = batchRank + 1;
    size_t numLoopDims = batchRank + 2;

    SmallVector<AffineExpr> xExprs;
    for (size_t i = 0; i < batchRank; ++i) xExprs.push_back(getAffineDimExpr(i, ctx));
    xExprs.push_back(getAffineDimExpr(kDim, ctx));
    AffineMap xMap = AffineMap::get(numLoopDims, 0, xExprs, ctx);

    AffineMap wMap = AffineMap::get(numLoopDims, 0, {getAffineDimExpr(kDim, ctx), getAffineDimExpr(nDim, ctx)}, ctx);

    SmallVector<AffineExpr> outExprs;
    for (size_t i = 0; i < batchRank; ++i) outExprs.push_back(getAffineDimExpr(i, ctx));
    outExprs.push_back(getAffineDimExpr(nDim, ctx));
    AffineMap outMap = AffineMap::get(numLoopDims, 0, outExprs, ctx);

    SmallVector<utils::IteratorType> iters(numLoopDims, utils::IteratorType::parallel);
    iters[kDim] = utils::IteratorType::reduction;

    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{x, w}, ValueRange{acc},
        ArrayRef<AffineMap>{xMap, wMap, outMap}, iters, [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue prod = arith::MulFOp::create(body, l, args[0], args[1]);
          MlirValue sum = arith::AddFOp::create(body, l, args[2], prod);
          linalg::YieldOp::create(body, l, ValueRange{sum});
        });
    return generic.getResult(0);
  }

  // Lowers x @ W + b (RM 3.9). When LinearBiasActivationFusion tagged
  // this linear and the Relu right after it with the same
  // "fused_linear_relu_*" fact, this function fuses that Relu into the
  // same linalg.generic bias-add op. It reads the graph-level tag
  // instead of finding the fusion pattern again on its own.
  MlirValue lowerLinear(const TypedGraph& graph, const GraphNode& node,
                        std::unordered_map<ValueId, MlirValue>& values, OpBuilder& b, Location loc,
                        RankedTensorType mlirOutType, std::unordered_set<NodeId>& alreadyLowered) {
    MlirValue x = values.at(node.inputs[0]);
    MlirValue w = values.at(node.inputs[1]);
    MlirValue bias = values.at(node.inputs[2]);

    // decide() only applies to an OpKind::MatMul node (ARCH-010's
    // C = A @ B reference computation). For a Linear node, it always
    // returns a no-tile plan, so this call never tiles. This function
    // reuses lowerMatMul only for its linalg.matmul emission, not for
    // its tiling behavior.
    size_t xRank = cast<RankedTensorType>(x.getType()).getRank();
    MlirValue matmulResult = xRank == 2 ? lowerMatMul(x, w, b, loc, mlirOutType, tilingDecision_.decide(node, graph, &currentLiveness_))
                                        : lowerLinearContraction(x, w, b, loc, mlirOutType);

    bool fuseRelu = false;
    NodeId reluId = 0;
    for (const std::string& fact : node.facts.all()) {
      if (fact.rfind("fused_linear_relu_", 0) != 0) continue;
      NodeId consumer = graph.consumersOf(node.output).empty() ? kNoProducer : graph.consumersOf(node.output)[0];
      if (consumer != kNoProducer && graph.node(consumer).op == OpKind::Relu &&
          graph.node(consumer).facts.has(fact)) {
        fuseRelu = true;
        reluId = consumer;
      }
    }

    size_t outRank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    AffineMap outMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
    AffineMap biasMap = AffineMap::get(outRank, 0, {getAffineDimExpr(outRank - 1, b.getContext())}, b.getContext());
    MlirValue zero = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(0.0f));

    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{matmulResult, bias}, ValueRange{init},
        ArrayRef<AffineMap>{outMap, biasMap, outMap}, parallelIterators(outRank),
        [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue biased = arith::AddFOp::create(body, l, args[0], args[1]);
          MlirValue result = fuseRelu ? MlirValue(arith::MaximumFOp::create(body, l, biased, zero)) : biased;
          linalg::YieldOp::create(body, l, ValueRange{result});
        });

    if (fuseRelu) {
      values[graph.node(reluId).output] = generic.getResult(0);
      alreadyLowered.insert(reluId);
    }
    return generic.getResult(0);
  }

  // RM 3.11: numerically-stable softmax along a named axis (subtract
  // the max, exponentiate, sum, then normalize). The result type
  // equals the input type exactly, matching the contract. There is no
  // new shape to compute.
  MlirValue lowerSoftmax(const TypedGraph& graph, const GraphNode& node, MlirValue x, OpBuilder& b, Location loc,
                        RankedTensorType mlirOutType) {
    const auto* xType = static_cast<const TensorType*>(graph.value(node.inputs[0]).type);
    size_t axisIdx = findAxisIndex(xType, *node.attrs.axisName);

    MlirValue m = reduceAlongAxis(b, loc, x, mlirOutType, axisIdx, /*isMax=*/true);
    MlirValue shifted = broadcastAxisBinary(b, loc, x, m, mlirOutType, axisIdx, OpKind::Sub);
    MlirValue expx = lowerExp(b, loc, shifted, mlirOutType);
    MlirValue s = reduceAlongAxis(b, loc, expx, mlirOutType, axisIdx, /*isMax=*/false);
    return broadcastAxisBinary(b, loc, expx, s, mlirOutType, axisIdx, OpKind::Div);
  }

  // RM 3.13: flatten keeps axis 0 (batch) and collapses the rest into
  // one axis. This is exactly `tensor.collapse_shape`'s reassociation
  // [[0],[1..rank-1]].
  MlirValue lowerFlatten(MlirValue x, OpBuilder& b, Location loc, RankedTensorType mlirOutType) {
    auto xType = cast<RankedTensorType>(x.getType());
    size_t rank = xType.getRank();
    ReassociationIndices restGroup;
    for (size_t i = 1; i < rank; ++i) restGroup.push_back(static_cast<int64_t>(i));
    SmallVector<ReassociationIndices> reassoc{ReassociationIndices{0}, restGroup};
    return tensor::CollapseShapeOp::create(b, loc, mlirOutType, x, reassoc);
  }

  // RM 3.14: transpose lowers to `linalg.transpose`. MLIR's own
  // TilingInterface and bufferization infrastructure already handle it
  // as a `LinalgOp`. Nothing project-specific is needed beyond the
  // permutation semantic analysis already verified.
  MlirValue lowerTranspose(MlirValue x, const std::vector<uint32_t>& perm, OpBuilder& b, Location loc,
                           RankedTensorType mlirOutType) {
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    SmallVector<int64_t> permI64(perm.begin(), perm.end());
    auto transpose = linalg::TransposeOp::create(b, loc, x, init, permI64);
    return transpose->getResult(0);
  }

  // RM 3.16: conv2d lowers to `linalg.conv_2d_nchw_fchw`. The input,
  // weight, and output layouts match the contract's NCHW/FCHW layout
  // exactly (RM 3.16). This function zero-fills the accumulator
  // (linalg's conv op accumulates into `outs`, the same as matmul), and
  // adds an explicit `tensor.pad` when padding > 0, since the linalg
  // conv op itself has no padding attribute. The bias-add generic op
  // broadcasts over the channel axis (index 1), not the last axis:
  // unlike Linear's bias, which applies per output position, conv2d's
  // bias applies per output channel.
  MlirValue lowerConv2D(const GraphNode& node, MlirValue x, MlirValue w, MlirValue bias, OpBuilder& b, Location loc,
                        RankedTensorType mlirOutType) {
    int64_t padding = node.attrs.padding;
    int64_t stride = node.attrs.stride;

    MlirValue input = x;
    if (padding > 0) {
      auto xType = cast<RankedTensorType>(x.getType());
      SmallVector<int64_t> paddedShape(xType.getShape().begin(), xType.getShape().end());
      paddedShape[2] += 2 * padding;
      paddedShape[3] += 2 * padding;
      RankedTensorType paddedType = RankedTensorType::get(paddedShape, xType.getElementType());
      MlirValue zero = arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(0.0f));
      auto lowVals = getAsIndexOpFoldResult(b.getContext(), SmallVector<int64_t>{0, 0, padding, padding});
      auto highVals = getAsIndexOpFoldResult(b.getContext(), SmallVector<int64_t>{0, 0, padding, padding});
      input = tensor::PadOp::create(b, loc, paddedType, x, lowVals, highVals, zero);
    }

    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    DenseIntElementsAttr strides = b.getI64TensorAttr({stride, stride});
    DenseIntElementsAttr dilations = b.getI64TensorAttr({1, 1});
    auto conv = linalg::Conv2DNchwFchwOp::create(b, loc, TypeRange{mlirOutType}, ValueRange{input, w}, ValueRange{acc},
                                                  strides, dilations);
    MlirValue convResult = conv.getResult(0);

    size_t outRank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    AffineMap outMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
    AffineMap biasMap = AffineMap::get(outRank, 0, {getAffineDimExpr(1, b.getContext())}, b.getContext());
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{convResult, bias}, ValueRange{init},
        ArrayRef<AffineMap>{outMap, biasMap, outMap}, parallelIterators(outRank),
        [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue result = arith::AddFOp::create(body, l, args[0], args[1]);
          linalg::YieldOp::create(body, l, ValueRange{result});
        });
    return generic.getResult(0);
  }

  // RM 3.15: sum and mean remove the named axis entirely (rank - 1),
  // unlike softmax's keep-dims reduction. Mean runs sum, then a
  // divide-by-axis-size elementwise pass. Sum's accumulator identity
  // is 0, the same zero-fill approach as matmul's and conv2d's
  // accumulators.
  MlirValue lowerReduce(const TypedGraph& graph, const GraphNode& node, MlirValue x, OpBuilder& b, Location loc,
                        RankedTensorType mlirOutType) {
    const auto* xType = static_cast<const TensorType*>(graph.value(node.inputs[0]).type);
    size_t axisIdx = findAxisIndex(xType, *node.attrs.axisName);
    size_t inRank = xType->rank();

    MlirValue acc = zeroFilledAccumulator(b, loc, mlirOutType, {});
    AffineMap inMap = AffineMap::getMultiDimIdentityMap(inRank, b.getContext());
    AffineMap outMap = dropAxisMap(b.getContext(), inRank, axisIdx);
    SmallVector<utils::IteratorType> iters(inRank, utils::IteratorType::parallel);
    iters[axisIdx] = utils::IteratorType::reduction;
    auto generic = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{x}, ValueRange{acc}, ArrayRef<AffineMap>{inMap, outMap}, iters,
        [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue sum = arith::AddFOp::create(body, l, args[0], args[1]);
          linalg::YieldOp::create(body, l, ValueRange{sum});
        });
    MlirValue result = generic.getResult(0);
    if (node.op != OpKind::Mean) return result;

    int64_t axisSize = xType->axes()[axisIdx].size.constant;
    MlirValue count =
        arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(static_cast<float>(axisSize)));
    size_t outRank = mlirOutType.getRank();
    MlirValue init = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    AffineMap idMap = AffineMap::getMultiDimIdentityMap(outRank, b.getContext());
    auto divGeneric = linalg::GenericOp::create(
        b, loc, TypeRange{mlirOutType}, ValueRange{result}, ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap},
        parallelIterators(outRank), [&](OpBuilder& body, Location l, ValueRange args) {
          MlirValue divided = arith::DivFOp::create(body, l, args[0], count);
          linalg::YieldOp::create(body, l, ValueRange{divided});
        });
    return divGeneric.getResult(0);
  }

  // RM 3.17: maxpool lowers to `linalg.pooling_nchw_max`. Its second
  // "input" is a shape-only operand: the op never reads its values,
  // only its spatial extent, which gives the pooling window shape. The
  // op accumulates a running max into `outs`, so this needs the same
  // -inf-fill-then-max pattern softmax's own reduce-max uses. A
  // zero-filled accumulator would be wrong here, unlike for sum,
  // conv2d, or matmul.
  MlirValue lowerMaxPool(const GraphNode& node, MlirValue x, OpBuilder& b, Location loc, RankedTensorType mlirOutType) {
    int64_t size = node.attrs.poolSize;
    int64_t stride = node.attrs.stride;

    MlirValue negInf =
        arith::ConstantOp::create(b, loc, b.getF32Type(), b.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
    MlirValue empty = tensor::EmptyOp::create(b, loc, mlirOutType, ValueRange{});
    MlirValue acc = linalg::FillOp::create(b, loc, TypeRange{mlirOutType}, ValueRange{negInf}, ValueRange{empty})
                        .getResult(0);
    RankedTensorType windowType = RankedTensorType::get({size, size}, b.getF32Type());
    MlirValue window = tensor::EmptyOp::create(b, loc, windowType, ValueRange{});
    DenseIntElementsAttr strides = b.getI64TensorAttr({stride, stride});
    DenseIntElementsAttr dilations = b.getI64TensorAttr({1, 1});
    auto pool = linalg::PoolingNchwMaxOp::create(b, loc, TypeRange{mlirOutType}, ValueRange{x, window},
                                                  ValueRange{acc}, strides, dilations);
    return pool.getResult(0);
  }

  MLIRContext& ctx_;
  TilingDecision tilingDecision_;
  LivenessResult currentLiveness_;  // for the graph currently being lowered; set at the top of lowerGraph
};

}  // namespace

ModuleOp MlirLowerer::lower(const TypedProgram& program) {
  return LowererImpl(ctx_, tilingDecision_).lower(program);
}

}  // namespace tensora
```

### `include/tensora/codegen/LoopTiling.h`
```cpp
#pragma once
#include <array>
#include <cstdint>

#include "tensora/optimization/Liveness.h"
#include "tensora/shared/Ir.h"

// Level-2 loop optimization (ARCH-009; INTERFACES.md §4a). This file
// holds the team's tiling *decision*. It runs on verified shapes, plus
// optional tensor liveness data, at the Typed Computation Graph level,
// before any MLIR exists. MlirLowerer calls this decision and stamps
// the result onto the lowered `linalg.matmul` op as an attribute.
// MatmulTilingPass (codegen/MatmulTilingPass.h) then applies exactly
// that stamped plan; it does not compute its own policy. This keeps
// one source of truth for the tile sizes, and this decision stays
// unit-testable on its own, without any MLIR dependency
// (ARCHITECTURE.md §2).
namespace tensora {

struct TilingPlan {
  bool tile = false;
  std::array<int64_t, 3> tileSizes = {0, 0, 0};  // M, N, K
};

struct TilingPolicy {
  int64_t minDimToTile = 128;   // Below this size on M, N, or K, tiling is not worth its overhead.
  int64_t maxTileSize = 32;     // The normal tile size, used when memory is not under pressure.
  int64_t conservativeTileSize = 16;  // Used instead when the schedule is near a memory budget.
  int64_t memoryBudgetBytes = 1 << 20;  // 1 MiB: an example default working-set budget.
};

class TilingDecision {
 public:
  explicit TilingDecision(TilingPolicy policy = {}) : policy_(policy) {}

  // `matmul` must be an OpKind::MatMul node with both operands at rank 2
  // (the whitepaper's C = A @ B reference computation, ARCH-010).
  // `graph` gives the operand types. `liveness` is optional; if given,
  // it can make the decision more conservative under memory pressure
  // (a real use of the peak-memory estimate, ARCH-009/LIVE-004).
  TilingPlan decide(const GraphNode& matmul, const TypedGraph& graph, const LivenessResult* liveness = nullptr) const;

 private:
  TilingPolicy policy_;
};

}  // namespace tensora
```

### `src/codegen/LoopTiling.cpp`
```cpp
// This file implements the Level-2 tiling decision.
// It picks a matmul tile size from verified shapes and, when available,
// the Level-1 liveness estimate. Codegen applies this decision to the
// lowered MLIR (see MatmulTilingPass.cpp).
#include "tensora/codegen/LoopTiling.h"

#include <algorithm>

namespace tensora {

TilingPlan TilingDecision::decide(const GraphNode& matmul, const TypedGraph& graph,
                                   const LivenessResult* liveness) const {
  TilingPlan plan;
  if (matmul.op != OpKind::MatMul) return plan;  // policy scope: the C = A @ B reference computation

  const auto* lhs = static_cast<const TensorType*>(graph.value(matmul.inputs[0]).type);
  const auto* rhs = static_cast<const TensorType*>(graph.value(matmul.inputs[1]).type);
  if (lhs->rank() != 2 || rhs->rank() != 2) return plan;  // batched matmul: not this policy's scope

  Dimension m = lhs->axes()[0].size;
  Dimension k = lhs->axes()[1].size;
  Dimension n = rhs->axes()[1].size;
  if (m.kind != Dimension::Kind::Const || k.kind != Dimension::Kind::Const || n.kind != Dimension::Kind::Const) {
    return plan;  // symbolic dims: conservative default is "do not tile" (no static loop bound to plan around)
  }

  if (m.constant < policy_.minDimToTile || n.constant < policy_.minDimToTile || k.constant < policy_.minDimToTile) {
    return plan;  // too small: tiling overhead isn't worth it
  }

  int64_t tileSize = policy_.maxTileSize;
  if (liveness && exceedsMemoryBudget(*liveness, policy_.memoryBudgetBytes)) {
    tileSize = policy_.conservativeTileSize;  // a real decision that consumes the peak-memory estimate (LIVE-004)
  }

  plan.tile = true;
  plan.tileSizes = {std::min(m.constant, tileSize), std::min(n.constant, tileSize), std::min(k.constant, tileSize)};
  return plan;
}

}  // namespace tensora
```

### `include/tensora/codegen/MatmulTilingPass.h`
```cpp
#pragma once
#include <memory>

#include "mlir/Pass/Pass.h"

#include "tensora/codegen/LoopTiling.h"

// The project-authored MLIR pass (USP requirement): shape-guided
// structured-loop tiling of linalg.matmul. It uses MLIR's own pass and
// TilingInterface infrastructure to do the mechanical rewrite. This is
// *our* pass. It is registered and run like any other MLIR pass in the
// pipeline, and is distinct from the standard MLIR/LLVM passes that run
// before and after it (canonicalization, bufferization, lowering to
// LLVM).
//
// Name:        MatmulTilingPass ("tensora-tile-matmul")
// Input:       linalg.matmul (tensor-level, before bufferization)
// Invariant:   operand shapes are the same verified M/N/K sema proved
//              (ARCH-005: concrete dims become static tensor dims,
//              carried across the erasure boundary with no data loss;
//              this pass does not re-derive or re-check anything sema
//              did not already prove)
// Legality:    linalg.matmul is defined as C = A @ B (RM 3.7's
//              contraction). Tiling only restructures the loop nest; it
//              keeps the same associative/commutative accumulation that
//              MLIR's own TilingInterface guarantees preserves the
//              computed value.
// Decision:    this pass applies the tile sizes tensora::TilingDecision
//              (LoopTiling.h) already chose; MlirLowerer stamps that
//              plan onto the op as an attribute, and this pass reads it
// Failure:     a matmul with no stamped plan (batched, symbolic, or
//              below the size threshold) is left unchanged; this is not
//              a pass failure
namespace tensora {

std::unique_ptr<mlir::Pass> createMatmulTilingPass(TilingPolicy policy = {});

}  // namespace tensora
```

### `src/codegen/MatmulTilingPass.cpp`
```cpp
// This file implements the Level-2 tiling pass.
// The pass applies the TilingDecision (LoopTiling.cpp) to each
// linalg.matmul op in the lowered MLIR, using MLIR's own tiling
// interface to do the mechanical rewrite.
#include "tensora/codegen/MatmulTilingPass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace tensora {

using namespace mlir;

namespace {

// See MatmulTilingPass.h for the full pass writeup (name, invariants,
// legality, decision, failure behavior).
class MatmulTilingPass : public PassWrapper<MatmulTilingPass, OperationPass<func::FuncOp>> {
 public:
  MatmulTilingPass() = default;
  explicit MatmulTilingPass(TilingPolicy policy) : policy_(policy) {}

  StringRef getArgument() const override { return "tensora-tile-matmul"; }
  StringRef getDescription() const override {
    return "Tensora-authored shape-guided tiling of linalg.matmul (Level-2, ARCH-009)";
  }

  // Applies exactly the tile sizes TilingDecision::decide()
  // (LoopTiling.cpp) already chose at the TCG level. MlirLowerer stamps
  // those sizes onto the op as "tensora.tile_sizes". This pass does not
  // compute its own policy from raw MLIR shapes. `policy_` exists only
  // so a caller can build this pass with the same TilingPolicy
  // MlirLowerer used; this pass does not otherwise use it.
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<linalg::MatmulOp> candidates;
    func.walk([&](linalg::MatmulOp op) { candidates.push_back(op); });

    for (linalg::MatmulOp op : candidates) {
      auto tileSizesAttr = op->getAttrOfType<DenseI64ArrayAttr>("tensora.tile_sizes");
      if (!tileSizesAttr || tileSizesAttr.size() != 3) continue;  // TilingDecision said not to tile this one

      ArrayRef<int64_t> sizes = tileSizesAttr.asArrayRef();
      IRRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(op);
      scf::SCFTilingOptions options;
      options.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(), SmallVector<int64_t>(sizes)));

      auto tilingOp = cast<TilingInterface>(op.getOperation());
      FailureOr<scf::SCFTilingResult> result = scf::tileUsingSCF(rewriter, tilingOp, options);
      if (failed(result)) continue;  // leave unchanged: not a pass failure (see header's "Failure behavior")
      rewriter.replaceOp(op, result->replacements);
      ++numTiled_;
    }
  }

 private:
  TilingPolicy policy_;
  int64_t numTiled_ = 0;
};

}  // namespace

std::unique_ptr<mlir::Pass> createMatmulTilingPass(TilingPolicy policy) {
  return std::make_unique<MatmulTilingPass>(policy);
}

}  // namespace tensora
```

### `tools/benchmark_matmul_tiling.cpp`
```cpp
// Benchmarks MatmulTilingPass across a few candidate tile sizes on the
// same 1024x1024x1024 matmul workload test_tiling_e2e.cpp uses, plus an
// untiled baseline, and prints the measured wall times. This is a
// stand-alone reporting tool, not a CTest case: LOOP-002/003/004
// already check tiling's structure and numerical correctness; this
// tool exists only to produce real, machine-measured timing evidence
// for the tile-size choice, instead of an assumed or invented number.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MatmulTilingPass.h"
#include "tensora/codegen/MlirLowerer.h"

namespace tns = tensora;
using namespace mlir;

namespace {

constexpr int64_t kDim = 1024;
constexpr int kReps = 5;  // report the best of kReps runs, to cut JIT/OS-noise outliers

tns::TypedProgram buildMatmulProgram(tns::TypeContext& types) {
  tns::GraphBuilder builder("mm");
  const tns::TensorType* aT =
      types.tensor({tns::Axis{"m", tns::Dimension::makeConst(kDim)}, tns::Axis{"k", tns::Dimension::makeConst(kDim)}});
  const tns::TensorType* bT =
      types.tensor({tns::Axis{"k", tns::Dimension::makeConst(kDim)}, tns::Axis{"n", tns::Dimension::makeConst(kDim)}});
  const tns::TensorType* cT =
      types.tensor({tns::Axis{"m", tns::Dimension::makeConst(kDim)}, tns::Axis{"n", tns::Dimension::makeConst(kDim)}});
  tns::ValueId a = builder.addInput("a", aT, {});
  tns::ValueId bIn = builder.addInput("b", bT, {});
  tns::ValueId c = builder.emit(tns::OpKind::MatMul, {a, bIn}, {}, cT, {}, "c");
  builder.setOutput(c);
  tns::TypedGraph graph = builder.finish();

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));
  return program;
}

StridedMemRefType<float, 2> makeMemRef(float* data) {
  StridedMemRefType<float, 2> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = kDim;
  m.sizes[1] = kDim;
  m.strides[0] = kDim;
  m.strides[1] = 1;
  return m;
}

// tileSize == 0 means "no tiling pass applied at all." Returns
// {result, best-of-kReps wall time in seconds}, or {NAN, 0} on failure.
std::pair<float, double> benchmarkOneConfig(int64_t tileSize) {
  tns::TypeContext types;
  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(buildMatmulProgram(types));
  if (failed(verify(module))) {
    fprintf(stderr, "module does not verify\n");
    return {NAN, 0.0};
  }

  if (tileSize > 0) {
    PassManager tilingPm(&ctx);
    tilingPm.addNestedPass<func::FuncOp>(
        tns::createMatmulTilingPass(tns::TilingPolicy{/*minDimToTile=*/128, /*maxTileSize=*/tileSize}));
    if (failed(tilingPm.run(module))) {
      fprintf(stderr, "tiling pass failed for tileSize=%lld\n", static_cast<long long>(tileSize));
      return {NAN, 0.0};
    }
  }

  PassManager pm(&ctx);
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.allowReturnAllocsFromLoops = true;
  bufOpts.functionBoundaryTypeConversion = bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(module))) {
    fprintf(stderr, "standard lowering pipeline failed\n");
    return {NAN, 0.0};
  }

  ExecutionEngineOptions engineOpts;
  static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
  static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
  engineOpts.sharedLibPaths = kSharedLibs;
  auto maybeEngine = ExecutionEngine::create(module, engineOpts);
  if (!maybeEngine) {
    fprintf(stderr, "ExecutionEngine::create failed\n");
    return {NAN, 0.0};
  }
  auto engine = std::move(*maybeEngine);
  auto maybeFn = engine->lookup("_mlir_ciface_mm");
  if (!maybeFn) {
    fprintf(stderr, "could not look up '_mlir_ciface_mm'\n");
    return {NAN, 0.0};
  }
  using CIfaceMatmul = void (*)(StridedMemRefType<float, 2>*, StridedMemRefType<float, 2>*,
                                 StridedMemRefType<float, 2>*);
  auto fn = reinterpret_cast<CIfaceMatmul>(*maybeFn);

  std::vector<float> aData(kDim * kDim, 1.0f);
  std::vector<float> bData(kDim * kDim, 1.0f);
  auto aMem = makeMemRef(aData.data());
  auto bMem = makeMemRef(bData.data());

  float result = 0.0f;
  double best = 1e18;
  for (int rep = 0; rep < kReps; ++rep) {
    StridedMemRefType<float, 2> cMem = {};
    auto t0 = std::chrono::steady_clock::now();
    fn(&cMem, &aMem, &bMem);
    auto t1 = std::chrono::steady_clock::now();
    result = cMem.data[cMem.offset];
    std::free(cMem.basePtr);
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (elapsed < best) best = elapsed;
  }
  return {result, best};
}

}  // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  const float expected = static_cast<float>(kDim);

  struct Config {
    const char* label;
    int64_t tileSize;
  };
  const std::vector<Config> configs = {
      {"untiled", 0},
      {"16x16x16", 16},
      {"32x32x32", 32},
      {"64x64x64", 64},
  };

  printf("workload: %lldx%lldx%lld matmul, f32, all-ones operands, best of %d runs per config\n",
         static_cast<long long>(kDim), static_cast<long long>(kDim), static_cast<long long>(kDim), kReps);
  printf("%-12s %12s %10s\n", "config", "seconds", "correct");

  double untiledTime = -1.0;
  for (const Config& c : configs) {
    auto [result, time] = benchmarkOneConfig(c.tileSize);
    bool correct = (result == expected);
    printf("%-12s %12.6f %10s", c.label, time, correct ? "yes" : "NO");
    if (c.tileSize == 0) {
      untiledTime = time;
    } else if (untiledTime > 0) {
      printf("   (%.2fx untiled)", untiledTime / time);
    }
    printf("\n");
    if (!correct) {
      fprintf(stderr, "BENCHMARK FAIL: config %s produced %f, expected %f\n", c.label, result, expected);
      return 1;
    }
  }
  return 0;
}
```

### `tools/codegen_mlp_native.cpp`
```cpp
// CodegenDriver (ARCH-009/ARCHITECTURE.md "BackendPipeline"). This
// compiles the same hand-built MLP TypedGraph LOW-E2E-MLP uses, all
// the way to a real native object file, through LLVM's TargetMachine,
// not the JIT. This lets NATIVE-001 link it and run it as an ahead-
// of-time-compiled executable, proving the pipeline produces real
// machine code, not just something an in-process JIT happens to run.
//
// Usage: codegen_mlp_native <output-object-file-path>
#include <cstdio>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MlirLowerer.h"
#include "tensora/optimization/GraphPass.h"

namespace tns = tensora;
using namespace mlir;
using MlirValue = mlir::Value;

namespace {

// The same graph shape as test_mlp_e2e.cpp's LOW-E2E-MLP (linear ->
// relu -> linear). This project keeps the two in sync on purpose, so
// NATIVE-001 tests the same workload through a different backend
// path as that JIT test, not a new numerical scenario.
const tns::TensorType* buildMlpGraph(tns::TypeContext& types, tns::GraphBuilder& b, tns::ValueId& outValue) {
  using tns::Axis;
  using tns::Dimension;
  const tns::TensorType* xT =
      types.tensor({Axis{"batch", Dimension::makeConst(1)}, Axis{"in", Dimension::makeConst(2)}});
  const tns::TensorType* w1T =
      types.tensor({Axis{"in", Dimension::makeConst(2)}, Axis{"hidden", Dimension::makeConst(2)}});
  const tns::TensorType* b1T = types.tensor({Axis{"hidden", Dimension::makeConst(2)}});
  const tns::TensorType* hT =
      types.tensor({Axis{"batch", Dimension::makeConst(1)}, Axis{"hidden", Dimension::makeConst(2)}});
  const tns::TensorType* w2T =
      types.tensor({Axis{"hidden", Dimension::makeConst(2)}, Axis{"out", Dimension::makeConst(1)}});
  const tns::TensorType* b2T = types.tensor({Axis{"out", Dimension::makeConst(1)}});
  const tns::TensorType* yT =
      types.tensor({Axis{"batch", Dimension::makeConst(1)}, Axis{"out", Dimension::makeConst(1)}});

  tns::ValueId x = b.addInput("x", xT, {});
  tns::ValueId w1 = b.addWeight("W1", w1T, {});
  tns::ValueId b1 = b.addWeight("b1", b1T, {});
  tns::ValueId w2 = b.addWeight("W2", w2T, {});
  tns::ValueId b2 = b.addWeight("b2", b2T, {});

  tns::ValueId h = b.emit(tns::OpKind::Linear, {x, w1, b1}, {}, hT, {}, "h");
  tns::ValueId hRelu = b.emit(tns::OpKind::Relu, {h}, {}, hT, {}, "h_relu");
  tns::ValueId y = b.emit(tns::OpKind::Linear, {hRelu, w2, b2}, {}, yT, {}, "y");
  b.setOutput(y);
  outValue = y;
  return yT;
}

MlirValue constTensor(OpBuilder& b, Location loc, RankedTensorType type, ArrayRef<float> values) {
  auto attr = DenseElementsAttr::get(type, values);
  return arith::ConstantOp::create(b, loc, type, attr);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <output.o>\n", argv[0]);
    return 1;
  }
  const char* outputPath = argv[1];

  // --- Build and optimize the TypedGraph (same as LOW-E2E-MLP). ---
  tns::TypeContext types;
  tns::GraphBuilder builder("mlp");
  tns::ValueId outValue;
  buildMlpGraph(types, builder, outValue);
  tns::TypedGraph graph = builder.finish();

  tns::GraphVerifier verifier;
  if (!verifier.verify(graph).empty()) {
    fprintf(stderr, "codegen_mlp_native FAIL: hand-built graph is not GraphVerifier-clean\n");
    return 1;
  }

  tns::DiagnosticEngine diags;
  tns::PassContext passCtx{types, diags};
  tns::PassManager pm;
  pm.add(tns::createLinearBiasActivationFusionPass());
  pm.add(tns::createDeadOpEliminationPass());
  pm.runAll(graph, passCtx);

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));

  // --- MLIR lowering. ---
  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(program);
  if (failed(verify(module))) {
    fprintf(stderr, "codegen_mlp_native FAIL: lowered module does not verify\n");
    return 1;
  }

  // --- Harness: a zero-arg, scalar-returning entry point, exactly
  // like LOW-E2E-MLP's "test_main". Its ABI is deliberately trivial
  // (no memref marshalling), so the native driver can call it as a
  // plain C function. ---
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  b.setInsertionPointToEnd(module.getBody());
  auto f32 = b.getF32Type();
  auto harnessType = b.getFunctionType({}, {f32});
  auto harness = func::FuncOp::create(b, loc, "tensora_entry", harnessType);
  Block* entry = harness.addEntryBlock();
  OpBuilder hb(entry, entry->begin());

  MlirValue x = constTensor(hb, loc, RankedTensorType::get({1, 2}, f32), {1.0f, 2.0f});
  MlirValue w1 = constTensor(hb, loc, RankedTensorType::get({2, 2}, f32), {1.0f, 0.0f, 0.0f, 1.0f});
  MlirValue b1 = constTensor(hb, loc, RankedTensorType::get({2}, f32), {0.0f, 0.0f});
  MlirValue w2 = constTensor(hb, loc, RankedTensorType::get({2, 1}, f32), {1.0f, 1.0f});
  MlirValue b2 = constTensor(hb, loc, RankedTensorType::get({1}, f32), {0.0f});

  auto mlpFunc = module.lookupSymbol<func::FuncOp>("mlp");
  if (!mlpFunc) {
    fprintf(stderr, "codegen_mlp_native FAIL: lowered module has no 'mlp' function\n");
    return 1;
  }
  auto call = func::CallOp::create(hb, loc, mlpFunc, ValueRange{x, w1, b1, w2, b2});
  MlirValue yTensor = call.getResult(0);
  MlirValue c0 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(0));
  MlirValue elem = tensor::ExtractOp::create(hb, loc, yTensor, ValueRange{c0, c0});
  func::ReturnOp::create(hb, loc, ValueRange{elem});

  if (failed(verify(module))) {
    fprintf(stderr, "codegen_mlp_native FAIL: module with harness does not verify\n");
    return 1;
  }

  // --- Standard MLIR/LLVM lowering pipeline (same as LOW-E2E-MLP). ---
  mlir::PassManager mlirPm(&ctx);
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  mlirPm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  mlirPm.addPass(createConvertLinalgToLoopsPass());
  mlirPm.addPass(createSCFToControlFlowPass());
  mlirPm.addPass(createConvertControlFlowToLLVMPass());
  mlirPm.addPass(createArithToLLVMConversionPass());
  mlirPm.addPass(createFinalizeMemRefToLLVMConversionPass());
  mlirPm.addPass(createConvertFuncToLLVMPass());
  mlirPm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(mlirPm.run(module))) {
    fprintf(stderr, "codegen_mlp_native FAIL: standard lowering pipeline failed\n");
    module.dump();
    return 1;
  }

  // --- Translate to LLVM IR (standard MLIR->LLVM IR export, not a
  // Tensora-authored step). ---
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext, "tensora_mlp");
  if (!llvmModule) {
    fprintf(stderr, "codegen_mlp_native FAIL: MLIR->LLVM IR translation failed\n");
    return 1;
  }

  // --- Emit a real native object file via LLVM's TargetMachine (ahead-
  // of-time codegen, not the ExecutionEngine's in-process JIT). ---
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  llvmModule->setTargetTriple(targetTriple);

  std::string lookupError;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple.str(), lookupError);
  if (!target) {
    fprintf(stderr, "codegen_mlp_native FAIL: TargetRegistry::lookupTarget: %s\n", lookupError.c_str());
    return 1;
  }

  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> targetMachine(target->createTargetMachine(
      targetTriple, /*CPU=*/"generic", /*Features=*/"", targetOptions, llvm::Reloc::PIC_));
  if (!targetMachine) {
    fprintf(stderr, "codegen_mlp_native FAIL: createTargetMachine returned null\n");
    return 1;
  }
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  std::error_code ec;
  llvm::raw_fd_ostream dest(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    fprintf(stderr, "codegen_mlp_native FAIL: could not open '%s': %s\n", outputPath, ec.message().c_str());
    return 1;
  }

  llvm::legacy::PassManager codegenPm;
  if (targetMachine->addPassesToEmitFile(codegenPm, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
    fprintf(stderr, "codegen_mlp_native FAIL: TargetMachine can't emit an object file for this target\n");
    return 1;
  }
  codegenPm.run(*llvmModule);
  dest.flush();

  printf("codegen_mlp_native: wrote native object file to %s (entry point: tensora_entry)\n", outputPath);
  return 0;
}
```

## 10.8 Driver and Build Listing (CLI driver, example programs, CMake)

### `src/driver/main.cpp`
```cpp
// The `tensora` driver: one binary, one flag per compiler phase. This
// lets each phase run on its own: --emit-tokens for the lexer,
// --emit-ast for the parser, --check/--emit-graph for semantic
// analysis, --emit-optimized-graph for optimization, --emit-mlir for
// lowering, --emit-llvm for backend translation. Two more flags run
// every phase together as a real compiler: --run JIT-executes the
// program, and -o emits a native object file. See
// docs/language/02_LANGUAGE_TUTORIAL.md Section 2.12 for the full CLI
// surface.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tensora/lexer/Lexer.h"
#include "tensora/parser/Parser.h"
#include "tensora/optimization/GraphPass.h"
#include "tensora/semantic/SemanticAnalyzer.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MatmulTilingPass.h"
#include "tensora/codegen/MlirLowerer.h"

using namespace tensora;

namespace {

// MlirLowerer computes the Level-2 TilingDecision and stamps its plan
// onto each linalg.matmul. MatmulTilingPass only applies that stamped
// plan. Both share this one TilingPolicy object, so they provably use
// the same policy, instead of two separately built structs that only
// happen to default to the same values.
const TilingPolicy kTilingPolicy{};

std::string readFile(const std::string& path, bool& ok) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    ok = false;
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  ok = true;
  return ss.str();
}

std::string exprToString(const ast::Expr& e);

std::string tensorTypeToString(const ast::TensorTypeSyntax& t) {
  std::string s = "Tensor[";
  for (size_t i = 0; i < t.axes.size(); ++i) {
    const ast::AxisSyntax& ax = t.axes[i];
    s += (i ? ", " : "") + ax.name + "=" +
         (ax.size.k == ast::DimSyntax::K::Const ? std::to_string(ax.size.constant) : ax.size.symbol);
  }
  return s + "]";
}

std::string argToString(const ast::Arg& a) {
  std::string prefix = a.keyword ? (*a.keyword + "=") : "";
  switch (a.kind) {
    case ast::Arg::Kind::Name: return prefix + a.name;
    case ast::Arg::Kind::Int: return prefix + std::to_string(a.intVal);
    case ast::Arg::Kind::Float: return prefix + std::to_string(a.floatVal);
    case ast::Arg::Kind::IntList: {
      std::string s = prefix + "[";
      for (size_t i = 0; i < a.intList.size(); ++i) s += (i ? "," : "") + std::to_string(a.intList[i]);
      return s + "]";
    }
    case ast::Arg::Kind::AxisList: {
      std::string s = prefix + "[";
      for (size_t i = 0; i < a.axisList.size(); ++i) {
        const ast::AxisSyntax& ax = a.axisList[i];
        s += (i ? "," : "") + ax.name + "=" +
             (ax.size.k == ast::DimSyntax::K::Const ? std::to_string(ax.size.constant) : ax.size.symbol);
      }
      return s + "]";
    }
  }
  return prefix;
}

std::string exprToString(const ast::Expr& e) {
  if (const auto* n = dynamic_cast<const ast::NameExpr*>(&e)) return n->name;
  if (const auto* s = dynamic_cast<const ast::ScalarExpr*>(&e)) return std::to_string(s->value);
  if (const auto* mm = dynamic_cast<const ast::MatMulExpr*>(&e)) {
    return exprToString(*mm->lhs) + " @ " + exprToString(*mm->rhs);
  }
  if (const auto* bin = dynamic_cast<const ast::BinaryExpr*>(&e)) {
    static const char* ops[] = {"+", "-", "*", "/"};
    return exprToString(*bin->lhs) + " " + ops[static_cast<int>(bin->op)] + " " + exprToString(*bin->rhs);
  }
  if (const auto* call = dynamic_cast<const ast::CallExpr*>(&e)) {
    std::string s = call->callee + "(";
    for (size_t i = 0; i < call->args.size(); ++i) s += (i ? ", " : "") + argToString(call->args[i]);
    return s + ")";
  }
  return "<?>";
}

// --emit-ast: prints the real AST node tree, using box-drawing
// characters. This is not the source text printed again. It shows the
// tree structure the parser actually built, not just a reformatted
// copy of the input program.
void printTreeLine(const std::string& prefix, bool isLast, const std::string& label) {
  std::cout << prefix << (isLast ? "└── " : "├── ") << label << "\n";
}

std::string childPrefix(const std::string& prefix, bool isLast) { return prefix + (isLast ? "    " : "│   "); }

void printExprTree(const ast::Expr& e, const std::string& prefix, bool isLast) {
  if (const auto* n = dynamic_cast<const ast::NameExpr*>(&e)) {
    printTreeLine(prefix, isLast, "Identifier(" + n->name + ")");
    return;
  }
  if (const auto* s = dynamic_cast<const ast::ScalarExpr*>(&e)) {
    printTreeLine(prefix, isLast, "Scalar(" + std::to_string(s->value) + ")");
    return;
  }
  if (const auto* mm = dynamic_cast<const ast::MatMulExpr*>(&e)) {
    printTreeLine(prefix, isLast, "MatMul");
    std::string cp = childPrefix(prefix, isLast);
    printExprTree(*mm->lhs, cp, false);
    printExprTree(*mm->rhs, cp, true);
    return;
  }
  if (const auto* bin = dynamic_cast<const ast::BinaryExpr*>(&e)) {
    static const char* ops[] = {"Add", "Sub", "Mul", "Div"};
    printTreeLine(prefix, isLast, std::string("BinaryExpr(") + ops[static_cast<int>(bin->op)] + ")");
    std::string cp = childPrefix(prefix, isLast);
    printExprTree(*bin->lhs, cp, false);
    printExprTree(*bin->rhs, cp, true);
    return;
  }
  if (const auto* call = dynamic_cast<const ast::CallExpr*>(&e)) {
    printTreeLine(prefix, isLast, "CallExpr(" + call->callee + ")");
    std::string cp = childPrefix(prefix, isLast);
    for (size_t i = 0; i < call->args.size(); ++i) {
      printTreeLine(cp, i + 1 == call->args.size(), "Arg(" + argToString(call->args[i]) + ")");
    }
    return;
  }
  printTreeLine(prefix, isLast, "?");
}

void printAstTree(const ast::AstProgram& program) {
  std::cout << "Program\n";
  size_t total = program.decls.size() + program.graphs.size();
  size_t i = 0;
  for (const ast::Decl& d : program.decls) {
    bool isLast = (++i == total);
    const char* kind = d.kind == ast::Decl::Kind::Input ? "InputDecl" : "WeightDecl";
    printTreeLine("", isLast, std::string(kind) + "(" + d.name + " : " + tensorTypeToString(d.type) + ")");
  }
  for (const ast::GraphDecl& g : program.graphs) {
    bool isLast = (++i == total);
    printTreeLine("", isLast, "GraphDecl(" + g.name + ")");
    std::string cp = childPrefix("", isLast);
    size_t stmtTotal = g.stmts.size() + 1;  // + 1 for the return statement
    size_t j = 0;
    for (const ast::Stmt& s : g.stmts) {
      bool stmtIsLast = (++j == stmtTotal);
      printTreeLine(cp, stmtIsLast, "Assignment(" + s.name + ")");
      printExprTree(*s.rhs, childPrefix(cp, stmtIsLast), true);
    }
    printTreeLine(cp, true, "Return(" + g.returnName + ")");
  }
}

// --emit-graph / --explain: prints one line per node, with the
// operator, its operands (by debug name), and the *verified* result
// type. This is the same information `describeType` renders in
// diagnostics, so it reuses the exact type-printing the rest of the
// compiler already uses, instead of inventing a second format.
void printTypedGraph(const TypedGraph& g, TypeContext& types, bool explainMode) {
  std::cout << "graph " << g.name << " {\n";
  for (ValueId p : g.params) {
    const Value& v = g.value(p);
    std::cout << "  param " << v.debugName << " : " << describeType(v.type, types) << "\n";
  }
  for (const GraphNode& node : g.nodes()) {
    const Value& out = g.value(node.output);
    std::cout << "  " << out.debugName << " = " << opKindName(node.op) << "(";
    for (size_t i = 0; i < node.inputs.size(); ++i) {
      std::cout << (i ? ", " : "") << g.value(node.inputs[i]).debugName;
    }
    std::cout << ") : " << describeType(out.type, types);
    if (explainMode) {
      // Names the operator contract that produced this node (whitepaper
      // Section 5), by which RM section governs it. That RM section is
      // the contract's identity: there is one contract function per
      // OpKind.
      std::cout << "   # RM " << opKindName(node.op);
    }
    std::cout << "\n";
  }
  std::cout << "  return " << g.value(g.output).debugName << "\n}\n";
}

const char* symbolKindName(Symbol::Kind k) {
  switch (k) {
    case Symbol::Kind::Input: return "input";
    case Symbol::Kind::Weight: return "weight";
    case Symbol::Kind::Value: return "local";
  }
  return "?";
}

// --emit-symbols: renders SemanticAnalyzer::symbolTable() (RM 3.6).
// This prints exactly the declaration-order entries SymbolTable
// actually accepted during name resolution (see Symbol.h's
// orderedGlobals/orderedLocals). It is not a second, separately built
// symbol table. There is one GLOBAL section (input/weight
// declarations, shared across every graph), plus one GRAPH section
// per graph (that graph's final local scope). This matches the two
// scope levels RM 3.6 defines. A trailing RESOLVED section per graph
// shows what each name reference in that graph's body actually
// resolved to: the real resolution log SemanticAnalyzer::resolveOperand
// produced during name resolution, not a replay of it.
void printSymbolTable(const SymbolTableReport& report, TypeContext& types) {
  std::cout << "=== Symbol Table ===\n\nGLOBAL\n";
  for (const Symbol& s : report.globals) {
    std::cout << "  " << s.name << "  " << symbolKindName(s.kind) << "  " << describeType(s.type, types) << "  ("
               << s.span.line << ":" << s.span.col << ")\n";
  }
  for (const SymbolTableReport::GraphScope& scope : report.graphs) {
    std::cout << "\nGRAPH " << scope.graphName << "\n";
    for (const Symbol& s : scope.locals) {
      std::cout << "  " << s.name << "  " << symbolKindName(s.kind) << "  " << describeType(s.type, types) << "  ("
                 << s.span.line << ":" << s.span.col << ")\n";
    }
    if (!scope.resolutions.empty()) {
      std::cout << "  RESOLVED\n";
      for (const auto& [name, resolvedAs] : scope.resolutions) {
        std::cout << "    " << name << " -> " << resolvedAs << "\n";
      }
    }
  }
  std::cout << "\n=== End Symbol Table ===\n";
}

// --emit-ast-dot: a Graphviz DOT dump of the AST, for demos and for
// visually checking that the parser built the tree a program expects
// (e.g. `tensora --emit-ast-dot f.tns | dot -Tpng -o ast.png`).
class AstDotWriter {
 public:
  std::string build(const ast::AstProgram& program) {
    for (const ast::Decl& d : program.decls) {
      node((d.kind == ast::Decl::Kind::Input ? "input " : "weight ") + d.name);
    }
    for (const ast::GraphDecl& g : program.graphs) {
      std::string graphId = node("graph " + g.name);
      for (const ast::Stmt& s : g.stmts) {
        std::string assignId = node("Assignment " + s.name);
        edge(graphId, assignId);
        edge(assignId, emitExpr(*s.rhs));
      }
      edge(graphId, node("return " + g.returnName));
    }
    return "digraph AST {\n" + out_.str() + "}\n";
  }

 private:
  std::string node(const std::string& label) {
    std::string id = "n" + std::to_string(counter_++);
    out_ << "  " << id << " [label=\"" << label << "\"];\n";
    return id;
  }
  void edge(const std::string& from, const std::string& to) { out_ << "  " << from << " -> " << to << ";\n"; }

  std::string emitExpr(const ast::Expr& e) {
    if (const auto* n = dynamic_cast<const ast::NameExpr*>(&e)) return node("Identifier " + n->name);
    if (const auto* s = dynamic_cast<const ast::ScalarExpr*>(&e)) return node("Scalar " + std::to_string(s->value));
    if (const auto* mm = dynamic_cast<const ast::MatMulExpr*>(&e)) {
      std::string id = node("MatMul");
      edge(id, emitExpr(*mm->lhs));
      edge(id, emitExpr(*mm->rhs));
      return id;
    }
    if (const auto* bin = dynamic_cast<const ast::BinaryExpr*>(&e)) {
      static const char* ops[] = {"Add", "Sub", "Mul", "Div"};
      std::string id = node(ops[static_cast<int>(bin->op)]);
      edge(id, emitExpr(*bin->lhs));
      edge(id, emitExpr(*bin->rhs));
      return id;
    }
    if (const auto* call = dynamic_cast<const ast::CallExpr*>(&e)) {
      std::string id = node("Call " + call->callee);
      for (const ast::Arg& a : call->args) edge(id, node(argToString(a)));
      return id;
    }
    return node("?");
  }

  std::ostringstream out_;
  int counter_ = 0;
};

std::string astToDot(const ast::AstProgram& program) { return AstDotWriter().build(program); }

// --emit-graph-dot: a Graphviz DOT dump of the Typed Computation Graph,
// one node per param/graph-node, edges following producer -> consumer.
std::string graphToDot(const TypedGraph& g, TypeContext& types) {
  std::ostringstream out;
  out << "digraph TypedGraph_" << g.name << " {\n";
  for (ValueId p : g.params) {
    const Value& v = g.value(p);
    out << "  v" << p << " [shape=box, style=dashed, label=\"" << v.debugName << "\\n"
        << describeType(v.type, types) << "\"];\n";
  }
  for (const GraphNode& node : g.nodes()) {
    const Value& out_v = g.value(node.output);
    out << "  n" << node.id << " [label=\"" << opKindName(node.op) << "\\n" << describeType(out_v.type, types)
        << "\"];\n";
    for (ValueId in : node.inputs) {
      NodeId producer = g.producerOf(in);
      std::string fromId = (producer == kNoProducer) ? ("v" + std::to_string(in)) : ("n" + std::to_string(producer));
      out << "  " << fromId << " -> n" << node.id << ";\n";
    }
  }
  NodeId outputProducer = g.producerOf(g.output);
  out << "  n" << outputProducer << " -> ret [label=\"return\"];\n";
  out << "  ret [shape=doublecircle, label=\"" << g.value(g.output).debugName << "\"];\n";
  out << "}\n";
  return out.str();
}

// Returns false, with a message on stderr, if any param or output has
// a symbolic dimension. --run and -o need concrete sizes, to build
// input data or to emit a fixed-shape function.
bool allConcreteShapes(const TypedGraph& g) {
  auto checkOne = [&](ValueId id) {
    const auto* t = static_cast<const TensorType*>(g.value(id).type);
    for (const Axis& ax : t->axes()) {
      if (ax.size.kind != Dimension::Kind::Const) return false;
    }
    return true;
  };
  for (ValueId p : g.params) {
    if (!checkOne(p)) return false;
  }
  return checkOne(g.output);
}

int64_t elementCount(const TensorType* t) {
  int64_t n = 1;
  for (const Axis& ax : t->axes()) n *= ax.size.constant;
  return n;
}

// A byte buffer compatible with StridedMemRefType at any runtime rank.
// MLIR's own C-interface ABI fixes this struct layout: {T* basePtr;
// T* data; int64_t offset; int64_t sizes[N]; int64_t strides[N];}
// (mlir/ExecutionEngine/CRunnerUtils.h). Building it through
// StridedMemRefType<float, N>, for a compile-time N, and copying its
// bytes, keeps the layout exactly right, without hand-writing the
// offsets.
struct MemRefBuffer {
  std::vector<char> bytes;
  void* ptr() { return bytes.data(); }
};

// One function per rank (1-4), instead of a template. StridedMemRefType
// itself needs a compile-time rank, since its `sizes` and `strides`
// arrays are fixed-size C arrays. So the rank must be a literal
// number somewhere. Writing it out four times here keeps this
// CLI-only helper readable, with no template code.
MemRefBuffer makeMemRefBuffer1(float* data, const std::vector<int64_t>& sizes) {
  StridedMemRefType<float, 1> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = sizes[0];
  m.strides[0] = 1;
  MemRefBuffer buf;
  buf.bytes.resize(sizeof(m));
  std::memcpy(buf.bytes.data(), &m, sizeof(m));
  return buf;
}

MemRefBuffer makeMemRefBuffer2(float* data, const std::vector<int64_t>& sizes) {
  StridedMemRefType<float, 2> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = sizes[0];
  m.sizes[1] = sizes[1];
  m.strides[1] = 1;
  m.strides[0] = sizes[1];
  MemRefBuffer buf;
  buf.bytes.resize(sizeof(m));
  std::memcpy(buf.bytes.data(), &m, sizeof(m));
  return buf;
}

MemRefBuffer makeMemRefBuffer3(float* data, const std::vector<int64_t>& sizes) {
  StridedMemRefType<float, 3> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = sizes[0];
  m.sizes[1] = sizes[1];
  m.sizes[2] = sizes[2];
  m.strides[2] = 1;
  m.strides[1] = sizes[2];
  m.strides[0] = sizes[1] * sizes[2];
  MemRefBuffer buf;
  buf.bytes.resize(sizeof(m));
  std::memcpy(buf.bytes.data(), &m, sizeof(m));
  return buf;
}

MemRefBuffer makeMemRefBuffer4(float* data, const std::vector<int64_t>& sizes) {
  StridedMemRefType<float, 4> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = sizes[0];
  m.sizes[1] = sizes[1];
  m.sizes[2] = sizes[2];
  m.sizes[3] = sizes[3];
  m.strides[3] = 1;
  m.strides[2] = sizes[3];
  m.strides[1] = sizes[2] * sizes[3];
  m.strides[0] = sizes[1] * sizes[2] * sizes[3];
  MemRefBuffer buf;
  buf.bytes.resize(sizeof(m));
  std::memcpy(buf.bytes.data(), &m, sizeof(m));
  return buf;
}

MemRefBuffer makeMemRefBuffer(float* data, const std::vector<int64_t>& sizes, bool& ok) {
  ok = true;
  switch (sizes.size()) {
    case 1: return makeMemRefBuffer1(data, sizes);
    case 2: return makeMemRefBuffer2(data, sizes);
    case 3: return makeMemRefBuffer3(data, sizes);
    case 4: return makeMemRefBuffer4(data, sizes);
    default:
      ok = false;
      return {};
  }
}

// Reads back element 0, and reports the total count, from a result
// descriptor, after the callee has overwritten it with its own
// allocation's data pointer. A returned-by-value memref is *not*
// written into the caller's original buffer. The ciface wrapper
// stores the callee's new pointer into the descriptor the caller
// passed in. So the descriptor's *fields*, not the original buffer,
// hold the real answer.
float readMemRefElement0(const MemRefBuffer& buf, int rank) {
  switch (rank) {
    case 1: return reinterpret_cast<const StridedMemRefType<float, 1>*>(buf.bytes.data())->data[0];
    case 2: return reinterpret_cast<const StridedMemRefType<float, 2>*>(buf.bytes.data())->data[0];
    case 3: return reinterpret_cast<const StridedMemRefType<float, 3>*>(buf.bytes.data())->data[0];
    case 4: return reinterpret_cast<const StridedMemRefType<float, 4>*>(buf.bytes.data())->data[0];
    default: return NAN;
  }
}

std::vector<int64_t> shapeOf(const TensorType* t) {
  std::vector<int64_t> shape;
  for (const Axis& ax : t->axes()) shape.push_back(ax.size.constant);
  return shape;
}

// Runs the Level-1, team-authored graph optimization passes on every
// graph in the program, in place. This is a separate function from
// lowerToModule below, so the CLI can show the optimization phase's
// own output (--emit-optimized-graph) before lowering ever runs.
// Otherwise the optimization phase's effect would stay invisible,
// hidden inside the MLIR the lowering phase prints.
void runOptimizationPasses(TypedProgram& program) {
  DiagnosticEngine passDiags;
  PassContext passCtx{program.types(), passDiags};
  PassManager passManager;
  passManager.add(createShapeFoldingPass());
  passManager.add(createReshapeCancellationPass());
  passManager.add(createTransposeCancellationPass());
  passManager.add(createElementwiseFusionPass());
  passManager.add(createLinearBiasActivationFusionPass());
  passManager.add(createDeadOpEliminationPass());
  // TypedProgram stores each graph by value. Each pass mutates the
  // graph in place, through the same TypedGraph& the program already
  // owns.
  for (const TypedGraph& g : program.graphs()) {
    passManager.runAll(const_cast<TypedGraph&>(g), passCtx);
  }
}

// Lowers an already-optimized program to MLIR. Call
// runOptimizationPasses first. This function does not run the
// Level-1 passes itself, so the lowering phase's own output
// (--emit-mlir) shows only what lowering itself contributes.
mlir::OwningOpRef<mlir::ModuleOp> lowerToModule(mlir::MLIRContext& ctx, TypedProgram& program) {
  MlirLowerer lowerer(ctx, kTilingPolicy);
  mlir::ModuleOp module = lowerer.lower(program);
  return mlir::OwningOpRef<mlir::ModuleOp>(module);
}

void printUsage(const char* prog) {
  std::cerr << "usage: " << prog << " <stage-flag> <file.tns>\n"
            << "  --emit-tokens          lexer: print the token stream\n"
            << "  --emit-parse-tree      parser: print the raw ANTLR parse tree (LISP-style)\n"
            << "  --emit-ast             parser: print the parsed AST\n"
            << "  --emit-ast-dot         parser: print the parsed AST as Graphviz DOT\n"
            << "  --check                semantic analysis: report OK or diagnostics\n"
            << "  --emit-symbols         semantic analysis: print the symbol table (RM 3.6)\n"
            << "  --emit-graph           semantic analysis: print the Typed Computation Graph\n"
            << "  --emit-graph-dot       semantic analysis: print the Typed Computation Graph as Graphviz DOT\n"
            << "  --explain              semantic analysis: like --emit-graph, annotated per RM section\n"
            << "  --emit-optimized-graph optimization: print the graph after the Level-1 passes\n"
            << "  --emit-mlir            lowering: print the MLIR module\n"
            << "  --emit-llvm            backend: print the translated LLVM IR\n"
            << "  --run                  full pipeline: JIT-execute with synthetic all-ones input data\n"
            << "  -o <out.o>             full pipeline: emit a native object file (ahead-of-time)\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string stage;
  std::string path;
  std::string outPath;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      stage = "-o";
      outPath = argv[++i];
    } else if (!arg.empty() && arg[0] == '-') {
      stage = arg;
    } else {
      path = arg;
    }
  }

  if (path.empty() || stage.empty()) {
    printUsage(argv[0]);
    return 2;
  }

  bool ok = false;
  std::string text = readFile(path, ok);
  if (!ok) {
    std::cerr << "cannot open " << path << "\n";
    return 2;
  }

  SourceBuffer src(path, text);
  DiagnosticEngine diags;

  if (stage == "--emit-tokens") {
    Lexer lexer(src, diags);
    std::vector<Token> tokens = lexer.tokenize();
    for (const Token& t : tokens) {
      std::cout << t.span.line << ":" << t.span.col << "  " << tokenKindName(t.kind);
      if (t.kind == TokenKind::Identifier || t.kind == TokenKind::IntLiteral || t.kind == TokenKind::FloatLiteral) {
        std::cout << "  '" << std::string(t.lexeme) << "'";
      }
      std::cout << "\n";
    }
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }

  if (stage == "--emit-parse-tree") {
    std::cout << dumpParseTree(src) << "\n";
    return 0;
  }

  Parser parser(src, diags);
  Result<ast::AstProgram> parsed = parser.parseProgram();

  if (stage == "--emit-ast") {
    printAstTree(parsed.value());
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (stage == "--emit-ast-dot") {
    std::cout << astToDot(parsed.value());
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (diags.hasErrors()) {
    std::cerr << diags.render(src);
    return 1;
  }

  TypeContext types;
  SemanticAnalyzer analyzer(types, diags);
  Result<TypedProgram> analyzed = analyzer.analyze(parsed.value());

  if (stage == "--check") {
    std::cerr << diags.render(src);
    if (diags.hasErrors()) return 1;
    std::cout << "OK: " << path << " is a well-formed Tensora program\n";
    return 0;
  }
  if (stage == "--emit-symbols") {
    printSymbolTable(analyzer.symbolTable(), types);
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (stage == "--emit-graph" || stage == "--explain") {
    for (const TypedGraph& g : analyzed.value().graphs()) printTypedGraph(g, types, stage == "--explain");
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (stage == "--emit-graph-dot") {
    for (const TypedGraph& g : analyzed.value().graphs()) std::cout << graphToDot(g, types);
    std::cerr << diags.render(src);
    return diags.hasErrors() ? 1 : 0;
  }
  if (diags.hasErrors()) {
    std::cerr << diags.render(src);
    return 1;
  }

  TypedProgram& program = analyzed.value();

  if (stage == "--emit-optimized-graph") {
    runOptimizationPasses(program);
    for (const TypedGraph& g : program.graphs()) printTypedGraph(g, types, /*explainMode=*/false);
    return 0;
  }

  // This driver found a known limitation by running against a real
  // symbolic-batch source file (mlp.tns): MlirLowerer's
  // `tensor.empty()` calls do not yet supply dynamic-size operands for
  // a symbolic dimension. ARCH-005 says symbolic maps to dynamic `?`,
  // but only the *type* lowering follows that rule; the *value*
  // lowering for every op that builds a fresh tensor (elementwise,
  // relu, matmul's and conv2d's accumulator, and so on) assumes every
  // dim is static. Fixing this touches every lowering function, and
  // deserves its own pass and its own tests, not a rushed patch here.
  // Until then, every stage past semantic analysis needs concrete
  // shapes throughout.
  for (const TypedGraph& g : program.graphs()) {
    if (!allConcreteShapes(g)) {
      std::cerr << "error: " << stage << " requires every input/weight and the output to have concrete "
                   "(non-symbolic) sizes; MLIR lowering does not yet support symbolic dimensions past "
                   "type-checking (known limitation -- see docs/testing/TEST_MATRIX.md LOW-005)\n";
      return 1;
    }
  }

  // Runs the optimization phase, then the lowering phase (two
  // separate CLI-visible stages: --emit-optimized-graph and
  // --emit-mlir). Both run here for every later stage
  // (--emit-mlir onward), since each stage needs the one before it.
  runOptimizationPasses(program);

  mlir::DialectRegistry registry = makeTensoraDialectRegistry();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module = lowerToModule(ctx, program);

  // Level-2 / USP: runs the project-authored shape-guided matmul
  // tiling pass on every function. This is a real MLIR pass, distinct
  // from the Level-1 TCG passes already applied above, and from the
  // standard MLIR/LLVM infrastructure below. The default policy (see
  // LoopTiling.h) only tiles a matmul at or above its size threshold.
  // So this is a genuine no-op on small demo graphs, not a forced
  // effect.
  {
    mlir::PassManager tilingPm(&ctx);
    tilingPm.addNestedPass<mlir::func::FuncOp>(createMatmulTilingPass(kTilingPolicy));
    if (failed(tilingPm.run(*module))) {
      std::cerr << "error: MatmulTilingPass failed\n";
      return 1;
    }
  }
  if (failed(mlir::verify(*module))) {
    std::cerr << "internal error: lowered module does not verify\n";
    module->dump();
    return 1;
  }

  if (stage == "--emit-mlir") {
    module->print(llvm::outs());
    return 0;
  }

  // The standard MLIR/LLVM lowering pipeline. Every pass here is
  // stock MLIR infrastructure, not written by this project. This
  // always runs the full set of passes every workload in this phase
  // has needed (subview-bearing tiled loops, softmax's math.exp,
  // plain elementwise/matmul/conv graphs). Each pass is a no-op when
  // its input pattern is not present.
  mlir::PassManager pm(&ctx);
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.allowReturnAllocsFromLoops = true;
  bufOpts.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (failed(pm.run(*module))) {
    std::cerr << "error: standard MLIR/LLVM lowering pipeline failed\n";
    module->dump();
    return 1;
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext, path);
  if (!llvmModule) {
    std::cerr << "error: MLIR->LLVM IR translation failed\n";
    return 1;
  }

  if (stage == "--emit-llvm") {
    llvmModule->print(llvm::outs(), nullptr);
    return 0;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  llvmModule->setTargetTriple(targetTriple);
  std::string lookupError;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple.str(), lookupError);
  if (!target) {
    std::cerr << "error: TargetRegistry::lookupTarget: " << lookupError << "\n";
    return 1;
  }
  llvm::TargetOptions targetOptions;
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      target->createTargetMachine(targetTriple, "generic", "", targetOptions, llvm::Reloc::PIC_));
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  if (stage == "-o") {
    std::error_code ec;
    llvm::raw_fd_ostream dest(outPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      std::cerr << "error: could not open '" << outPath << "': " << ec.message() << "\n";
      return 1;
    }
    llvm::legacy::PassManager codegenPm;
    if (targetMachine->addPassesToEmitFile(codegenPm, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
      std::cerr << "error: TargetMachine can't emit an object file for this target\n";
      return 1;
    }
    codegenPm.run(*llvmModule);
    dest.flush();
    std::cout << "wrote native object file: " << outPath << "\n";
    return 0;
  }

  if (stage == "--run") {
    if (program.graphs().size() != 1) {
      std::cerr << "error: --run requires exactly one graph in the file (found " << program.graphs().size()
                << "); disambiguation across multiple graphs is not supported yet\n";
      return 1;
    }
    const TypedGraph& graph = program.graphs().front();
    if (!allConcreteShapes(graph)) {
      std::cerr << "error: --run requires every input/weight and the output to have concrete (non-symbolic) "
                   "sizes, since it must synthesize input data and cannot allocate a symbolic-sized buffer\n";
      return 1;
    }

    mlir::ExecutionEngineOptions engineOpts;
    static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
    static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
    engineOpts.sharedLibPaths = kSharedLibs;
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOpts);
    if (!maybeEngine) {
      std::cerr << "error: ExecutionEngine::create failed\n";
      return 1;
    }
    auto engine = std::move(*maybeEngine);

    // Uses synthetic all-ones data for every declared input and
    // weight. Tensora source declares shapes, not values (there is no
    // tensor-literal syntax), so a CLI demo run has no other source of
    // data. All-ones is a deliberate, checkable choice, not an
    // arbitrary one: for example, in a matmul, every output element
    // then equals the contracted dimension K, which you can check by
    // hand.
    std::vector<std::vector<float>> storage;
    std::vector<MemRefBuffer> argBufs;
    std::vector<void*> packedArgs;

    const auto* outType = static_cast<const TensorType*>(graph.value(graph.output).type);
    std::vector<int64_t> outShape = shapeOf(outType);
    std::vector<float> outStorage(static_cast<size_t>(elementCount(outType)), 0.0f);
    bool rankOk = false;
    MemRefBuffer resultBuf = makeMemRefBuffer(outStorage.data(), outShape, rankOk);
    if (!rankOk) {
      std::cerr << "error: --run only supports rank 1-4 tensors; output has rank " << outShape.size() << "\n";
      return 1;
    }
    packedArgs.push_back(resultBuf.ptr());

    for (ValueId p : graph.params) {
      const auto* pType = static_cast<const TensorType*>(graph.value(p).type);
      std::vector<int64_t> pShape = shapeOf(pType);
      storage.emplace_back(static_cast<size_t>(elementCount(pType)), 1.0f);
      MemRefBuffer buf = makeMemRefBuffer(storage.back().data(), pShape, rankOk);
      if (!rankOk) {
        std::cerr << "error: --run only supports rank 1-4 tensors; '" << graph.value(p).debugName << "' has rank "
                   << pShape.size() << "\n";
        return 1;
      }
      argBufs.push_back(std::move(buf));
    }
    for (MemRefBuffer& buf : argBufs) packedArgs.push_back(buf.ptr());

    // Calls the ciface wrapper directly, as a plain function pointer
    // with one void* per argument, in declared order, result first.
    // This does not go through ExecutionEngine::invokePacked. This is
    // the same raw-lookup-and-cast pattern this project's other JIT
    // tests already use reliably, generalized over argument count
    // with a switch, since a CLI driver cannot know the parameter
    // count at compile time the way a hand-written test can.
    auto maybeFn = engine->lookup("_mlir_ciface_" + graph.name);
    if (!maybeFn) {
      std::cerr << "error: could not look up '_mlir_ciface_" << graph.name << "'\n";
      return 1;
    }
    void* fnPtr = *maybeFn;
    switch (packedArgs.size()) {
      case 2: reinterpret_cast<void (*)(void*, void*)>(fnPtr)(packedArgs[0], packedArgs[1]); break;
      case 3: reinterpret_cast<void (*)(void*, void*, void*)>(fnPtr)(packedArgs[0], packedArgs[1], packedArgs[2]); break;
      case 4:
        reinterpret_cast<void (*)(void*, void*, void*, void*)>(fnPtr)(packedArgs[0], packedArgs[1], packedArgs[2],
                                                                       packedArgs[3]);
        break;
      case 5:
        reinterpret_cast<void (*)(void*, void*, void*, void*, void*)>(fnPtr)(
            packedArgs[0], packedArgs[1], packedArgs[2], packedArgs[3], packedArgs[4]);
        break;
      case 6:
        reinterpret_cast<void (*)(void*, void*, void*, void*, void*, void*)>(fnPtr)(
            packedArgs[0], packedArgs[1], packedArgs[2], packedArgs[3], packedArgs[4], packedArgs[5]);
        break;
      case 7:
        reinterpret_cast<void (*)(void*, void*, void*, void*, void*, void*, void*)>(fnPtr)(
            packedArgs[0], packedArgs[1], packedArgs[2], packedArgs[3], packedArgs[4], packedArgs[5], packedArgs[6]);
        break;
      case 8:
        reinterpret_cast<void (*)(void*, void*, void*, void*, void*, void*, void*, void*)>(fnPtr)(
            packedArgs[0], packedArgs[1], packedArgs[2], packedArgs[3], packedArgs[4], packedArgs[5], packedArgs[6],
            packedArgs[7]);
        break;
      default:
        std::cerr << "error: --run only supports up to 7 declared inputs/weights (found "
                   << (packedArgs.size() - 1) << ")\n";
        return 1;
    }

    int64_t total = elementCount(outType);
    float first = readMemRefElement0(resultBuf, static_cast<int>(outShape.size()));
    std::cout << "ran '" << graph.name << "' with synthetic all-ones inputs\n";
    std::cout << "output shape: [";
    for (size_t i = 0; i < outShape.size(); ++i) std::cout << (i ? ", " : "") << outShape[i];
    std::cout << "], " << total << " elements; element[0] = " << first << "\n";
    return 0;
  }

  printUsage(argv[0]);
  return 2;
}
```

### `examples/mlp.tns`
```text
# A small multilayer perceptron (used to demo the lexer).
input  x:  Tensor[batch=B, feat=784]
weight W1: Tensor[in=784, out=256]
weight b1: Tensor[out=256]
weight W2: Tensor[in=256, out=10]
weight b2: Tensor[out=10]

graph mlp {
    h = linear(x, W1, b1)
    h = relu(h)
    y = linear(h, W2, b2)
    return y
}
```

### `examples/mlp_concrete.tns`
```text
# A small MLP with a concrete batch size (used to demo codegen/--run/-o,
# which do not yet support symbolic dimensions -- see mlp.tns for the
# symbolic-batch version used to demo the lexer/parser/semantic phases).
input  x:  Tensor[batch=4, feat=8]
weight W1: Tensor[in=8, out=16]
weight b1: Tensor[out=16]
weight W2: Tensor[in=16, out=4]
weight b2: Tensor[out=4]

graph mlp {
    h = linear(x, W1, b1)
    h = relu(h)
    y = linear(h, W2, b2)
    return y
}
```

### `examples/matmul_large.tns`
```text
# A large matmul (256x256x256, above MatmulTilingPass's default 128
# threshold) -- used to demo the Level-2 loop-tiling pass through the
# real compiler, not just its dedicated unit tests.
input a: Tensor[m=256, k=256]
input b: Tensor[k=256, n=256]

graph matmul_large {
    c = a @ b
    return c
}
```

### `examples/cnn.tns`
```text
# A small CNN classifier (conv2d -> relu -> flatten -> linear ->
# softmax), matching tests/codegen/test_cnn_e2e.cpp's shapes so the
# hand-computed reference there (class-1 probability = 0.539902 with a
# specific 1..16 input) can be cross-checked; --run instead uses
# synthetic all-ones data, so its printed value will differ from that
# reference (it exercises the full source-to-execution pipeline, not
# this exact numeric case).
input  x:     Tensor[n=1, c=1, h=4, w=4]
weight convW: Tensor[oc=1, ic=1, kh=2, kw=2]
weight convB: Tensor[oc=1]
weight fcW:   Tensor[flat=4, classes=2]
weight fcB:   Tensor[classes=2]

graph cnn {
    conv   = conv2d(x, convW, convB, stride=2, padding=0)
    relu1  = relu(conv)
    flat   = flatten(relu1)
    logits = linear(flat, fcW, fcB)
    probs  = softmax(logits, axis=classes)
    return probs
}
```

### `examples/attention.tns`
```text
# Single-head scaled dot-product attention (matmul -> transpose -> scale
# -> softmax -> matmul), matching tests/codegen/test_attention_e2e.cpp's
# shapes. This is a rank-2 (no batch/head axes) demonstration: the
# backend's matmul lowering is rank-2 only for now (batched matmul is a
# known limitation -- see docs/testing/TEST_MATRIX.md), so a compiled
# multi-head version (as shown in docs/language/02_LANGUAGE_TUTORIAL.md
# Section 2.11) is not runnable through this driver yet, even though it
# type-checks under semantic analysis.
input q: Tensor[seq_q=2, d_k=2]
input k: Tensor[seq_k=2, d_k=2]
input v: Tensor[seq_k=2, d_v=2]

graph attention {
    kt      = transpose(k, perm=[1, 0])
    scores  = q @ kt
    scaled  = scores * 0.70710678
    weights = softmax(scaled, axis=seq_k)
    out     = weights @ v
    return out
}
```

### `CMakeLists.txt`
```cmake
cmake_minimum_required(VERSION 3.16)

# NEW: re-extract FetchContent URL downloads' timestamps at extraction
# time instead of preserving the archive's own timestamps, so a change
# to a pinned URL's content is always seen as a change. Set before
# `project()` per the policy's own documented scope.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(tensora CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Give the vendored ANTLR4 runtime's own CMakeLists a build type so it
# does not warn and fall back to Release itself; Release is also this
# project's own default when none is requested on the command line.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Frontend generator: ANTLR4. The build fetches both the generator tool (a
# jar, run through a system `java`) and the C++ runtime it needs. This
# keeps the build reproducible. No machine-specific install path is
# ever hard-coded.
find_package(Java QUIET COMPONENTS Runtime)
if(NOT Java_JAVA_EXECUTABLE)
  message(FATAL_ERROR "ANTLR code generation needs a Java runtime (`java`) on PATH; "
                       "see docs/development/BUILD.md.")
endif()

set(ANTLR4_VERSION 4.13.2)
include(FetchContent)

FetchContent_Declare(antlr4_jar
  URL https://www.antlr.org/download/antlr-${ANTLR4_VERSION}-complete.jar
  DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(antlr4_jar)
set(ANTLR4_JAR_PATH ${antlr4_jar_SOURCE_DIR}/antlr-${ANTLR4_VERSION}-complete.jar)

# FORCE here always wins over a `-D...` passed on the `cmake` command
# line, on purpose: none of these are options we want a developer to
# be able to turn on by accident (WITH_DEMO in particular builds
# ANTLR's own vendored demo CLI, not anything of ours).
set(ANTLR4_INSTALL OFF CACHE BOOL "" FORCE)
set(WITH_DEMO OFF CACHE BOOL "" FORCE)
set(WITH_LIBCXX OFF CACHE BOOL "" FORCE)
set(ANTLR_BUILD_CPP_TESTS OFF CACHE BOOL "" FORCE)
set(ANTLR_BUILD_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(antlr4_runtime
  URL https://www.antlr.org/download/antlr4-cpp-runtime-${ANTLR4_VERSION}-source.zip)
FetchContent_MakeAvailable(antlr4_runtime)
# The "Building without demo. To enable demo build use: -DWITH_DEMO=True"
# line just above comes from the vendored ANTLR runtime's own
# CMakeLists.txt -- it has no way to know we FORCE that flag off above,
# so its hint does not actually work in this project. This message
# clarifies that in the same configure log, since we cannot edit or
# suppress the vendored one.
message(STATUS "(the WITH_DEMO hint above is from the vendored ANTLR runtime and does not apply here -- this project always disables it; see CMakeLists.txt)")

# antlr4_static's own CMakeLists exports its headers as normal (not
# SYSTEM) INTERFACE_INCLUDE_DIRECTORIES, so every consumer that links
# it -- including tensora_lexer/tensora_parser below -- was compiling vendored
# headers such as RuleContext.h/ParseTree.h as if they were our own
# code, surfacing -Woverloaded-virtual/-Wunused-parameter warnings from
# a project we do not own. Re-declaring the same paths as
# INTERFACE_SYSTEM_INCLUDE_DIRECTORIES (a target property CMake reads
# specifically for this situation) fixes that without patching the
# vendored CMakeLists itself.
set_property(TARGET antlr4_static PROPERTY
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
  "$<TARGET_PROPERTY:antlr4_static,INTERFACE_INCLUDE_DIRECTORIES>")

# Generate TensoraLexer/TensoraParser (and a visitor base) from
# grammar/Tensora.g4 into the build tree. Generated code is never
# committed and never hand-edited. It stays separate from project-owned
# C++ in src/parser/AstBuilder.cpp, which walks the parse tree.
set(ANTLR4_GEN_DIR ${CMAKE_BINARY_DIR}/antlr4_generated)
file(MAKE_DIRECTORY ${ANTLR4_GEN_DIR})
set(ANTLR4_GENERATED_SRCS
  ${ANTLR4_GEN_DIR}/TensoraLexer.cpp
  ${ANTLR4_GEN_DIR}/TensoraParser.cpp)
add_custom_command(
  OUTPUT ${ANTLR4_GENERATED_SRCS}
  COMMAND ${Java_JAVA_EXECUTABLE} -jar ${ANTLR4_JAR_PATH}
          -Dlanguage=Cpp -no-listener -no-visitor
          -o ${ANTLR4_GEN_DIR}
          -package tensora::generated
          ${CMAKE_SOURCE_DIR}/grammar/Tensora.g4
  DEPENDS ${CMAKE_SOURCE_DIR}/grammar/Tensora.g4
  COMMENT "ANTLR: generating TensoraLexer/TensoraParser from grammar/Tensora.g4")
add_custom_target(antlr4_generate DEPENDS ${ANTLR4_GENERATED_SRCS})

include_directories(${CMAKE_SOURCE_DIR}/include)

# -Wall -Wextra for project-owned targets only (docs/development/
# 07_DEVELOPMENT_ENVIRONMENT.md §7.5). Placed after the ANTLR
# FetchContent/generate setup above, so it does not apply to the
# vendored ANTLR runtime target, only to the libraries and executables
# this CMakeLists.txt defines below.
add_compile_options(-Wall -Wextra)

# Lexer library: the project's own Lexer/Token wrapper (used by the
# standalone lexer demo/tests only -- Parser.cpp below drives the
# generated ANTLR lexer directly, per Parser.h's own comment) plus the
# generated ANTLR lexer. Kept as its own target, separate from
# tensora_parser below, so `cmake --build build --target lexer` (docs/
# development/BUILD.md) does not also compile any parser-only source.
add_library(tensora_lexer
  src/lexer/Lexer.cpp
  src/lexer/TokenAdapter.cpp
  ${ANTLR4_GEN_DIR}/TensoraLexer.cpp)
add_dependencies(tensora_lexer antlr4_generate)
target_include_directories(tensora_lexer PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tensora_lexer PRIVATE ${ANTLR4_GEN_DIR})
target_link_libraries(tensora_lexer PUBLIC antlr4_static)
# The generated headers include <antlr4-runtime.h>. So every consumer of
# the generated code (this library, and anything that links against it)
# needs the runtime's public include dir. The PUBLIC link to
# antlr4_static already carries that dir, through its own
# INTERFACE_INCLUDE_DIRECTORIES.

# Parser library: AstBuilder (walks ANTLR's parse tree into the AST),
# the parser error listener/strategy, and the generated ANTLR parser.
# Parser.cpp instantiates the generated ANTLR lexer itself (Parser.h's
# own comment: no project Token vector sits between the two here), and
# calls TokenAdapter's adaptToken() for its one diagnostic path -- both
# reasons this links tensora_lexer, which is otherwise correct anyway:
# a parser always needs a lexer.
add_library(tensora_parser
  src/parser/Parser.cpp
  src/parser/AstBuilder.cpp
  src/parser/ParserErrorListener.cpp
  src/parser/ParserErrorStrategy.cpp
  ${ANTLR4_GEN_DIR}/TensoraParser.cpp)
add_dependencies(tensora_parser antlr4_generate)
target_include_directories(tensora_parser PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tensora_parser PRIVATE ${ANTLR4_GEN_DIR})
target_link_libraries(tensora_parser PUBLIC tensora_lexer)

# Semantic analysis phase. shared/{Type,DimSolver,Ir}.cpp holds the
# types and the Typed Computation Graph (used by more than one phase).
# semantic/ holds the symbol table, the operator contracts, and the
# analyzer driver. This library has no MLIR/LLVM dependency
# (ARCHITECTURE.md §2).
add_library(tensora_semantic
  src/shared/Type.cpp
  src/shared/DimSolver.cpp
  src/shared/Ir.cpp
  src/semantic/Symbol.cpp
  src/semantic/Contracts.cpp
  src/semantic/SemanticAnalyzer.cpp)
target_include_directories(tensora_semantic PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tensora_semantic PUBLIC tensora_parser)

# Optimization phase (RM 3.21): the Level-1 graph passes, plus the
# tensor-liveness analysis. Plain C++ over the TCG. No MLIR/LLVM
# dependency (ARCHITECTURE.md §2).
add_library(tensora_optimization src/optimization/GraphPasses.cpp src/optimization/Liveness.cpp)
target_include_directories(tensora_optimization PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tensora_optimization PUBLIC tensora_semantic)

# Codegen phase: the MLIR/LLVM backend (INTEG-003 pins LLVM/MLIR 21, apt
# packages). Only codegen/ touches MLIR/LLVM headers (ARCHITECTURE.md
# §2); semantic/ and shared/ above never do. This block must come
# before tensora_codegen below, which needs MLIR_INCLUDE_DIRS and the
# MLIR imported target.
list(APPEND CMAKE_PREFIX_PATH
  /usr/lib/llvm-21/lib/cmake/mlir
  /usr/lib/llvm-21/lib/cmake/llvm)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
message(STATUS "Found MLIR (${MLIR_DIR})")
add_definitions(${LLVM_DEFINITIONS})

# A JIT-run module can lower to a non-identity-layout memref (tiling's
# subviews, flatten's collapse_shape, conv2d's pad). That code calls
# MLIR's own `memrefCopy` runtime helper. The JIT must know where to
# find that helper. So this line finds it once, here, for every target
# below that needs it (the driver and several end-to-end test binaries).
find_library(MLIR_C_RUNNER_UTILS_LIB mlir_c_runner_utils PATHS ${LLVM_LIBRARY_DIR} ${MLIR_LIBRARY_DIR} REQUIRED)
message(STATUS "Found mlir_c_runner_utils: ${MLIR_C_RUNNER_UTILS_LIB}")

# TCG -> MLIR (ARCHITECTURE.md §7.1). Only this library touches MLIR/
# LLVM headers (ARCHITECTURE.md §2).
add_library(tensora_codegen
  src/codegen/DialectSetup.cpp
  src/codegen/MlirLowerer.cpp
  src/codegen/LoopTiling.cpp
  src/codegen/MatmulTilingPass.cpp)
target_include_directories(tensora_codegen PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tensora_codegen SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(tensora_codegen PUBLIC tensora_optimization PRIVATE MLIR)

# Command-line driver: one flag per compiler phase, plus --run/-o,
# which run the full pipeline (docs/language/02_LANGUAGE_TUTORIAL.md
# §2.12).
add_executable(tensora src/driver/main.cpp)
target_include_directories(tensora SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(tensora PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(tensora PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")

# Tests run through CTest. tests/support/TestHarness.h is the shared,
# no-framework harness (Reporter/Case registry). Every test binary uses it.
enable_testing()

add_executable(test_lexer tests/lexer/test_lexer.cpp)
target_link_libraries(test_lexer PRIVATE tensora_lexer)
target_include_directories(test_lexer PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_parser tests/parser/test_parser.cpp)
target_link_libraries(test_parser PRIVATE tensora_parser)
target_include_directories(test_parser PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_semantic tests/semantic/test_semantic.cpp)
target_link_libraries(test_semantic PRIVATE tensora_semantic)
target_include_directories(test_semantic PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_graph tests/semantic/test_graph.cpp)
target_link_libraries(test_graph PRIVATE tensora_semantic)
target_include_directories(test_graph PRIVATE ${CMAKE_SOURCE_DIR}/tests)

add_executable(test_passes tests/optimization/test_passes.cpp)
target_link_libraries(test_passes PRIVATE tensora_optimization)
target_include_directories(test_passes PRIVATE ${CMAKE_SOURCE_DIR}/tests)

# Each case ID gets its own CTest test (test_lexer/test_parser <ID>
# runs just that case). So `ctest` reports which rule failed, instead
# of one pass/fail result for everything. These IDs match
# docs/testing/TEST_MATRIX.md. Keep the two lists in sync.
set(LEXER_TEST_CASES
  LEX-001 LEX-002 LEX-003 LEX-004 LEX-005 LEX-006 LEX-007 LEX-008
  LEX-009 LEX-010 LEX-011
  BND-LEX-001 BND-LEX-002 BND-LEX-003 BND-LEX-004)
foreach(case_id ${LEXER_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_lexer ${case_id})
endforeach()

set(PARSER_TEST_CASES
  PAR-001 PAR-002 PAR-003 PAR-004 PAR-005 PAR-006 PAR-007 PAR-008 PAR-009
  PAR-010 PAR-011 PAR-012 PAR-013 PAR-014 PAR-015 PAR-016 PAR-017 PAR-018
  PAR-019 PAR-020 PAR-021 PAR-022 PAR-023 PAR-024 PAR-025 PAR-026 PAR-027
  PAR-028 PAR-029 PAR-030 PAR-031 PAR-032
  BND-PAR-001 BND-PAR-002 BND-PAR-003 BND-PAR-004)
foreach(case_id ${PARSER_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_parser ${case_id})
endforeach()

set(SEMANTIC_TEST_CASES
  SEM-DECL-001 SEM-DECL-002 SEM-DECL-003 SEM-DECL-004 SEM-DECL-005 SEM-NAME-001 SEM-NAME-002 SEM-NAME-003
  SEM-SCOPE-001 SEM-SCOPE-002 SEM-TYPE-001
  SEM-ARITY-001 SEM-ARITY-002 SEM-ARITY-003 SEM-ARITY-004 SEM-ARITY-005
  SEM-POISON-001 SEM-POISON-002
  SEM-MM-001 SEM-MM-002 SEM-MM-003 SEM-MM-004
  SEM-LIN-001 SEM-LIN-002 SEM-LIN-003
  SEM-EW-001 SEM-EW-002 SEM-EW-003 SEM-EW-004 SEM-EW-005
  SEM-RS-001 SEM-RS-002 SEM-RS-003 SEM-RS-004
  SEM-FL-001 SEM-FL-002
  SEM-TR-001 SEM-TR-002 SEM-TR-003
  SEM-RD-001 SEM-RD-002
  SEM-SM-001 SEM-SM-002
  SEM-CV-001 SEM-CV-002 SEM-CV-003 SEM-CV-004 SEM-CV-005
  SEM-PL-001 SEM-PL-002
  SEM-DIM-001 SEM-DIM-002 SEM-DIM-003
  SYM-001 SYM-002 SYM-003 SYM-004 SYM-005 SYM-006 SYM-007 SYM-008
  ATT-001 ATT-002
  BND-MM-001 BND-MM-002 BND-BC-001 BND-RS-001 BND-FL-001
  BND-CV-001 BND-CV-002 BND-RD-001 BND-TR-001 BND-DIM-001)
foreach(case_id ${SEMANTIC_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_semantic ${case_id})
endforeach()

set(GRAPH_TEST_CASES GR-001 GR-002 GR-003 GR-004 GR-005 GR-006 GR-007)
foreach(case_id ${GRAPH_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_graph ${case_id})
endforeach()

set(PASSES_TEST_CASES
  OPT-SF-001 OPT-SF-002 OPT-DCE-001 OPT-DCE-002 OPT-DCE-003
  OPT-RC-001 OPT-RC-002 OPT-RC-003 OPT-TC-001 OPT-TC-002 OPT-TC-003
  OPT-EF-001 OPT-EF-002 OPT-LBA-001 OPT-LBA-002 OPT-FIX-001 OPT-FIX-002
  LIVE-001 LIVE-002 LIVE-003 LIVE-004)
# (LOW-E2E-MLP is registered separately below, next to its executable.)
foreach(case_id ${PASSES_TEST_CASES})
  add_test(NAME ${case_id} COMMAND test_passes ${case_id})
endforeach()

# SMOKE-MLIR (TEST_PLAN §2b): checks the MLIR/LLVM link+JIT path works,
# before any real lowering code is built on it. This is not a CTest
# case. It has no checks beyond "did it crash." Run it directly.
add_executable(smoke_mlir tests/smoke/smoke_mlir.cpp)
target_include_directories(smoke_mlir SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(smoke_mlir PRIVATE MLIR MLIRExecutionEngineShared)
add_test(NAME SMOKE-MLIR COMMAND smoke_mlir)

# LOW-E2E-MLP: checks that MlirLowerer's matmul/linear/relu lowering
# gives the right numbers, through real JIT execution (not just a
# golden-MLIR-text check).
add_executable(test_mlp_e2e tests/codegen/test_mlp_e2e.cpp)
target_include_directories(test_mlp_e2e SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_mlp_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
add_test(NAME LOW-E2E-MLP COMMAND test_mlp_e2e)

# LOOP-001: TilingDecision is plain C++. It needs no MLIR headers here.
add_executable(test_tiling_decision tests/codegen/test_tiling_decision.cpp)
target_link_libraries(test_tiling_decision PRIVATE tensora_codegen)
target_include_directories(test_tiling_decision PRIVATE ${CMAKE_SOURCE_DIR}/tests)
foreach(case_id LOOP-001-A LOOP-001-B LOOP-001-C LOOP-001-D LOOP-001-E LOOP-001-F)
  add_test(NAME ${case_id} COMMAND test_tiling_decision ${case_id})
endforeach()

# LOOP-002/003/004: the project-authored MLIR tiling pass. Checks its
# structure, checks that its numbers match the untiled version, and
# reports timing.
add_executable(test_tiling_e2e tests/codegen/test_tiling_e2e.cpp)
target_include_directories(test_tiling_e2e SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_tiling_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_tiling_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME LOOP-002 COMMAND test_tiling_e2e)

# LOOP-005: proves MlirLowerer's own internal LivenessAnalysis call
# (not a hand-built LivenessResult) drives TilingDecision's memory-
# pressure branch for real, through the real lowering path.
add_executable(test_liveness_tiling_e2e tests/codegen/test_liveness_tiling_e2e.cpp)
target_include_directories(test_liveness_tiling_e2e SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_liveness_tiling_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
add_test(NAME LOOP-005 COMMAND test_liveness_tiling_e2e)

# NATIVE-001: a real ahead-of-time-compiled executable. codegen_mlp_native
# lowers the same MLP TypedGraph as LOW-E2E-MLP, all the way to a real
# object file, through LLVM's TargetMachine (no JIT or ExecutionEngine
# involved). A small C driver, with no other dependency, is then built
# and linked against that object file by the system toolchain. This
# makes a standalone executable that CTest runs directly.
add_executable(codegen_mlp_native tools/codegen_mlp_native.cpp)
target_include_directories(codegen_mlp_native SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(codegen_mlp_native PRIVATE tensora_codegen MLIR LLVM)

# Reporting tool, not a CTest case: measures real wall-clock time for a
# few candidate matmul tile sizes so the tiling policy's defaults are an
# evidence-based choice, not an assumed one.
add_executable(benchmark_matmul_tiling tools/benchmark_matmul_tiling.cpp)
target_include_directories(benchmark_matmul_tiling SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(benchmark_matmul_tiling PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(benchmark_matmul_tiling PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")

set(MLP_NATIVE_OBJ ${CMAKE_CURRENT_BINARY_DIR}/mlp_native.o)
add_custom_command(
  OUTPUT ${MLP_NATIVE_OBJ}
  COMMAND codegen_mlp_native ${MLP_NATIVE_OBJ}
  DEPENDS codegen_mlp_native
  COMMENT "Emitting native object code for the MLP workload via Tensora's MLIR/LLVM backend")
add_custom_target(mlp_native_object DEPENDS ${MLP_NATIVE_OBJ})

add_executable(test_mlp_native tests/codegen/mlp_native_driver.c ${MLP_NATIVE_OBJ})
set_source_files_properties(${MLP_NATIVE_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
add_dependencies(test_mlp_native mlp_native_object)
target_link_libraries(test_mlp_native PRIVATE m)
add_test(NAME NATIVE-001 COMMAND test_mlp_native)

# LOW-006..009: transpose, flatten, softmax, and conv2d. Each op is
# checked on its own through JIT execution, before the CNN and
# attention end-to-end tests below combine them.
add_executable(test_new_ops_e2e tests/codegen/test_new_ops_e2e.cpp)
target_include_directories(test_new_ops_e2e SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_new_ops_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_new_ops_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME LOW-NEWOPS COMMAND test_new_ops_e2e)

# RANK-001..004: checks the numbers for rank-1/rank>=3 `linear` and
# rank>=2 (with broadcasting) `@` lowering. This closes a gap between
# the language and the backend: semantic analysis accepted ranks that
# MlirLowerer could not lower.
add_executable(test_rank_e2e tests/codegen/test_rank_e2e.cpp)
target_include_directories(test_rank_e2e SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_rank_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_rank_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME RANK-E2E COMMAND test_rank_e2e)

# E2E-002: CNN classifier (conv2d->relu->flatten->linear->softmax).
add_executable(test_cnn_e2e tests/codegen/test_cnn_e2e.cpp)
target_include_directories(test_cnn_e2e SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_cnn_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_cnn_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME E2E-002 COMMAND test_cnn_e2e)

# E2E-004: single-head scaled dot-product attention.
add_executable(test_attention_e2e tests/codegen/test_attention_e2e.cpp)
target_include_directories(test_attention_e2e SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_attention_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
target_compile_definitions(test_attention_e2e PRIVATE MLIR_C_RUNNER_UTILS_PATH="${MLIR_C_RUNNER_UTILS_LIB}")
add_test(NAME E2E-004 COMMAND test_attention_e2e)

# E2E-003: Transformer FFN sublayer (linear->relu->linear at FFN
# shapes). Reuses lowering already proven above. It adds no new ops.
add_executable(test_ffn_e2e tests/codegen/test_ffn_e2e.cpp)
target_include_directories(test_ffn_e2e SYSTEM PRIVATE ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
target_link_libraries(test_ffn_e2e PRIVATE tensora_codegen MLIR MLIRExecutionEngineShared)
add_test(NAME E2E-003 COMMAND test_ffn_e2e)

# INT-FULL-*: the real `tensora` CLI driver, run from source text all
# the way to JIT execution, against examples/*.tns. This differs from
# every e2e test above, which builds its TypedGraph by hand in C++.
# Expected values are checked by hand (see each .tns file's header
# comment) against all-ones synthetic input data, which --run always
# uses (Tensora source declares shapes, not values).
# INT-FULL-SYMBOLS: proof, with a golden output, that --emit-symbols
# prints SemanticAnalyzer's real symbol table (RM 3.6) for a real
# program: global input/weight declarations, graph-local bindings
# (including h's shadowing, which collapses to one entry), and the
# resolution log. The output is the same every run.
add_test(NAME INT-FULL-SYMBOLS COMMAND tensora --emit-symbols ${CMAKE_SOURCE_DIR}/examples/mlp.tns)
set_tests_properties(INT-FULL-SYMBOLS PROPERTIES PASS_REGULAR_EXPRESSION
  "x  input  Tensor\\[batch=B, feat=784\\]")

add_test(NAME INT-FULL-MLP-CHECK COMMAND tensora --check ${CMAKE_SOURCE_DIR}/examples/mlp_concrete.tns)
add_test(NAME INT-FULL-MLP-RUN COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/mlp_concrete.tns)
set_tests_properties(INT-FULL-MLP-RUN PROPERTIES PASS_REGULAR_EXPRESSION "element\\[0\\] = 145")

add_test(NAME INT-FULL-MM-RUN COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/matmul_large.tns)
set_tests_properties(INT-FULL-MM-RUN PROPERTIES PASS_REGULAR_EXPRESSION "element\\[0\\] = 256")

add_test(NAME INT-FULL-CNN-RUN COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/cnn.tns)
set_tests_properties(INT-FULL-CNN-RUN PROPERTIES PASS_REGULAR_EXPRESSION "element\\[0\\] = 0.5")

add_test(NAME INT-FULL-ATTN-RUN COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/attention.tns)
set_tests_properties(INT-FULL-ATTN-RUN PROPERTIES PASS_REGULAR_EXPRESSION "element\\[0\\] = 1")

# This is a deliberate negative case. mlp.tns has a symbolic batch dim.
# --run must reject it with the documented limitation message. It must
# not crash, and it must not silently give a wrong answer.
add_test(NAME INT-FULL-SYMBOLIC-REJECT COMMAND tensora --run ${CMAKE_SOURCE_DIR}/examples/mlp.tns)
set_tests_properties(INT-FULL-SYMBOLIC-REJECT PROPERTIES WILL_FAIL TRUE)

# Category labels for `ctest -L <category>` filtering. These come from
# the same ID-prefix rule the C++ harness uses
# (tests/support/TestHarness.h categoryOf). This block sets the label
# once here, for every test already registered above, instead of
# repeating it per test.
get_property(ALL_TEST_NAMES DIRECTORY PROPERTY TESTS)
foreach(t ${ALL_TEST_NAMES})
  if(t MATCHES "^BND-")
    set(kind_label boundary)
  elseif(t MATCHES "^SMOKE")
    set(kind_label smoke)
  elseif(t MATCHES "^INT-")
    set(kind_label integration)
  elseif(t MATCHES "^(E2E-|LOW-|NATIVE-|LOOP-|LIVE-|RANK-)")
    set(kind_label integration)
  else()
    set(kind_label unit)
  endif()

  # Compiler-phase label (`ctest -L <phase>`), so a phase can be tested
  # in isolation the same way it can be built in isolation (the
  # `lexer`/`parser`/`semantic`/`optimization`/`codegen` CMake targets
  # below). Boundary-case IDs (BND-LEX-*, BND-MM-*, etc.) carry the
  # same phase prefix rule as the unit-test IDs they extend.
  if(t MATCHES "^(LEX-|BND-LEX-)")
    set(phase_label lexer)
  elseif(t MATCHES "^(PAR-|BND-PAR-)")
    set(phase_label parser)
  elseif(t MATCHES "^(SEM-|SYM-|ATT-|GR-|BND-MM-|BND-BC-|BND-RS-|BND-FL-|BND-CV-|BND-RD-|BND-TR-|BND-DIM-)")
    set(phase_label semantic)
  elseif(t MATCHES "^(OPT-|LIVE-)")
    set(phase_label optimization)
  elseif(t MATCHES "^(SMOKE|LOW-|LOOP-|NATIVE-|RANK-|E2E-|INT-)")
    set(phase_label codegen)
  else()
    set(phase_label unknown)
  endif()

  set_tests_properties(${t} PROPERTIES LABELS "${kind_label};${phase_label}")
endforeach()

# Named per-phase build targets (docs/development/BUILD.md "Build one
# phase at a time"). Each just depends on the executables that phase
# actually needs, so `cmake --build build --target lexer` (etc.)
# builds only that slice of the DAG instead of everything -- useful
# when submitting or demoing one phase at a time.
add_custom_target(lexer DEPENDS test_lexer)
add_custom_target(parser DEPENDS test_parser)
add_custom_target(semantic DEPENDS test_semantic test_graph)
add_custom_target(optimization DEPENDS test_passes)
# Named "backend", not "codegen": CMake reserves the target name
# "codegen" for its own CODEGEN custom-command feature (CMP0171).
add_custom_target(backend DEPENDS
  smoke_mlir test_mlp_e2e test_tiling_decision test_tiling_e2e
  test_liveness_tiling_e2e test_new_ops_e2e test_rank_e2e test_cnn_e2e
  test_attention_e2e test_ffn_e2e test_mlp_native codegen_mlp_native
  benchmark_matmul_tiling tensora)
add_custom_target(full DEPENDS lexer parser semantic optimization backend)
```

### `tools/setup-dev.sh`
```bash
#!/usr/bin/env bash
# One-time developer bootstrap. Checks the toolchain this repo actually
# needs (cmake, a C++20 compiler, Java, LLVM/MLIR 21 -- see
# docs/development/DEPENDENCIES.md) and prints the next commands to
# run. There is no git-hook or CI setup here: formatting and linting
# are a developer's own judgment call, not an enforced gate.
#
# Usage, once after cloning:
#
#   ./tools/setup-dev.sh
set -uo pipefail
cd "$(dirname "$0")/.."

./tools/check-deps.sh
status=$?

if [ "$status" -ne 0 ]; then
  echo
  echo "Install the missing dependencies above, then re-run ./tools/setup-dev.sh."
  exit "$status"
fi

echo
echo "All required dependencies are present. Next steps:"
echo "  cmake -S . -B build"
echo "  cmake --build build -j4"
echo "  ctest --test-dir build --output-on-failure"
```

### `tools/check-deps.sh`
```bash
#!/usr/bin/env bash
# The closest C++-project equivalent to a Python requirements.txt:
# there is no single manifest file, because the dependencies come from
# several different ecosystems (OS packages, LLVM/MLIR, Java; ANTLR is
# CMake-managed and does not belong here at all -- see below). This
# script is the check, not the installer: it reports what is present,
# what is missing, and the exact command to fix each gap, rather than
# blindly running `sudo apt install` on a machine it does not control.
# See docs/development/DEPENDENCIES.md for the full pinned-version
# reference this script checks against.
#
# Usage: ./tools/check-deps.sh
# Exit status: 0 if every dependency is present, 1 otherwise.

set -uo pipefail
cd "$(dirname "$0")/.."

ok=0
missing=0

pass() { printf '  [ok]      %s\n' "$1"; ok=$((ok + 1)); }
fail() { printf '  [MISSING] %s\n' "$1"; missing=$((missing + 1)); }

echo "=== Required: build toolchain ==="

if command -v cmake >/dev/null 2>&1; then
  pass "cmake ($(cmake --version | head -1))"
else
  fail "cmake -- install with: sudo apt install -y cmake"
fi

if command -v g++ >/dev/null 2>&1; then
  cxx_version=$(g++ -dumpversion)
  if [ "${cxx_version%%.*}" -ge 10 ] 2>/dev/null; then
    pass "g++ ($cxx_version, supports C++20 per CMakeLists.txt CMAKE_CXX_STANDARD)"
  else
    fail "g++ $cxx_version is too old for C++20 -- install a newer one, e.g.: sudo apt install -y g++-13"
  fi
else
  fail "g++ -- install with: sudo apt install -y build-essential"
fi

if command -v java >/dev/null 2>&1; then
  pass "java ($(java -version 2>&1 | head -1)) -- runs the ANTLR generator jar CMake fetches"
else
  fail "java -- install with: sudo apt install -y default-jre"
fi

echo
echo "=== Required: LLVM/MLIR 21.1.8 (pinned, INTEG-003) ==="

if command -v llvm-config-21 >/dev/null 2>&1; then
  pass "llvm-config-21 ($(llvm-config-21 --version))"
else
  fail "llvm-21-dev -- install with: sudo apt install -y llvm-21-dev"
fi

if [ -d /usr/lib/llvm-21/lib/cmake/mlir ]; then
  pass "MLIR 21 CMake package (/usr/lib/llvm-21/lib/cmake/mlir)"
else
  fail "libmlir-21-dev -- install with: sudo apt install -y libmlir-21-dev mlir-21-tools"
fi

if [ -f /usr/lib/llvm-21/lib/libmlir_c_runner_utils.so ]; then
  pass "libmlir_c_runner_utils.so (needed by every JIT-executed test)"
else
  fail "libmlir_c_runner_utils.so not found -- reinstall libmlir-21-dev"
fi

echo
echo "=== Not checked here on purpose ==="
echo "  ANTLR (jar + C++ runtime): fetched reproducibly by CMake's"
echo "  FetchContent on first configure, not an OS package -- do not"
echo "  install it system-wide."

echo
echo "$ok ok, $missing missing"
if [ "$missing" -gt 0 ]; then
  echo "Fix the [MISSING] items above before running cmake -S . -B build."
  exit 1
fi
exit 0
```

### `tools/regen_appendix.py`
````python
#!/usr/bin/env python3
"""Regenerate docs/development/10_APPENDIX_SOURCE_LISTING.md.

The appendix is a mechanical concatenation of the real source tree
(see its own Section 10.10), not a hand-maintained document. Run this
after any change to a file the appendix already lists, so the listing
never silently drifts from what was actually built and tested:

    ./tools/regen_appendix.py

What it does:
  - Refreshes every "### `path`" file dump already in the appendix
    (Sections 10.3-10.9) from the file's current content on disk,
    tagged with the language matching its extension.
  - Regenerates the Section 10.2 repository manifest (`find grammar
    include src tools examples -type f | sort`) and the Section 10.9
    test manifest (`find tests -type f | sort`) from scratch.
  - Updates Section 10.10's base commit hash and generation date.
  - Reports, but does not silently fix, two situations that need a
    human decision: a manifest entry with no "### `path`" dump
    anywhere (a newly added file -- decide which of Sections 10.3-10.8
    it belongs in), and a "### `path`" dump whose file no longer
    exists on disk (a removed or renamed file -- decide whether to
    delete the section or update the path).

This script does not add or remove "### `path`" headers itself: the
grouping into Sections 10.3-10.8 is a curated-by-topic breakdown (one
section per compiler phase), not something a manifest diff can infer
on its own.
"""
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APPENDIX = REPO / "docs/development/10_APPENDIX_SOURCE_LISTING.md"

LANG_BY_EXT = {
    ".h": "cpp",
    ".hpp": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".g4": "antlr",
    ".cmake": "cmake",
    ".txt": "text",
    ".sh": "bash",
    ".py": "python",
    ".json": "json",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".md": "markdown",
    ".tns": "text",
}


def lang_for(path: str) -> str:
    if path.rsplit("/", 1)[-1] == "CMakeLists.txt":
        return "cmake"
    for ext, lang in LANG_BY_EXT.items():
        if path.endswith(ext):
            return lang
    return "text"


def find_sorted(*dirs: str) -> list[str]:
    out = subprocess.run(
        ["find", *dirs, "-type", "f", "-not", "-path", "*/__pycache__/*"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    return sorted(out.splitlines())


_CLOSE_FENCE_RE = re.compile(r"^`{3,}[ \t]*$", re.MULTILINE)


def fence_marker_for(content: str) -> str:
    """A backtick run one longer than any run already in `content`, so the
    fence itself can never be confused with a literal ``` inside content
    that itself talks about markdown fences (this script's own source,
    dumped into the appendix, is exactly that case)."""
    runs = re.findall(r"`+", content)
    longest = max((len(r) for r in runs), default=0)
    return "`" * max(3, longest + 1)


def replace_fenced_block(text: str, start: int, new_lines: list[str]) -> tuple[str, int]:
    fence_start = text.index("```", start)
    line_end = text.index("\n", fence_start)
    body_start = line_end + 1
    close_match = _CLOSE_FENCE_RE.search(text, body_start)
    fence_end = close_match.end()
    new_body = "\n".join(new_lines)
    marker = fence_marker_for(new_body)
    replaced_block = f"{marker}text\n{new_body}\n{marker}"
    replaced = text[:fence_start] + replaced_block + text[fence_end:]
    return replaced, fence_start + len(replaced_block)


def refresh_file_dumps(text: str, end_idx: int) -> tuple[str, int, list[str]]:
    body, rest = text[:end_idx], text[end_idx:]
    header_re = re.compile(r"^### `([^`]+)`.*$", re.MULTILINE)
    matches = list(header_re.finditer(body))
    missing_on_disk = []
    for m in reversed(matches):
        rel_path = m.group(1)
        full_path = REPO / rel_path
        if not full_path.is_file():
            missing_on_disk.append(rel_path)
            continue
        content = full_path.read_text().rstrip("\n")
        lang = lang_for(rel_path)
        marker = fence_marker_for(content)
        header_end = m.end()
        fence_start = body.index("```", header_end)
        line_end = body.index("\n", fence_start)
        close_match = _CLOSE_FENCE_RE.search(body, line_end + 1)
        body = body[:fence_start] + f"{marker}{lang}\n{content}\n{marker}" + body[close_match.end() :]
    return body + rest, len(matches), sorted(missing_on_disk)


def find_heading(text: str, heading: str) -> int:
    """Locate `heading` as a whole line, not a substring match. A plain
    `text.index(heading)` would also match this same heading text
    quoted inside this script's own docstring/source once this script
    dumps itself into the appendix (Section 10.8) -- anchoring to a
    full line rules that false match out, since the quoted occurrences
    are substrings of a longer source-code line, never a whole line by
    themselves."""
    m = re.search(r"^" + re.escape(heading) + r"$", text, re.MULTILINE)
    if m is None:
        raise ValueError(f"heading not found as a whole line: {heading!r}")
    return m.start()


def main() -> int:
    text = APPENDIX.read_text()
    end_idx = find_heading(text, "## 10.10 Mechanical Generation")
    text, n_dumps, missing_on_disk = refresh_file_dumps(text, end_idx)

    repo_manifest = [f"tensora/{p}" for p in find_sorted("grammar", "include", "src", "tools", "examples")]
    test_manifest = [f"tensora/{p}" for p in find_sorted("tests")]

    manifest_start = find_heading(text, "## 10.2 Repository Manifest")
    text, _ = replace_fenced_block(text, manifest_start, repo_manifest)

    test_manifest_start = find_heading(text, "## 10.9 Test Listing")
    text, _ = replace_fenced_block(text, test_manifest_start, test_manifest)

    end_idx = find_heading(text, "## 10.10 Mechanical Generation")
    dumped_paths = {m.group(1) for m in re.finditer(r"^### `([^`]+)`.*$", text[:end_idx], re.MULTILINE)}
    manifest_paths = {p.removeprefix("tensora/") for p in repo_manifest}
    undumped = sorted(manifest_paths - dumped_paths)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()
    text = re.sub(r"Base commit at generation time:\n`[0-9a-f]+`", f"Base commit at generation time:\n`{commit}`", text)
    text = re.sub(r"Generation\ndate: \d{4}-\d{2}-\d{2}", f"Generation\ndate: {date.today().isoformat()}", text)
    text = re.sub(r"Generation date: \d{4}-\d{2}-\d{2}", f"Generation date: {date.today().isoformat()}", text)

    APPENDIX.write_text(text)

    print(f"Refreshed {n_dumps} file dumps.")
    print(f"Repository manifest: {len(repo_manifest)} files. Test manifest: {len(test_manifest)} files.")
    if missing_on_disk:
        print(f"\nNEEDS ATTENTION -- {len(missing_on_disk)} dumped path(s) no longer exist on disk:")
        for p in missing_on_disk:
            print(f"  {p}")
    if undumped:
        print(f"\nNEEDS ATTENTION -- {len(undumped)} manifest path(s) have no ### dump in Sections 10.3-10.8 yet:")
        for p in undumped:
            print(f"  {p}")
    if missing_on_disk or undumped:
        print("\nPlace/remove these by hand in the right Section 10.3-10.8, then re-run this script.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

## 10.9 Test Listing

```text
tensora/tests/codegen/mlp_native_driver.c
tensora/tests/codegen/test_attention_e2e.cpp
tensora/tests/codegen/test_cnn_e2e.cpp
tensora/tests/codegen/test_ffn_e2e.cpp
tensora/tests/codegen/test_liveness_tiling_e2e.cpp
tensora/tests/codegen/test_mlp_e2e.cpp
tensora/tests/codegen/test_new_ops_e2e.cpp
tensora/tests/codegen/test_rank_e2e.cpp
tensora/tests/codegen/test_tiling_decision.cpp
tensora/tests/codegen/test_tiling_e2e.cpp
tensora/tests/lexer/test_lexer.cpp
tensora/tests/optimization/test_passes.cpp
tensora/tests/parser/test_parser.cpp
tensora/tests/semantic/test_graph.cpp
tensora/tests/semantic/test_semantic.cpp
tensora/tests/smoke/smoke_mlir.cpp
tensora/tests/support/TestHarness.h
```

### `tests/optimization/test_passes.cpp` (representative Level-1 pass test)
```cpp
// Level-1 graph optimization tests (RM 3.21), matching
// docs/testing/TEST_MATRIX.md OPT-*. These use hand-built GraphBuilder
// fixtures (ARCH-008), not the full source pipeline. This is fast and
// precise, and it is what 08_TEST_PLAN.md recommends for optimization
// tests.
#include <string>
#include <vector>

#include "tensora/optimization/Liveness.h"
#include "tensora/optimization/GraphPass.h"
#include "support/TestHarness.h"

using namespace tensora;
using tensora::testing::Reporter;
#define CASE TEST_CASE

namespace {

const TensorType* t2(TypeContext& types, const char* n0, int64_t s0, const char* n1, int64_t s1) {
  return types.tensor({Axis{n0, Dimension::makeConst(s0)}, Axis{n1, Dimension::makeConst(s1)}});
}

bool hasFact(const TypedGraph& g, NodeId id, const std::string& fact) { return g.node(id).facts.has(fact); }

}  // namespace

// --- OPT-SF-001: ShapeFolding eliminates a no-op reshape. ---
CASE(OPT_SF_001, "OPT-SF-001", "ShapeFolding: a reshape to the same shape is eliminated") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrs;
  attrs.reshapeTarget = tt->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrs, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createShapeFoldingPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "the no-op reshape is folded away");
  R.check(g.nodes().size() == 1 && g.nodes()[0].op == OpKind::Relu, "only relu remains");
  R.check(g.nodes()[0].inputs[0] == x, "relu now reads x directly");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_SF_002, "OPT-SF-002", "ShapeFolding near-miss: a reshape that actually changes shape is kept") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  const TensorType* tt2 = types.tensor({Axis{"c", Dimension::makeConst(6)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrs;
  attrs.reshapeTarget = tt2->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrs, tt2, {}, "v1");
  b.setOutput(v1);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createShapeFoldingPass()->run(g, ctx);
  R.check(!r.changed, "a real shape change is not folded");
  R.check(g.nodes().size() == 1, "the reshape node remains");
}

// --- OPT-DCE-001/002: DeadOpElimination drops unreached values, keeps reached ones. ---
CASE(OPT_DCE_001, "OPT-DCE-001", "DeadOpElimination drops a value not reaching return") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId dead = b.emit(OpKind::Relu, {x}, {}, tt, {}, "dead");
  ValueId live = b.emit(OpKind::Relu, {x}, {}, tt, {}, "live");
  (void)dead;
  b.setOutput(live);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createDeadOpEliminationPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "one dead node removed");
  R.check(g.nodes().size() == 1, "only the live node remains");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_DCE_002, "OPT-DCE-002", "DeadOpElimination near-miss: a value reaching return is kept") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createDeadOpEliminationPass()->run(g, ctx);
  R.check(!r.changed, "nothing is dead when every node reaches return");
  R.check(g.nodes().size() == 2, "both nodes remain");
}

// --- OPT-DCE-003: a whole dead chain (not just one dead node) is removed,
// while a same-shaped live chain right next to it is kept -- e.g.
// `h = linear(x, W1, b1); a = relu(h); y = linear(x, W2, b2); return a`:
// `y` is dead on its own, but so is any chain built only on top of it.
CASE(OPT_DCE_003, "OPT-DCE-003", "DeadOpElimination removes a whole dead chain, not just one dead node") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  // Live chain: x -> live1 -> live2, and live2 is returned.
  ValueId live1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "live1");
  ValueId live2 = b.emit(OpKind::Relu, {live1}, {}, tt, {}, "live2");
  // Dead chain: x -> dead1 -> dead2. Nothing ever reads dead2, and
  // dead2 is not returned, so neither dead1 nor dead2 reaches return.
  ValueId dead1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "dead1");
  ValueId dead2 = b.emit(OpKind::Relu, {dead1}, {}, tt, {}, "dead2");
  (void)dead2;
  b.setOutput(live2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createDeadOpEliminationPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 2, "both dead nodes are removed in one pass, not just the first");
  R.check(g.nodes().size() == 2, "only the two live nodes remain");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

// --- OPT-RC-001: ReshapeCancellation merges a chain into one reshape. ---
CASE(OPT_RC_001, "OPT-RC-001", "ReshapeCancellation merges reshape(reshape(x,A),B) into reshape(x,B)") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  const TensorType* ttA = types.tensor({Axis{"c", Dimension::makeConst(6)}});
  const TensorType* ttB = types.tensor({Axis{"d", Dimension::makeConst(2)}, Axis{"e", Dimension::makeConst(3)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrsA;
  attrsA.reshapeTarget = ttA->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrsA, ttA, {}, "v1");
  NodeAttrs attrsB;
  attrsB.reshapeTarget = ttB->axes();
  ValueId v2 = b.emit(OpKind::Reshape, {v1}, attrsB, ttB, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createReshapeCancellationPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "the inner reshape is fused away");
  R.check(g.nodes().size() == 1, "one reshape node remains");
  if (g.nodes().size() == 1) {
    R.check(g.nodes()[0].inputs[0] == x, "the single reshape now reads x directly");
    R.check(g.nodes()[0].attrs.reshapeTarget.size() == 2, "its target is B (the outer target)");
  }
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_RC_002, "OPT-RC-002", "ReshapeCancellation near-miss: inner reshape used elsewhere is kept") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  const TensorType* ttA = types.tensor({Axis{"c", Dimension::makeConst(6)}});
  const TensorType* ttB = types.tensor({Axis{"d", Dimension::makeConst(2)}, Axis{"e", Dimension::makeConst(3)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrsA;
  attrsA.reshapeTarget = ttA->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrsA, ttA, {}, "v1");
  NodeAttrs attrsB;
  attrsB.reshapeTarget = ttB->axes();
  ValueId v2 = b.emit(OpKind::Reshape, {v1}, attrsB, ttB, {}, "v2");
  ValueId v3 = b.emit(OpKind::Relu, {v1}, {}, ttA, {}, "v3");  // second consumer of v1
  ValueId v4 = b.emit(OpKind::Add, {v2}, NodeAttrs{}, ttB, {}, "v4");
  (void)v3;
  b.setOutput(v4);
  // v2 = Add(v2_reshape) needs a tensor operand for Add; simplify: just
  // verify the pass leaves v1 alone because it has two consumers (v2, v3).
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createReshapeCancellationPass()->run(g, ctx);
  R.check(!r.changed, "inner reshape has two consumers, so it cannot be dropped");
  R.check(g.nodes().size() == 4, "all four nodes remain");
}

// --- OPT-RC-003: interaction with ShapeFolding (reshape immediately undone). ---
CASE(OPT_RC_003, "OPT-RC-003",
     "interaction: ReshapeCancellation + ShapeFolding together eliminate reshape-then-undo") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  const TensorType* ttFlat = types.tensor({Axis{"c", Dimension::makeConst(6)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs attrsFlat;
  attrsFlat.reshapeTarget = ttFlat->axes();
  ValueId v1 = b.emit(OpKind::Reshape, {x}, attrsFlat, ttFlat, {}, "v1");
  NodeAttrs attrsBack;
  attrsBack.reshapeTarget = tt->axes();  // reshape right back to x's original shape
  ValueId v2 = b.emit(OpKind::Reshape, {v1}, attrsBack, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassManager pm;
  pm.add(createReshapeCancellationPass());
  pm.add(createShapeFoldingPass());
  PassResult r = pm.runAll(g, ctx);
  R.check(r.changed, "the pipeline changes the graph");
  R.check(g.nodes().empty(), "both reshapes fully cancel: merge to reshape(x,tt), then fold away as a no-op");
  R.check(g.output == x, "the graph's output is x directly");
}

// --- OPT-TC-001: TransposeCancellation fully cancels inverse perms. ---
CASE(OPT_TC_001, "OPT-TC-001", "TransposeCancellation: inverse perms cancel completely") {
  TypeContext types;
  const TensorType* tt = types.tensor(
      {Axis{"a", Dimension::makeConst(1)}, Axis{"b", Dimension::makeConst(2)}, Axis{"c", Dimension::makeConst(3)}});
  const TensorType* ttSwapped = types.tensor(
      {Axis{"c", Dimension::makeConst(3)}, Axis{"b", Dimension::makeConst(2)}, Axis{"a", Dimension::makeConst(1)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs p1;
  p1.perm = {2, 1, 0};
  ValueId v1 = b.emit(OpKind::Transpose, {x}, p1, ttSwapped, {}, "v1");
  NodeAttrs p2;
  p2.perm = {2, 1, 0};  // its own inverse
  ValueId v2 = b.emit(OpKind::Transpose, {v1}, p2, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createTransposeCancellationPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 2, "both transpose nodes are removed");
  R.check(g.nodes().empty(), "no nodes remain");
  R.check(g.output == x, "output is x directly");
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_TC_002, "OPT-TC-002", "TransposeCancellation: non-inverse perms compose into one transpose") {
  TypeContext types;
  const TensorType* tt = types.tensor(
      {Axis{"a", Dimension::makeConst(1)}, Axis{"b", Dimension::makeConst(2)}, Axis{"c", Dimension::makeConst(3)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs p1;
  p1.perm = {1, 2, 0};
  ValueId v1 = b.emit(OpKind::Transpose, {x}, p1, tt, {}, "v1");
  NodeAttrs p2;
  p2.perm = {1, 2, 0};  // same perm twice != identity when composed (a 3-cycle squared is another 3-cycle)
  ValueId v2 = b.emit(OpKind::Transpose, {v1}, p2, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createTransposeCancellationPass()->run(g, ctx);
  R.check(r.changed, "the pair still merges into one transpose (a real optimization, fewer ops)");
  R.check(g.nodes().size() == 1, "one transpose node remains");
  if (g.nodes().size() == 1) {
    R.check(g.nodes()[0].inputs[0] == x, "reads x directly");
    R.check(g.nodes()[0].attrs.perm == std::vector<uint32_t>{2, 0, 1}, "perm is the correct composition");
  }
}

CASE(OPT_TC_003, "OPT-TC-003", "TransposeCancellation near-miss: inner transpose used elsewhere is kept") {
  TypeContext types;
  const TensorType* tt = types.tensor(
      {Axis{"a", Dimension::makeConst(1)}, Axis{"b", Dimension::makeConst(2)}, Axis{"c", Dimension::makeConst(3)}});
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs p1;
  p1.perm = {2, 1, 0};
  ValueId v1 = b.emit(OpKind::Transpose, {x}, p1, tt, {}, "v1");
  NodeAttrs p2;
  p2.perm = {2, 1, 0};
  ValueId v2 = b.emit(OpKind::Transpose, {v1}, p2, tt, {}, "v2");
  ValueId v3 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v3");  // second use of v1
  ValueId v4 = b.emit(OpKind::Add, {v2}, NodeAttrs{}, tt, {}, "v4");
  (void)v3;
  b.setOutput(v4);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createTransposeCancellationPass()->run(g, ctx);
  R.check(!r.changed, "v1 has two consumers, so the pair cannot be touched");
  R.check(g.nodes().size() == 4, "all four nodes remain");
}

// --- OPT-EF-001: ElementwiseFusion tags a chain. ---
CASE(OPT_EF_001, "OPT-EF-001", "ElementwiseFusion tags a chain of >= 2 elementwise ops") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs a1;
  a1.scalar = 2.0;
  ValueId v1 = b.emit(OpKind::Add, {x}, a1, tt, {}, "v1");
  NodeAttrs a2;
  a2.scalar = 3.0;
  ValueId v2 = b.emit(OpKind::Mul, {v1}, a2, tt, {}, "v2");
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createElementwiseFusionPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "one fusion group recognized");
  R.check(g.nodes().size() == 2, "the pass only tags; it does not restructure the graph");
  R.check(hasFact(g, 0, "fusion_group_0") && hasFact(g, 1, "fusion_group_0"), "both nodes share one fact tag");
}

CASE(OPT_EF_002, "OPT-EF-002", "ElementwiseFusion near-miss: a lone elementwise op is not tagged") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  NodeAttrs a1;
  a1.scalar = 2.0;
  ValueId v1 = b.emit(OpKind::Add, {x}, a1, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");  // breaks the elementwise chain
  b.setOutput(v2);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createElementwiseFusionPass()->run(g, ctx);
  R.check(!r.changed, "a chain of length 1 (Add alone) is not a fusion group");
  R.check(!hasFact(g, 0, "fusion_group_0"), "no fact recorded");
}

// --- OPT-LBA-001/002: LinearBiasActivationFusion. ---
CASE(OPT_LBA_001, "OPT-LBA-001", "LinearBiasActivationFusion tags linear immediately followed by relu") {
  TypeContext types;
  const TensorType* x = t2(types, "batch", 2, "in", 3);
  const TensorType* w = t2(types, "in", 3, "out", 4);
  const TensorType* bias = types.tensor({Axis{"out", Dimension::makeConst(4)}});
  const TensorType* y = t2(types, "batch", 2, "out", 4);
  GraphBuilder b("g");
  ValueId xId = b.addInput("x", x, {});
  ValueId wId = b.addWeight("w", w, {});
  ValueId bId = b.addWeight("b", bias, {});
  ValueId lin = b.emit(OpKind::Linear, {xId, wId, bId}, {}, y, {}, "lin");
  ValueId relu = b.emit(OpKind::Relu, {lin}, {}, y, {}, "relu");
  b.setOutput(relu);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createLinearBiasActivationFusionPass()->run(g, ctx);
  R.check(r.changed && r.stats.numChanges == 1, "one fusion pair recognized");
  R.check(hasFact(g, 0, "fused_linear_relu_0") && hasFact(g, 1, "fused_linear_relu_0"),
          "both linear and relu share one fact tag");
}

CASE(OPT_LBA_002, "OPT-LBA-002", "LinearBiasActivationFusion near-miss: linear+softmax is not tagged") {
  TypeContext types;
  const TensorType* x = t2(types, "batch", 2, "in", 3);
  const TensorType* w = t2(types, "in", 3, "out", 4);
  const TensorType* bias = types.tensor({Axis{"out", Dimension::makeConst(4)}});
  const TensorType* y = t2(types, "batch", 2, "out", 4);
  GraphBuilder b("g");
  ValueId xId = b.addInput("x", x, {});
  ValueId wId = b.addWeight("w", w, {});
  ValueId bId = b.addWeight("b", bias, {});
  ValueId lin = b.emit(OpKind::Linear, {xId, wId, bId}, {}, y, {}, "lin");
  NodeAttrs smAttrs;
  smAttrs.axisName = "out";
  ValueId sm = b.emit(OpKind::Softmax, {lin}, smAttrs, y, {}, "sm");
  b.setOutput(sm);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassResult r = createLinearBiasActivationFusionPass()->run(g, ctx);
  R.check(!r.changed, "linear followed by softmax (not relu) is not a recognized fusion pattern");
}

// --- OPT-FIX-001: one sweep of DeadOpElimination can expose a fusion
// opportunity a later sweep must pick up. Graph: linear "lin" feeds two
// consumers, "dead" (a relu that never reaches return) and "live" (a
// relu that does). Because "lin" has two consumers, LinearBiasActivation-
// Fusion cannot tag it in the first sweep -- RM 3.9's fused-region
// legality needs the Linear's result read exactly once, since fusing it
// with a second, different consumer's activation would be wrong.
// DeadOpElimination removes "dead", dropping "lin" to one consumer, so a
// second sweep's LinearBiasActivationFusion call now tags "lin"+"live".
// A PassManager that runs every pass exactly once, in order, cannot see
// this: DeadOpElimination runs after the fusion passes in one sweep, so
// the opportunity would appear one sweep too late for a single pass.
CASE(OPT_FIX_001, "OPT-FIX-001",
     "DeadOpElimination removing a linear's dead consumer lets a later sweep fuse the surviving relu") {
  TypeContext types;
  const TensorType* x = t2(types, "batch", 2, "in", 3);
  const TensorType* w = t2(types, "in", 3, "out", 4);
  const TensorType* bias = types.tensor({Axis{"out", Dimension::makeConst(4)}});
  const TensorType* y = t2(types, "batch", 2, "out", 4);
  GraphBuilder b("g");
  ValueId xId = b.addInput("x", x, {});
  ValueId wId = b.addWeight("w", w, {});
  ValueId bId = b.addWeight("b", bias, {});
  ValueId lin = b.emit(OpKind::Linear, {xId, wId, bId}, {}, y, {}, "lin");
  ValueId dead = b.emit(OpKind::Relu, {lin}, {}, y, {}, "dead");  // never reaches return
  ValueId live = b.emit(OpKind::Relu, {lin}, {}, y, {}, "live");
  (void)dead;
  b.setOutput(live);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  PassManager pm;
  pm.add(createElementwiseFusionPass());
  pm.add(createLinearBiasActivationFusionPass());
  pm.add(createDeadOpEliminationPass());
  PassResult r = pm.runAll(g, ctx);

  R.check(r.changed, "the pipeline changes the graph across sweeps");
  R.check(r.stats.iterations >= 2, "reaching the fixed point took more than one sweep over the pass list");
  R.check(g.nodes().size() == 2, "\"dead\" is gone; only \"lin\" and \"live\" remain");
  if (g.nodes().size() == 2) {
    R.check(hasFact(g, 0, "fused_linear_relu_0") && hasFact(g, 1, "fused_linear_relu_0"),
            "lin+live are tagged fused, even though that was not possible in the first sweep");
  }
  GraphVerifier verifier;
  R.check(verifier.verify(g).empty(), "GraphVerifier-clean");
}

CASE(OPT_FIX_002, "OPT-FIX-002",
     "near-miss: a single sweep (no re-run) leaves the same pair untagged") {
  TypeContext types;
  const TensorType* x = t2(types, "batch", 2, "in", 3);
  const TensorType* w = t2(types, "in", 3, "out", 4);
  const TensorType* bias = types.tensor({Axis{"out", Dimension::makeConst(4)}});
  const TensorType* y = t2(types, "batch", 2, "out", 4);
  GraphBuilder b("g");
  ValueId xId = b.addInput("x", x, {});
  ValueId wId = b.addWeight("w", w, {});
  ValueId bId = b.addWeight("b", bias, {});
  ValueId lin = b.emit(OpKind::Linear, {xId, wId, bId}, {}, y, {}, "lin");
  ValueId dead = b.emit(OpKind::Relu, {lin}, {}, y, {}, "dead");
  ValueId live = b.emit(OpKind::Relu, {lin}, {}, y, {}, "live");
  (void)dead;
  b.setOutput(live);
  TypedGraph g = b.finish();

  DiagnosticEngine d;
  PassContext ctx{types, d};
  createLinearBiasActivationFusionPass()->run(g, ctx);  // one sweep only, matching the old single-pass behavior
  createDeadOpEliminationPass()->run(g, ctx);
  R.check(!hasFact(g, 0, "fused_linear_relu_0"),
          "without a second sweep, \"lin\" still has two consumers when fusion runs, so it is never tagged");
}

// --- LIVE-001: live intervals on a %1->%2->%3 chain. ---
CASE(LIVE_001, "LIVE-001", "live intervals on a %1->%2->%3 chain") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {v1}, {}, tt, {}, "v2");
  ValueId v3 = b.emit(OpKind::Relu, {v2}, {}, tt, {}, "v3");
  b.setOutput(v3);
  TypedGraph g = b.finish();

  LivenessResult live = LivenessAnalysis().run(g);
  R.check(live.intervals.at(x).firstDef == kNoProducer && live.intervals.at(x).lastUse == 0,
          "x: live from graph entry to node 0 (v1's only use of it)");
  R.check(live.intervals.at(v1).firstDef == 0 && live.intervals.at(v1).lastUse == 1,
          "v1: defined at node 0, used at node 1");
  R.check(live.intervals.at(v2).firstDef == 1 && live.intervals.at(v2).lastUse == 2,
          "v2: defined at node 1, used at node 2");
  R.check(live.intervals.at(v3).firstDef == 2 && live.intervals.at(v3).lastUse == 3,
          "v3: defined at node 2, used at the virtual `return` position (numNodes=3)");
}

// --- LIVE-002: byte size of Tensor[4,256] = 4*256*4. ---
CASE(LIVE_002, "LIVE-002", "byte size of Tensor[batch=4,feat=256] is 4*256*4 = 4096 bytes") {
  TypeContext types;
  const TensorType* tt = t2(types, "batch", 4, "feat", 256);
  R.check(tensorByteSize(tt) == 4 * 256 * 4, "4*256 elements * 4 bytes/f32 = 4096");
}

// --- LIVE-003: peak-memory = max simultaneously-live, matches a hand calculation. ---
CASE(LIVE_003, "LIVE-003", "peak-memory equals the hand-computed max of simultaneously-live sizes") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);  // 2*3*4 = 24 bytes each
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");   // node 0, consumes x
  ValueId v2 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v2");   // node 1, also consumes x
  ValueId v3 = b.emit(OpKind::Add, {v1, v2}, {}, tt, {}, "v3");  // node 2, consumes both v1 and v2
  b.setOutput(v3);
  TypedGraph g = b.finish();

  LivenessResult live = LivenessAnalysis().run(g);
  // At node 2 (computing v3), v1 and v2 must both still be live (its
  // operands) and v3 itself becomes live: 3 tensors * 24 bytes = 72.
  // x is already dead by then (last used at node 1).
  R.check(live.peakLiveBytes == 3 * 24, "peak = 3 * 24 = 72 bytes (v1, v2, v3 simultaneously live)");
}

// --- LIVE-004: a decision consumes peakLiveBytes. ---
CASE(LIVE_004, "LIVE-004", "a real decision (exceedsMemoryBudget) consumes peakLiveBytes") {
  TypeContext types;
  const TensorType* tt = t2(types, "a", 2, "b", 3);
  GraphBuilder b("g");
  ValueId x = b.addInput("x", tt, {});
  ValueId v1 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v1");
  ValueId v2 = b.emit(OpKind::Relu, {x}, {}, tt, {}, "v2");
  ValueId v3 = b.emit(OpKind::Add, {v1, v2}, {}, tt, {}, "v3");
  b.setOutput(v3);
  TypedGraph g = b.finish();

  LivenessResult live = LivenessAnalysis().run(g);  // peak = 72 bytes (LIVE-003)
  R.check(exceedsMemoryBudget(live, 50), "a 50-byte budget is exceeded by a 72-byte peak");
  R.check(!exceedsMemoryBudget(live, 100), "a 100-byte budget is not exceeded by a 72-byte peak");
}

#undef CASE

int main(int argc, char** argv) { return tensora::testing::runRegisteredCases(argc, argv); }
```

### `tests/codegen/test_cnn_e2e.cpp` (representative MLIR lowering JIT e2e test)
```cpp
// E2E-002: a small CNN classifier (conv2d -> relu -> flatten -> linear
// -> softmax). This combines LOW-001/003/007/008/009 into one real
// workload. The test lowers it through MlirLowerer, runs it with JIT
// execution, and checks the result against a hand-computed reference.
// This is the CNN counterpart to LOW-E2E-MLP.
//
// Shapes: x: 1x1x4x4 (NCHW) -> conv2d(1x1x2x2 kernel, stride=2,
// padding=0, bias=0) -> 1x1x2x2 -> relu (no-op, all positive) ->
// flatten -> 1x4 -> linear(W: 4x2, b: 2) -> 1x2 -> softmax(axis
// "classes") -> 1x2.
//
// x = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]; kernel = all
// ones (2x2); stride=2 means the two 2x2 windows don't overlap:
// conv = [[1+2+5+6, 3+4+7+8], [9+10+13+14, 11+12+15+16]]
//      = [[14, 22], [46, 54]]
// flatten (row-major over c,h,w) = [14, 22, 46, 54]
// W (each input feature routes to exactly one of 2 classes, scaled by
// 0.01): row0=[0.01,0], row1=[0,0.01], row2=[0.01,0], row3=[0,0.01]
// linear = [0.01*(14+46), 0.01*(22+54)] = [0.60, 0.76]
// softmax([0.60, 0.76]): e^0.60=1.822119, e^0.76=2.138264,
// sum=3.960382 -> [0.460098, 0.539902]
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MlirLowerer.h"
#include "tensora/optimization/GraphPass.h"

namespace tns = tensora;
using namespace mlir;
using MlirValue = mlir::Value;

namespace {

MlirValue constTensor(OpBuilder& b, Location loc, RankedTensorType type, ArrayRef<float> values) {
  auto attr = DenseElementsAttr::get(type, values);
  return arith::ConstantOp::create(b, loc, type, attr);
}

tns::TypedProgram buildCnnGraph(tns::TypeContext& types) {
  tns::GraphBuilder builder("cnn");
  using tns::Axis;
  using tns::Dimension;

  const tns::TensorType* xT = types.tensor({Axis{"n", Dimension::makeConst(1)}, Axis{"c", Dimension::makeConst(1)},
                                            Axis{"h", Dimension::makeConst(4)}, Axis{"w", Dimension::makeConst(4)}});
  const tns::TensorType* convWT = types.tensor({Axis{"oc", Dimension::makeConst(1)}, Axis{"ic", Dimension::makeConst(1)},
                                                Axis{"kh", Dimension::makeConst(2)}, Axis{"kw", Dimension::makeConst(2)}});
  const tns::TensorType* convBT = types.tensor({Axis{"oc", Dimension::makeConst(1)}});
  const tns::TensorType* convOutT =
      types.tensor({Axis{"n", Dimension::makeConst(1)}, Axis{"c", Dimension::makeConst(1)},
                     Axis{"h", Dimension::makeConst(2)}, Axis{"w", Dimension::makeConst(2)}});
  const tns::TensorType* flatT =
      types.tensor({Axis{"n", Dimension::makeConst(1)}, Axis{"flat", Dimension::makeConst(4)}});
  const tns::TensorType* fcWT =
      types.tensor({Axis{"flat", Dimension::makeConst(4)}, Axis{"classes", Dimension::makeConst(2)}});
  const tns::TensorType* fcBT = types.tensor({Axis{"classes", Dimension::makeConst(2)}});
  const tns::TensorType* logitsT =
      types.tensor({Axis{"n", Dimension::makeConst(1)}, Axis{"classes", Dimension::makeConst(2)}});

  tns::ValueId x = builder.addInput("x", xT, {});
  tns::ValueId convW = builder.addWeight("convW", convWT, {});
  tns::ValueId convB = builder.addWeight("convB", convBT, {});
  tns::ValueId fcW = builder.addWeight("fcW", fcWT, {});
  tns::ValueId fcB = builder.addWeight("fcB", fcBT, {});

  tns::NodeAttrs convAttrs;
  convAttrs.stride = 2;
  convAttrs.padding = 0;
  tns::ValueId conv = builder.emit(tns::OpKind::Conv2D, {x, convW, convB}, convAttrs, convOutT, {}, "conv");
  tns::ValueId reluOut = builder.emit(tns::OpKind::Relu, {conv}, {}, convOutT, {}, "relu");
  tns::ValueId flat = builder.emit(tns::OpKind::Flatten, {reluOut}, {}, flatT, {}, "flat");
  tns::ValueId logits = builder.emit(tns::OpKind::Linear, {flat, fcW, fcB}, {}, logitsT, {}, "logits");
  tns::NodeAttrs smAttrs;
  smAttrs.axisName = "classes";
  tns::ValueId probs = builder.emit(tns::OpKind::Softmax, {logits}, smAttrs, logitsT, {}, "probs");
  builder.setOutput(probs);
  tns::TypedGraph graph = builder.finish();

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));
  return program;
}

}  // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  tns::TypeContext types;
  tns::TypedProgram program = buildCnnGraph(types);

  tns::GraphVerifier verifier;
  if (!verifier.verify(program.graphs().front()).empty()) {
    fprintf(stderr, "E2E-002 FAIL: hand-built CNN graph is not GraphVerifier-clean\n");
    return 1;
  }

  DialectRegistry registry = tns::makeTensoraDialectRegistry();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  tns::MlirLowerer lowerer(ctx);
  ModuleOp module = lowerer.lower(program);
  if (failed(verify(module))) {
    fprintf(stderr, "E2E-002 FAIL: lowered CNN module does not verify\n");
    module.dump();
    return 1;
  }

  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  b.setInsertionPointToEnd(module.getBody());
  auto f32 = b.getF32Type();
  auto harness = func::FuncOp::create(b, loc, "test_main", b.getFunctionType({}, {f32}));
  Block* entry = harness.addEntryBlock();
  OpBuilder hb(entry, entry->begin());

  MlirValue x = constTensor(hb, loc, RankedTensorType::get({1, 1, 4, 4}, f32),
                            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  MlirValue convW = constTensor(hb, loc, RankedTensorType::get({1, 1, 2, 2}, f32), {1, 1, 1, 1});
  MlirValue convB = constTensor(hb, loc, RankedTensorType::get({1}, f32), {0.0f});
  MlirValue fcW = constTensor(hb, loc, RankedTensorType::get({4, 2}, f32),
                              {0.01f, 0.0f, 0.0f, 0.01f, 0.01f, 0.0f, 0.0f, 0.01f});
  MlirValue fcB = constTensor(hb, loc, RankedTensorType::get({2}, f32), {0.0f, 0.0f});

  auto fn = module.lookupSymbol<func::FuncOp>("cnn");
  auto call = func::CallOp::create(hb, loc, fn, ValueRange{x, convW, convB, fcW, fcB});
  MlirValue c0 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(0));
  MlirValue c1 = arith::ConstantOp::create(hb, loc, hb.getIndexType(), hb.getIndexAttr(1));
  MlirValue elem = tensor::ExtractOp::create(hb, loc, call.getResult(0), ValueRange{c0, c1});
  func::ReturnOp::create(hb, loc, ValueRange{elem});

  if (failed(verify(module))) {
    fprintf(stderr, "E2E-002 FAIL: module with harness does not verify\n");
    module.dump();
    return 1;
  }

  mlir::PassManager pm(&ctx);
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.allowReturnAllocsFromLoops = true;
  bufOpts.functionBoundaryTypeConversion = bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(module))) {
    fprintf(stderr, "E2E-002 FAIL: standard lowering pipeline failed\n");
    module.dump();
    return 1;
  }

  ExecutionEngineOptions engineOpts;
  static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
  static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
  engineOpts.sharedLibPaths = kSharedLibs;
  auto maybeEngine = ExecutionEngine::create(module, engineOpts);
  if (!maybeEngine) {
    fprintf(stderr, "E2E-002 FAIL: ExecutionEngine::create failed\n");
    return 1;
  }
  auto engine = std::move(*maybeEngine);
  auto maybeFn = engine->lookup("test_main");
  if (!maybeFn) {
    fprintf(stderr, "E2E-002 FAIL: could not look up 'test_main'\n");
    return 1;
  }
  auto testMain = reinterpret_cast<float (*)()>(*maybeFn);
  float result = testMain();

  float expected = 0.539902f;
  if (std::fabs(result - expected) > 1e-4f) {
    fprintf(stderr, "E2E-002 FAIL: expected %f, got %f\n", expected, result);
    return 1;
  }
  printf(
      "E2E-002 PASS: CNN(conv2d->relu->flatten->linear->softmax) class-1 probability = %f (expected %f)\n", result,
      expected);
  return 0;
}
```

### `tests/codegen/test_tiling_e2e.cpp` (Level-2 loop-tiling test)
```cpp
// LOOP-002/003/004: checks the structure and the numbers of the
// project-authored MatmulTilingPass, applied to a real
// MlirLowerer-produced module. It also reports timing, but does not
// assert a timing value.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

#include "tensora/codegen/DialectSetup.h"
#include "tensora/codegen/MatmulTilingPass.h"
#include "tensora/codegen/MlirLowerer.h"

namespace tns = tensora;
using namespace mlir;
using MlirValue = mlir::Value;

namespace {

// Large enough to show a real, dramatic tiled-vs-untiled speedup (cache
// blocking actually matters at this size, not just JIT-call noise);
// still small enough to fill/verify by hand (every output element =
// kDim for all-ones operands) and to add only a few seconds to `ctest`,
// not minutes.
constexpr int64_t kDim = 1024;

tns::TypedProgram buildMatmulProgram(tns::TypeContext& types) {
  tns::GraphBuilder builder("mm");
  const tns::TensorType* aT =
      types.tensor({tns::Axis{"m", tns::Dimension::makeConst(kDim)}, tns::Axis{"k", tns::Dimension::makeConst(kDim)}});
  const tns::TensorType* bT =
      types.tensor({tns::Axis{"k", tns::Dimension::makeConst(kDim)}, tns::Axis{"n", tns::Dimension::makeConst(kDim)}});
  const tns::TensorType* cT =
      types.tensor({tns::Axis{"m", tns::Dimension::makeConst(kDim)}, tns::Axis{"n", tns::Dimension::makeConst(kDim)}});
  tns::ValueId a = builder.addInput("a", aT, {});
  tns::ValueId bIn = builder.addInput("b", bT, {});
  tns::ValueId c = builder.emit(tns::OpKind::MatMul, {a, bIn}, {}, cT, {}, "c");
  builder.setOutput(c);
  tns::TypedGraph graph = builder.finish();

  tns::TypedProgram program(types);
  program.addGraph(std::move(graph));
  return program;
}

// Builds a StridedMemRefType view over caller-owned, row-major kDim x
// kDim data -- the layout the standard lowering pipeline's LLVM-level
// ABI expects for a tensor<kDim x kDim x f32> argument/result.
StridedMemRefType<float, 2> makeMemRef(float* data) {
  StridedMemRefType<float, 2> m;
  m.basePtr = data;
  m.data = data;
  m.offset = 0;
  m.sizes[0] = kDim;
  m.sizes[1] = kDim;
  m.strides[0] = kDim;
  m.strides[1] = 1;
  return m;
}

// Returns {result, wall_time_seconds}, or {NAN, 0} on failure. Calls the
// lowered "mm" function directly through its emitted C interface
// (_mlir_ciface_mm), passing memref descriptors for two all-ones kDim x
// kDim operands -- this avoids routing through a second, calling
// function entirely, so the module under test (and under
// bufferization) is always the single function MlirLowerer produced
// plus whatever MatmulTilingPass did to it, nothing more.
std::pair<float, double> runPipeline(ModuleOp module, MLIRContext& ctx, bool applyTiling, bool* sawTiledLoop) {
  if (applyTiling) {
    PassManager tilingPm(&ctx);
    // 128 is a reasonable cache-blocking tile size for a 1024x1024x1024
    // matmul (8 tiles per dimension), not just the smallest size that
    // still exercises the pass.
    tilingPm.addNestedPass<func::FuncOp>(
        tns::createMatmulTilingPass(tns::TilingPolicy{/*minDimToTile=*/128, /*maxTileSize=*/128}));
    if (failed(tilingPm.run(module))) {
      fprintf(stderr, "tiling pass failed\n");
      return {NAN, 0.0};
    }
  }
  if (sawTiledLoop) {
    *sawTiledLoop = false;
    module.walk([&](scf::ForOp) { *sawTiledLoop = true; });
  }

  PassManager pm(&ctx);
  bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.allowReturnAllocsFromLoops = true;  // tiling's scf.for loops carry tensor iter_args/results
  // The default InferLayoutMap tries to compute a precise (possibly
  // static-stride) layout per function boundary; the subview-derived
  // layouts inside a tiled loop nest defeat that inference. Forcing
  // fully-dynamic-strided memrefs at every function boundary sidesteps
  // the inference entirely (still correct, just less precise typing).
  bufOpts.functionBoundaryTypeConversion = bufferization::LayoutMapOption::FullyDynamicLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  // memref.subview (introduced by tiling) lowers to LLVM only after its
  // offset/stride metadata has been expanded into explicit index
  // arithmetic; this must run *before* ArithToLLVM so that the
  // newly-introduced index-typed arithmetic gets converted to i64 in the
  // same sweep as everything else, instead of being stranded afterward.
  pm.addPass(memref::createExpandStridedMetadataPass());
  // ExpandStridedMetadata computes subview offsets via affine.apply;
  // that needs lowering to arith before ArithToLLVM can see it.
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(module))) {
    fprintf(stderr, "standard lowering pipeline failed\n");
    module.dump();
    return {NAN, 0.0};
  }

  // Tiling's per-tile memref.copy on non-identity-layout subviews lowers
  // to a call to the `memrefCopy` runtime helper (a strided element-wise
  // copy loop; not something the LLVM conversion inlines), which lives
  // in MLIR's own runner-support shared library, not in this process.
  ExecutionEngineOptions engineOpts;
  static const std::string kRunnerUtils = MLIR_C_RUNNER_UTILS_PATH;
  static const llvm::SmallVector<llvm::StringRef, 1> kSharedLibs = {kRunnerUtils};
  engineOpts.sharedLibPaths = kSharedLibs;
  auto maybeEngine = ExecutionEngine::create(module, engineOpts);
  if (!maybeEngine) {
    fprintf(stderr, "ExecutionEngine::create failed\n");
    module.dump();
    return {NAN, 0.0};
  }
  auto engine = std::move(*maybeEngine);
  auto maybeFn = engine->lookup("_mlir_ciface_mm");
  if (!maybeFn) {
    fprintf(stderr, "could not look up '_mlir_ciface_mm'\n");
    return {NAN, 0.0};
  }
  using CIfaceMatmul = void (*)(StridedMemRefType<float, 2>*, StridedMemRefType<float, 2>*,
                                 StridedMemRefType<float, 2>*);
  auto fn = reinterpret_cast<CIfaceMatmul>(*maybeFn);

  std::vector<float> aData(kDim * kDim, 1.0f);
  std::vector<float> bData(kDim * kDim, 1.0f);
  auto aMem = makeMemRef(aData.data());
  auto bMem = makeMemRef(bData.data());
  // "mm" returns its result by value; the ciface wrapper mallocs the
  // backing storage itself and overwrites this descriptor's fields (data
  // /basePtr included) with the new allocation -- this is purely an
  // out-parameter for the wrapper to fill in, not caller-owned storage.
  StridedMemRefType<float, 2> cMem = {};

  auto t0 = std::chrono::steady_clock::now();
  fn(&cMem, &aMem, &bMem);
  auto t1 = std::chrono::steady_clock::now();
  float result = cMem.data[cMem.offset];
  std::free(cMem.basePtr);
  return {result, std::chrono::duration<double>(t1 - t0).count()};
}

}  // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  const float expected = static_cast<float>(kDim);  // all-ones 1024x1024 @ 1024x1024 -> every element = 1024

  // --- Untiled baseline. ---
  tns::TypeContext types1;
  DialectRegistry registry1 = tns::makeTensoraDialectRegistry();
  MLIRContext ctx1(registry1);
  ctx1.loadAllAvailableDialects();
  tns::MlirLowerer lowerer1(ctx1);
  ModuleOp moduleUntiled = lowerer1.lower(buildMatmulProgram(types1));
  if (failed(verify(moduleUntiled))) {
    fprintf(stderr, "LOOP FAIL: untiled module does not verify\n");
    return 1;
  }
  bool untiledSawLoop = false;
  auto [untiledResult, untiledTime] = runPipeline(moduleUntiled, ctx1, /*applyTiling=*/false, &untiledSawLoop);

  // --- Tiled (MatmulTilingPass applied). ---
  tns::TypeContext types2;
  DialectRegistry registry2 = tns::makeTensoraDialectRegistry();
  MLIRContext ctx2(registry2);
  ctx2.loadAllAvailableDialects();
  tns::MlirLowerer lowerer2(ctx2);
  ModuleOp moduleTiled = lowerer2.lower(buildMatmulProgram(types2));
  if (failed(verify(moduleTiled))) {
    fprintf(stderr, "LOOP FAIL: tiled module does not verify\n");
    return 1;
  }
  bool tiledSawLoop = false;
  auto [tiledResult, tiledTime] = runPipeline(moduleTiled, ctx2, /*applyTiling=*/true, &tiledSawLoop);

  bool ok = true;

  // LOOP-002: structural -- the tiled copy has scf.for loops from
  // tiling (introduced before the generic linalg->scf.for conversion
  // pass runs at all); the untiled copy has none at that point.
  if (!tiledSawLoop) {
    fprintf(stderr, "LOOP-002 FAIL: expected scf.for loops after MatmulTilingPass, found none\n");
    ok = false;
  }
  if (untiledSawLoop) {
    fprintf(stderr, "LOOP-002 FAIL: untiled copy unexpectedly has scf.for before the generic loop pass ran\n");
    ok = false;
  }

  // LOOP-003: numerical equivalence.
  if (untiledResult != expected) {
    fprintf(stderr, "LOOP-003 FAIL: untiled result %f != expected %f\n", untiledResult, expected);
    ok = false;
  }
  if (tiledResult != expected) {
    fprintf(stderr, "LOOP-003 FAIL: tiled result %f != expected %f\n", tiledResult, expected);
    ok = false;
  }
  if (tiledResult != untiledResult) {
    fprintf(stderr, "LOOP-003 FAIL: tiled (%f) != untiled (%f)\n", tiledResult, untiledResult);
    ok = false;
  }

  if (!ok) return 1;

  // LOOP-004: reported, not asserted. At 1024x1024x1024 the wall times
  // below are real, measurable numbers (seconds), not JIT-call noise --
  // but this single untuned run is still not a rigorous performance
  // claim on its own (no warmup, no repetition, no machine info).
  // `tools/benchmark_matmul_tiling.cpp` (best-of-5, sweeps 16/32/64) is
  // the actual rigorous measurement behind the tiling policy's default
  // tile size -- see docs/testing/TEST_MATRIX.md's Loop optimization
  // section for the measured numbers.
  printf(
      "LOOP-002/003/004 PASS: tiled and untiled both compute %.1f (expected %.1f); tiled has scf.for "
      "loops, untiled does not; wall time untiled=%.6fs tiled=%.6fs (workload=%lldx%lldx%lld, single run, "
      "not a rigorous performance claim)\n",
      untiledResult, expected, untiledTime, tiledTime, static_cast<long long>(kDim), static_cast<long long>(kDim),
      static_cast<long long>(kDim));
  return 0;
}
```

## 10.10 Mechanical Generation

This listing was generated by direct shell concatenation of the files
listed in the manifest above (`find grammar include src tools examples
-type f | sort`, then `cat` per file into a fenced code block tagged
with the language matching its extension (for example ` ```cpp ` for
`.cpp`/`.h`, ` ```bash ` for `.sh`, ` ```cmake ` for `CMakeLists.txt`);
the test manifest and three representative test files were generated
the same way), not by hand-transcription -- so it cannot silently
diverge from what was actually built and tested.

This repository is under git version control, hosted at
[github.com/Rmuk655/TensoraCompiler](https://github.com/Rmuk655/TensoraCompiler)
(`origin`, branch `main`). Base commit at generation time:
`4812b9742784e8378cad741de1ad71d5c6f2efbf`. The concatenation itself
reflects the working tree at generation time, which may include
changes made after that commit and not yet committed -- the manifest
and file contents above are the authoritative record either way, since
they are read directly from disk, not from git history. Generation
date: 2026-08-22. The 175 CTest cases passing at generation time are
the authoritative record of what this listing implements; see
`docs/testing/TEST_MATRIX.md` and `docs/development/CHANGELOG.md`.
