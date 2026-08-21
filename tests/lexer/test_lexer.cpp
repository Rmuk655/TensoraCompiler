// Phase 1 lexer tests (RM 3.1). These cover keywords, identifiers,
// literals, operators, delimiters, comments, whitespace and newline
// handling, illegal characters, and SourceSpan tracking. This file uses
// no test framework. Each rule is its own named case (LEX-*, BND-LEX-*),
// with the stable ID from docs/testing/TEST_MATRIX.md. Run the binary
// with no arguments to run every case. This prints a PASS/FAIL line for
// each case, then a summary. Run it with a case ID (`test_lexer
// LEX-003`) to run just that case. This is how CMakeLists.txt registers
// one CTest test per ID, so `ctest` reports which lexer rule failed,
// instead of one pass/fail result for the whole binary.
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "tensora/lexer/Lexer.h"
#include "support/TestHarness.h"

using namespace tensora;
using tensora::testing::Reporter;
#define CASE TEST_CASE

namespace {

// Tokens hold string_views into their SourceBuffer, so every buffer used
// by any case must outlive that case. A deque never moves its elements.
std::deque<SourceBuffer> g_buffers;

std::vector<Token> lex(const std::string& code, DiagnosticEngine& diags) {
  g_buffers.emplace_back("test.tns", code);
  Lexer lexer(g_buffers.back(), diags);
  return lexer.tokenize();
}

std::vector<TokenKind> kinds(const std::vector<Token>& toks) {
  std::vector<TokenKind> ks;
  ks.reserve(toks.size());
  for (const Token& t : toks) ks.push_back(t.kind);
  return ks;
}

// --- LEX-001: keywords, and identifiers including keyword-prefixed /
// underscore / digit-suffixed names (maximal munch: the identifier rule
// must win over a keyword rule whenever it matches more characters). ---
CASE(LEX_001, "LEX-001", "keywords and identifiers, incl. keyword-prefix maximal munch") {
  {
    DiagnosticEngine d;
    auto t = lex("input weight graph return Tensor", d);
    R.check(t.size() == 6, "five keywords plus eof");
    TokenKind want[] = {TokenKind::KwInput, TokenKind::KwWeight, TokenKind::KwGraph,
                        TokenKind::KwReturn, TokenKind::KwTensor, TokenKind::EndOfFile};
    for (int i = 0; i < 6; ++i) {
      R.check(t[i].kind == want[i], "keyword token " + std::to_string(i));
    }
    R.check(!d.hasErrors(), "keywords alone produce no diagnostics");
  }
  {
    DiagnosticEngine d;
    auto t = lex("inputs weightX graphed returning Tensor2 _leading trailing_ mixedCase123", d);
    R.check(t.size() == 9, "eight identifiers plus eof");
    for (int i = 0; i < 8; ++i) {
      R.check(t[i].kind == TokenKind::Identifier,
              "keyword-prefixed/underscore/digit name #" + std::to_string(i) + " is an identifier");
    }
    R.check(!d.hasErrors(), "identifiers alone produce no diagnostics");
  }
}

// --- LEX-002: integer and float literals, including a leading-zero
// integer and multiple fractional precisions, with value correctness. ---
CASE(LEX_002, "LEX-002", "int and float literal values, incl. leading zeros") {
  DiagnosticEngine d;
  auto t = lex("0 007 784 0.125 3.14159 100.0", d);
  R.check(t.size() == 7, "six literals plus eof");
  R.check(t[0].kind == TokenKind::IntLiteral && t[0].intValue == 0, "int 0");
  R.check(t[1].kind == TokenKind::IntLiteral && t[1].intValue == 7, "leading-zero int 007 == 7");
  R.check(t[2].kind == TokenKind::IntLiteral && t[2].intValue == 784, "int 784");
  R.check(t[3].kind == TokenKind::FloatLiteral && t[3].floatValue == 0.125, "float 0.125");
  R.check(t[4].kind == TokenKind::FloatLiteral && std::abs(t[4].floatValue - 3.14159) < 1e-9,
          "float 3.14159");
  R.check(t[5].kind == TokenKind::FloatLiteral && t[5].floatValue == 100.0, "float 100.0");
  R.check(!d.hasErrors(), "well-formed literals produce no diagnostics");
}

// --- LEX-003: every operator/delimiter, whitespace-separated. ---
CASE(LEX_003, "LEX-003", "every operator and delimiter, spaced") {
  DiagnosticEngine d;
  auto t = lex("@ + - * / = : , [ ] { } ( )", d);
  TokenKind want[] = {
      TokenKind::At,      TokenKind::Plus,     TokenKind::Minus,   TokenKind::Star,
      TokenKind::Slash,   TokenKind::Equal,    TokenKind::Colon,   TokenKind::Comma,
      TokenKind::LBracket, TokenKind::RBracket, TokenKind::LBrace, TokenKind::RBrace,
      TokenKind::LParen,  TokenKind::RParen};
  R.check(t.size() == 15, "fourteen operator/delimiter tokens plus eof");
  for (int i = 0; i < 14; ++i) {
    R.check(t[i].kind == want[i], "punctuation token " + std::to_string(i));
  }
  R.check(!d.hasErrors(), "operators/delimiters alone produce no diagnostics");
}

// --- LEX-004: the same tokens with no separating whitespace at all, to
// confirm single-char operators never merge into an unintended token
// (there is no multi-char operator in the grammar) and correctly split
// around identifiers/int literals on either side. ---
CASE(LEX_004, "LEX-004", "operators/delimiters with no whitespace (adjacency)") {
  DiagnosticEngine d;
  auto t = lex("[1,2]{a:b}(c,d)a+b-c*d/e@f", d);
  TokenKind want[] = {
      TokenKind::LBracket, TokenKind::IntLiteral, TokenKind::Comma,      TokenKind::IntLiteral,
      TokenKind::RBracket, TokenKind::LBrace,     TokenKind::Identifier, TokenKind::Colon,
      TokenKind::Identifier, TokenKind::RBrace,   TokenKind::LParen,     TokenKind::Identifier,
      TokenKind::Comma,     TokenKind::Identifier, TokenKind::RParen,    TokenKind::Identifier,
      TokenKind::Plus,      TokenKind::Identifier, TokenKind::Minus,     TokenKind::Identifier,
      TokenKind::Star,      TokenKind::Identifier, TokenKind::Slash,     TokenKind::Identifier,
      TokenKind::At,        TokenKind::Identifier};
  R.check(t.size() == 27, "twenty-six adjacent tokens plus eof");
  size_t n = std::min<size_t>(t.size() - 1, 26);
  for (size_t i = 0; i < n; ++i) {
    R.check(t[i].kind == want[i], "adjacent token " + std::to_string(i));
  }
  R.check(!d.hasErrors(), "adjacency alone produces no diagnostics");
}

// --- LEX-005: comment variants (full-line, empty, no-space-before,
// no-trailing-newline, back-to-back, and a comment whose text contains
// keywords/operators/illegal characters that must never be scanned). ---
CASE(LEX_005, "LEX-005", "comment variants are fully skipped") {
  {
    DiagnosticEngine d;
    auto t = lex("# a comment\ninput", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::KwInput, TokenKind::EndOfFile},
            "full-line comment then keyword on line 2");
    R.check(t[0].span.line == 2, "input is on line 2 after the comment line");
  }
  {
    DiagnosticEngine d;
    auto t = lex("#\ninput", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::KwInput, TokenKind::EndOfFile},
            "an empty comment (just '#') is still skipped");
  }
  {
    DiagnosticEngine d;
    auto t = lex("input# no space before comment\nweight", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::KwInput, TokenKind::KwWeight,
                                                TokenKind::EndOfFile},
            "'#' starts a comment even with no preceding space");
  }
  {
    DiagnosticEngine d;
    auto t = lex("input # trailing comment with no newline", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::KwInput, TokenKind::EndOfFile},
            "a comment reaching end of file with no trailing newline is skipped");
  }
  {
    DiagnosticEngine d;
    auto t = lex("# one\n# two\ninput", d);
    R.check(t[0].kind == TokenKind::KwInput && t[0].span.line == 3,
            "back-to-back comment lines are each skipped");
  }
  {
    DiagnosticEngine d;
    auto t = lex("# input @ $ ignored entirely\ngraph", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::KwGraph, TokenKind::EndOfFile},
            "keywords/operators/illegal chars inside a comment are never scanned");
    R.check(!d.hasErrors(), "an illegal char inside a comment produces no diagnostic");
  }
}

// --- LEX-006: whitespace and newlines are otherwise interchangeable
// separators; the resulting token *kind* sequence must be identical
// regardless of which separator is used. ---
CASE(LEX_006, "LEX-006", "whitespace/newline insensitivity across separator styles") {
  DiagnosticEngine d0;
  auto baseline = kinds(lex("input weight graph return Tensor", d0));

  DiagnosticEngine d1;
  auto newlines = kinds(lex("input\nweight\ngraph\nreturn\nTensor", d1));
  R.check(newlines == baseline, "newline-separated matches space-separated");

  DiagnosticEngine d2;
  auto tabs = kinds(lex("input\t\tweight\t\t\tgraph\treturn\tTensor", d2));
  R.check(tabs == baseline, "tab-separated matches space-separated");

  DiagnosticEngine d3;
  auto crlf = kinds(lex("input\r\nweight\r\ngraph\r\nreturn\r\nTensor", d3));
  R.check(crlf == baseline, "CRLF-separated matches space-separated");

  DiagnosticEngine d4;
  auto blankLines = kinds(lex("input\n\n\nweight\n\ngraph\nreturn\n\n\nTensor", d4));
  R.check(blankLines == baseline, "extra blank lines between tokens match space-separated");
}

// --- LEX-007: malformed numeric literals. The float rule requires a
// digit on both sides of the dot, so a stray '.' always falls through to
// the illegal-character catch-all rather than silently forming a bad
// float or merging with a neighboring literal/identifier. ---
CASE(LEX_007, "LEX-007", "malformed floats fall through to illegal-character, not silently accepted") {
  {
    DiagnosticEngine d;
    auto t = lex("3.", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::IntLiteral, TokenKind::Error,
                                                TokenKind::EndOfFile},
            "'3.' is int 3 then an illegal '.'");
    R.check(t[0].intValue == 3, "the int part is still read correctly");
    R.check(d.all().size() == 1 && d.all()[0].code == DiagCode::LexIllegalChar,
            "'3.' reports exactly one LexIllegalChar");
  }
  {
    DiagnosticEngine d;
    auto t = lex(".5", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::Error, TokenKind::IntLiteral,
                                                TokenKind::EndOfFile},
            "'.5' is an illegal '.' then int 5 (no int part before the dot)");
    R.check(t[1].intValue == 5, "the trailing int is still read correctly");
  }
  {
    DiagnosticEngine d;
    auto t = lex("3.x", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::IntLiteral, TokenKind::Error,
                                                TokenKind::Identifier, TokenKind::EndOfFile},
            "'3.x' is int 3, illegal '.', then identifier x");
  }
  {
    DiagnosticEngine d;
    auto t = lex("1..2", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::IntLiteral, TokenKind::Error,
                                                TokenKind::Error, TokenKind::IntLiteral,
                                                TokenKind::EndOfFile},
            "'1..2' is int 1, two illegal '.' in a row, then int 2");
    R.check(d.all().size() == 2, "'1..2' reports two diagnostics, one per illegal '.'");
    R.check(d.all()[0].span.col == 2 && d.all()[1].span.col == 3,
            "the two illegal '.' diagnostics land at consecutive columns");
  }
}

// --- LEX-008: illegal-character variety, including several consecutive
// illegal characters and an illegal character between two valid tokens
// (scanning continues past an error rather than aborting the file). ---
CASE(LEX_008, "LEX-008", "illegal-character variety and continued scanning") {
  {
    DiagnosticEngine d;
    auto t = lex("$?;\\~^!", d);
    R.check(t.size() == 8, "seven illegal chars plus eof");
    for (int i = 0; i < 7; ++i) {
      R.check(t[i].kind == TokenKind::Error, "illegal char #" + std::to_string(i));
    }
    R.check(d.all().size() == 7, "seven distinct illegal characters report seven diagnostics");
    for (const auto& diag : d.all()) {
      R.check(diag.code == DiagCode::LexIllegalChar, "every diagnostic here is LexIllegalChar");
    }
  }
  {
    DiagnosticEngine d;
    auto t = lex("a$b", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::Identifier, TokenKind::Error,
                                                TokenKind::Identifier, TokenKind::EndOfFile},
            "an illegal char between two identifiers does not swallow either neighbor");
    R.check(d.all().size() == 1, "exactly one diagnostic for the one illegal char");
  }
}

// --- LEX-009: SourceSpan / offset correctness across lines and tabs,
// plus the invariant that every token's lexeme is exactly the source
// slice named by its span. ---
CASE(LEX_009, "LEX-009", "SourceSpan line/col/offset tracking, incl. tabs and CRLF") {
  {
    DiagnosticEngine d;
    auto t = lex("ab\ncd", d);
    R.check(t[0].span.line == 1 && t[0].span.col == 1 && t[0].span.beginOffset == 0 &&
                 t[0].span.endOffset == 2,
             "'ab' spans line 1 col 1, offsets [0,2)");
    R.check(t[1].span.line == 2 && t[1].span.col == 1 && t[1].span.beginOffset == 3 &&
                 t[1].span.endOffset == 5,
             "'cd' spans line 2 col 1, offsets [3,5) (newline consumes offset 2)");
  }
  {
    DiagnosticEngine d;
    auto t = lex("a\tb", d);
    R.check(t[0].span.col == 1, "'a' is at column 1");
    R.check(t[1].span.col == 3 && t[1].span.beginOffset == 2,
            "'b' is at column 3 (a tab advances one column, like any other char)");
  }
  {
    DiagnosticEngine d;
    auto t = lex("a\r\nb", d);
    R.check(t[0].span.line == 1, "'a' is on line 1");
    R.check(t[1].span.line == 2 && t[1].span.col == 1,
            "'b' is on line 2 col 1 (CRLF is one line break)");
  }
  {
    DiagnosticEngine d;
    const std::string src = "x1 + y2";
    g_buffers.emplace_back("span.tns", src);
    Lexer lexer(g_buffers.back(), d);
    auto t = lexer.tokenize();
    for (const Token& tok : t) {
      if (tok.kind == TokenKind::EndOfFile) continue;
      std::string slice = src.substr(tok.span.beginOffset,
                                      tok.span.endOffset - tok.span.beginOffset);
      R.check(slice == std::string(tok.lexeme),
              "token lexeme '" + std::string(tok.lexeme) + "' matches source slice '" + slice + "'");
    }
  }
}

// --- LEX-010: a full representative multi-line program combining
// declarations, a graph body, comments, literals, and operators in one
// pass (a golden token-kind dump), verifying the exact token sequence
// end-to-end with zero diagnostics. ---
CASE(LEX_010, "LEX-010", "full representative program: golden token-kind dump") {
  const std::string src =
      "# a tiny representative program\n"
      "input x: Tensor[batch=B, feat=784]\n"
      "weight W1: Tensor[in=784, out=256]\n"
      "graph mlp {\n"
      "  h = x @ W1\n"
      "  h = relu(h)\n"
      "  return h\n"
      "}\n";
  DiagnosticEngine d;
  auto t = lex(src, d);

  using K = TokenKind;
  std::vector<K> want = {
      // input x: Tensor[batch=B, feat=784]
      K::KwInput, K::Identifier, K::Colon, K::KwTensor, K::LBracket, K::Identifier, K::Equal,
      K::Identifier, K::Comma, K::Identifier, K::Equal, K::IntLiteral, K::RBracket,
      // weight W1: Tensor[in=784, out=256]
      K::KwWeight, K::Identifier, K::Colon, K::KwTensor, K::LBracket, K::Identifier, K::Equal,
      K::IntLiteral, K::Comma, K::Identifier, K::Equal, K::IntLiteral, K::RBracket,
      // graph mlp {
      K::KwGraph, K::Identifier, K::LBrace,
      // h = x @ W1
      K::Identifier, K::Equal, K::Identifier, K::At, K::Identifier,
      // h = relu(h)
      K::Identifier, K::Equal, K::Identifier, K::LParen, K::Identifier, K::RParen,
      // return h
      K::KwReturn, K::Identifier,
      // }
      K::RBrace,
      K::EndOfFile};

  R.check(t.size() == want.size(),
          "token count is " + std::to_string(want.size()) + " (got " +
              std::to_string(t.size()) + ")");
  size_t n = std::min(t.size(), want.size());
  for (size_t i = 0; i < n; ++i) {
    R.check(t[i].kind == want[i], "program token " + std::to_string(i));
  }
  R.check(!d.hasErrors(), "the representative program produces no diagnostics");
  R.check(t[0].span.line == 2, "the leading comment line is fully skipped (first token on line 2)");
}

// --- LEX-011: intLiteral is bounded to a signed 64-bit integer
// (LANG-001, fixes KNOWN_BUGS BUG-001). Boundary pair: the literal
// exactly at INT64_MAX accepts; one past it, and a far-wider literal,
// reject with LexIntegerOverflow and no crash. ---
CASE(LEX_011, "LEX-011", "intLiteral bounded to int64 (LANG-001): max accepts, overflow rejects") {
  {
    DiagnosticEngine d;
    auto t = lex("9223372036854775807", d);
    R.check(t[0].kind == TokenKind::IntLiteral && t[0].intValue == 9223372036854775807LL,
            "INT64_MAX literal is accepted with the exact value");
    R.check(!d.hasErrors(), "INT64_MAX literal produces no diagnostics");
  }
  {
    DiagnosticEngine d;
    auto t = lex("9223372036854775808", d);
    R.check(kinds(t) == std::vector<TokenKind>{TokenKind::Error, TokenKind::EndOfFile},
            "INT64_MAX+1 is rejected as an Error token, not silently truncated");
    R.check(d.all().size() == 1 && d.all()[0].code == DiagCode::LexIntegerOverflow,
            "INT64_MAX+1 reports exactly one LexIntegerOverflow diagnostic");
  }
  {
    DiagnosticEngine d;
    bool threw = false;
    try {
      auto t = lex(std::string(40, '9'), d);
      R.check(kinds(t) == std::vector<TokenKind>{TokenKind::Error, TokenKind::EndOfFile},
              "a far-wider literal (40 nines) is also rejected as a single Error token");
      R.check(d.all().size() == 1 && d.all()[0].code == DiagCode::LexIntegerOverflow,
              "a far-wider literal reports LexIntegerOverflow, not a crash");
    } catch (const std::exception&) {
      threw = true;
    }
    R.check(!threw, "an int literal wider than int64 does not throw/crash the lexer");
  }
}

// --- BND-LEX-001/002: empty file and comments-only file, the minimum
// boundary of RM 3.1 (a file with no meaningful tokens at all). ---
CASE(BND_LEX_001, "BND-LEX-001", "empty file yields only eof") {
  DiagnosticEngine d;
  auto t = lex("", d);
  R.check(t.size() == 1 && t[0].kind == TokenKind::EndOfFile, "empty input yields only eof");
  R.check(t[0].span.line == 1 && t[0].span.col == 1, "eof is at line 1 col 1 for empty input");
}

CASE(BND_LEX_002, "BND-LEX-002", "comments-only file yields only eof") {
  DiagnosticEngine d;
  auto t = lex("# just a comment", d);
  R.check(t.size() == 1 && t[0].kind == TokenKind::EndOfFile,
          "single-comment input yields only eof");

  DiagnosticEngine d2;
  auto t2 = lex("# c1\n# c2\n# c3", d2);
  R.check(t2.size() == 1 && t2[0].kind == TokenKind::EndOfFile,
          "multiple comment lines yield only eof");
}

// --- BND-LEX-003: a token at end of file with no trailing newline, for
// one representative of each token category (identifier, keyword, int,
// float, operator). ---
CASE(BND_LEX_003, "BND-LEX-003", "a token at EOF with no trailing newline, per token category") {
  struct Sample {
    const char* src;
    TokenKind kind;
    const char* label;
  };
  Sample samples[] = {
      {"W", TokenKind::Identifier, "identifier"},
      {"graph", TokenKind::KwGraph, "keyword"},
      {"42", TokenKind::IntLiteral, "int literal"},
      {"1.5", TokenKind::FloatLiteral, "float literal"},
      {"@", TokenKind::At, "operator"},
  };
  for (const Sample& s : samples) {
    DiagnosticEngine d;
    auto t = lex(s.src, d);
    R.check(t.size() == 2 && t[0].kind == s.kind && t[1].kind == TokenKind::EndOfFile,
            std::string("a lone ") + s.label + " with no trailing newline is followed by eof");
  }
}

// --- BND-LEX-004: whatever kind of token starts the file, it always
// begins at line 1, column 1, offset 0. ---
CASE(BND_LEX_004, "BND-LEX-004", "the first token of any file starts at line 1, col 1, offset 0") {
  const char* starts[] = {"x", "3", "3.0", "@", "graph"};
  for (const char* s : starts) {
    DiagnosticEngine d;
    auto t = lex(s, d);
    R.check(t[0].span.line == 1 && t[0].span.col == 1 && t[0].span.beginOffset == 0,
            std::string("'") + s + "' starts at line 1, col 1, offset 0");
  }
}

#undef CASE

}  // namespace

int main(int argc, char** argv) { return tensora::testing::runRegisteredCases(argc, argv); }
