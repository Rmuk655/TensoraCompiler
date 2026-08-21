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
