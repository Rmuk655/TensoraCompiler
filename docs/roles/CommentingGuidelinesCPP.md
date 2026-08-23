# Tensora C++ Commenting Guidelines

## Purpose

These are permanent commenting guidelines for Tensora project-owned code.

All new or modified comments must follow ASD-STE100 Simplified Technical English principles as closely as practical.

The purpose of comments is to help a reader understand what the code is for, why it exists, and how it fits into the compiler.

Do not comment every line.

## 1. Main rule

A useful comment explains purpose, design, input, output, or a non-obvious reason.

Do not write a comment that only repeats the code.

Prefer:

```cpp
// Keep the source span so later phases can report an error at this location.
```

Avoid:

```cpp
// Set the span.
node.span = span;
```

## 2. File-level comments

Every important project-owned `.h`, `.hpp`, `.cpp`, `.cc`, `.g4`, and test source file must have a short file-level comment.

The comment should explain, when applicable:

1. What the file contains.
2. Which compiler phase uses it.
3. Why the file exists.
4. What important input it receives.
5. What important output it produces.
6. Which phase uses that output.

Example:

```cpp
// This file defines the lexer.
// The lexer reads source text and creates tokens.
// The parser reads these tokens.
```

Keep the file comment short.

Do not put a tutorial at the top of a source file.

## 3. Use Simplified Technical English

Use these ASD-STE100-style rules:

- Use short sentences.
- Use common words.
- Use active voice.
- Use one term for one concept.
- Keep the subject close to the verb.
- Avoid unnecessary synonyms.
- Avoid vague words.
- Avoid long noun groups.
- Avoid unnecessary parenthetical text.
- Explain an abbreviation or technical term when a new reader can misunderstand it.

Prefer:

```cpp
// The parser reads the tokens.
```

Avoid:

```cpp
// The token sequence is subsequently consumed by the syntactic analysis component.
```

Prefer:

```cpp
// Create the AST node for the function call.
```

Avoid:

```cpp
// Materialize the semantic representation of the parsed call construct.
```

## 4. Use active voice

Prefer:

```cpp
// The semantic analyzer checks the tensor shape.
```

Avoid:

```cpp
// The tensor shape is checked by the semantic analyzer.
```

Prefer:

```cpp
// The optimizer removes nodes that have no users.
```

Avoid:

```cpp
// Nodes that have no users are removed by the optimizer.
```

## 5. Explain technical terms once

Use necessary compiler terms, but explain them when useful.

Example:

```cpp
// The AST (abstract syntax tree) stores the source program structure.
```

Example:

```cpp
// The IR (intermediate representation) stores verified operations.
// Optimization and code generation use this representation.
```

Do not replace precise compiler terms with inaccurate everyday words.

## 6. Use consistent terminology

Use one name for one concept.

Important distinctions include:

```text
token      = one lexical item
parse tree = grammar structure created by the parser generator
AST        = project representation of source program structure
IR / TCG   = verified representation used by later compiler phases
```

Do not use `parse tree` and `AST` as synonyms when both exist.

Do not switch between multiple names for the same project concept without a reason.

## 7. Class and struct comments

Comment an important class or struct when its purpose is not obvious.

Explain:

- what it represents
- which phase creates or owns it
- which phase uses it

Example:

```cpp
// This structure stores one source token.
// The lexer creates it and the parser reads it.
struct Token {
    ...
};
```

Example:

```cpp
// This node represents a binary expression.
// Semantic analysis checks whether the operation is valid.
struct BinaryExpr {
    ...
};
```

Do not document every trivial field when its name and type are sufficient.

## 8. Public function comments

Important public functions should explain:

- what the function does
- what the input means when it is not obvious
- what it returns or changes
- important error behavior

Example:

```cpp
// Read the source text and create tokens in source order.
// Report lexical errors through the diagnostic engine.
std::vector<Token> tokenize();
```

Example:

```cpp
// Check the AST for semantic errors.
// Create the typed IR when the required checks succeed.
Result analyze(const AstProgram& program);
```

Keep these comments concise.

## 9. Private function comments

Do not automatically comment every private function.

Add a comment when the function name cannot explain the purpose or when the function implements an important compiler rule.

Example:

```cpp
// Check that both input shapes are valid for matrix multiplication.
bool checkMatMulShapes(...);
```

## 10. Explain compiler phase boundaries

Use comments to make data flow clear.

Examples:

```cpp
// The lexer creates tokens.
// It does not check tensor types.
```

```cpp
// The parser creates the AST.
// Semantic analysis performs type and shape checks later.
```

```cpp
// Semantic analysis creates the verified IR.
// The optimizer reads this IR.
```

```cpp
// Code generation converts the optimized IR to MLIR.
```

Do not repeat these comments in every function. Put them where they help a new reader understand the boundary.

## 11. Shared data comments

A shared type should explain why multiple phases use it.

Example:

```cpp
// This file defines source spans used by compiler diagnostics.
// The lexer creates spans and later phases preserve them.
```

Example:

```cpp
// This file defines the AST.
// The parser creates the AST and semantic analysis reads it.
```

Example:

```cpp
// This file defines the compiler IR.
// Semantic analysis creates the IR.
// Optimization and code generation use it.
```

## 12. Explain non-obvious design decisions

Comment the reason for an important design choice.

Example:

```cpp
// Keep the AST separate from the typed IR.
// The AST represents source syntax.
// The IR contains verified type and shape information.
```

Example:

```cpp
// Convert the parser location here so the rest of the compiler uses SourceSpan.
```

Do not explain an obvious language statement.

## 13. Diagnostics and recovery comments

Explain why recovery or error handling is necessary.

Examples:

```cpp
// Report the invalid token and continue so the lexer can find more errors.
```

```cpp
// Skip tokens until the parser reaches the next statement boundary.
```

```cpp
// Report overflow instead of storing a wrapped integer value.
```

Do not write comments such as:

```cpp
// Report the error.
reportError(...);
```

unless the reason for reporting at that location is not obvious.

## 14. Semantic analysis comments

Explain the language rule being checked.

Examples:

```cpp
// A name must be declared before the program uses it.
```

```cpp
// The inner dimensions must match for matrix multiplication.
```

```cpp
// Check that the call has the required number of arguments.
```

Do not hide a complicated semantic rule behind a vague comment such as:

```cpp
// Validate types.
```

## 15. Optimization comments

Explain the transformation and its safety condition.

Examples:

```cpp
// Remove the operation because no later operation uses its result.
```

```cpp
// Replace the constant expression with its computed value.
```

```cpp
// Do not fuse these operations because another node uses the intermediate value.
```

Avoid comments that only say:

```cpp
// Optimize the graph.
```

## 16. Code generation comments

Explain the direction of lowering.

Examples:

```cpp
// Convert this IR operation to the matching MLIR operation.
```

```cpp
// Lower the tensor operation to loops before LLVM code generation.
```

When needed:

```cpp
// MLIR is the backend intermediate representation.
// LLVM later converts the lowered program to native code.
```

## 17. ANTLR grammar comments

Keep grammar comments short.

Explain an important language rule or non-obvious restriction.

Example:

```antlr
// A program contains declarations followed by one or more graphs.
program
    : declaration* graphDecl+
    ;
```

Do not comment every grammar rule when the rule name already explains it.

Keep project design and AST construction details out of the grammar when they belong in C++ code.

## 18. Test comments

A test comment should explain the behavior or boundary that the test checks.

Examples:

```cpp
// Verify that the lexer accepts an identifier.
```

```cpp
// Verify that an integer larger than int64_t reports an overflow error.
```

```cpp
// Verify that incompatible matrix shapes fail semantic analysis.
```

Do not add comments such as:

```cpp
// Run the test.
```

Use clear test names so fewer comments are necessary.

## 19. Do not add boilerplate comments

Do not add comments only to increase comment count.

Bad:

```cpp
// This is a parser test.
// This test tests the parser.
// Run the parser.
```

Good:

```cpp
// Verify that a declaration after a graph is rejected.
```

## 20. Remove obsolete comments

When code changes, update or remove comments that describe:

- old tools
- old directory names
- removed classes
- removed APIs
- old data flow
- old ownership
- old behavior

For example, after a Flex/Bison migration, do not leave comments about Flex scan codes or Bison semantic values unless that technology still exists.

Comments must describe the current implementation.

## 21. Generated third-party code

Do not edit generated ANTLR, MLIR, LLVM, or other third-party generated files only to add project comments.

These guidelines apply to project-owned source and generated/scaffolded code that the team maintains.

## 22. Review checklist

Before finishing a code change, review all comments that you added or touched:

- [ ] The comment is necessary.
- [ ] The comment explains purpose, design, or a non-obvious rule.
- [ ] The sentence is short.
- [ ] The words are simple and precise.
- [ ] The sentence uses active voice where practical.
- [ ] The terminology matches the rest of the project.
- [ ] The comment does not repeat the code.
- [ ] The comment does not describe obsolete behavior.
- [ ] A beginner can understand the comment with minimal project context.

The code explains the implementation.
The comments explain the purpose and the reason.
