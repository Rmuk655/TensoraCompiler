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
