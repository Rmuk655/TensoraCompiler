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
