#pragma once
// Loads test cases at runtime from a fixture `.tns` file (see
// tests/fixtures/*.tns), instead of each case's source living as an
// escaped C++ string literal. A fixture file holds many small `.tns`
// programs, one per test case, as real Tensora source:
//
//     ### <ID> | <description>
//     ### expect: ok
//     <source, runs until the next '### ' line or end of file>
//
//     ### <ID> | <description>
//     ### expect: reject <DiagCode>
//     <source>
//
// registerFixtureCases() reads one such file and adds one Case to the
// shared registry (tests/support/TestHarness.h) per entry -- so a
// fixture case is indistinguishable at runtime from a hand-written
// TEST_CASE(...): it shows up in `<binary> --list`, runs on its own via
// `<binary> <ID>`, and gets its own CTest (cmake/Tests.cmake's
// tensora_discover_fixture_cases greps this same fixture file for its
// '### <ID> |' lines, so CMake's case list and the runtime registry
// never drift apart).
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "support/TestHarness.h"
#include "tensora/shared/Diagnostic.h"

namespace tensora::testing {

// The identifier written after "DiagCode::" in a fixture's
// '### expect: reject <Name>' line. Add an arm here whenever DiagCode
// (include/tensora/shared/Diagnostic.h) gains a member a fixture case
// needs to name -- this mapping exists only for fixture cases to
// compare a rendered name against expected text; production code never
// needs a DiagCode -> string conversion.
inline const char* diagCodeName(DiagCode c) {
  switch (c) {
#define TENSORA_DIAG_NAME(x) \
  case DiagCode::x: return #x;
    TENSORA_DIAG_NAME(LexIllegalChar)
    TENSORA_DIAG_NAME(LexIntegerOverflow)
    TENSORA_DIAG_NAME(ParseUnexpectedToken)
    TENSORA_DIAG_NAME(ParseExpected)
    TENSORA_DIAG_NAME(SemaUndefinedName)
    TENSORA_DIAG_NAME(SemaRedeclaration)
    TENSORA_DIAG_NAME(SemaArgCount)
    TENSORA_DIAG_NAME(SemaArgKind)
    TENSORA_DIAG_NAME(SemaTypeMismatch)
    TENSORA_DIAG_NAME(SemaRankMismatch)
    TENSORA_DIAG_NAME(SemaShapeMismatch)
    TENSORA_DIAG_NAME(SemaAxisNameMismatch)
    TENSORA_DIAG_NAME(SemaBroadcast)
    TENSORA_DIAG_NAME(SemaReshapeCount)
    TENSORA_DIAG_NAME(SemaTransposePerm)
    TENSORA_DIAG_NAME(SemaAxisNotFound)
    TENSORA_DIAG_NAME(SemaConvChannel)
    TENSORA_DIAG_NAME(SemaConvArithmetic)
    TENSORA_DIAG_NAME(SemaUnresolvableShape)
    TENSORA_DIAG_NAME(SemaNonPositiveDimension)
    TENSORA_DIAG_NAME(SemaDuplicateAxisName)
    TENSORA_DIAG_NAME(IrInvariantViolation)
#undef TENSORA_DIAG_NAME
  }
  return "<unknown DiagCode>";
}

struct FixtureEntry {
  std::string id;
  std::string desc;
  bool hasExpect = false;
  bool expectOk = false;
  std::string expectDiag;  // only meaningful when !expectOk
  std::string src;
};

// A '### expect:' line is optional. An entry without one is a
// source-only entry (see loadFixtureSource, below) -- not meant to be
// auto-registered as a case (registerFixtureCases rejects it).
inline std::vector<FixtureEntry> loadFixtureFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("FixtureLoader: cannot open " + path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) lines.push_back(line);

  auto startsWith = [](const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; };
  std::vector<FixtureEntry> entries;
  size_t i = 0;
  while (i < lines.size()) {
    if (!startsWith(lines[i], "### ") || startsWith(lines[i], "### expect:")) {
      ++i;
      continue;
    }
    std::string header = lines[i].substr(4);
    size_t sep = header.find(" | ");
    if (sep == std::string::npos) throw std::runtime_error("FixtureLoader: bad header line: " + lines[i]);
    FixtureEntry e;
    e.id = header.substr(0, sep);
    e.desc = header.substr(sep + 3);
    ++i;
    const std::string kExpectPrefix = "### expect: ";
    if (i < lines.size() && startsWith(lines[i], kExpectPrefix)) {
      e.hasExpect = true;
      std::string expect = lines[i].substr(kExpectPrefix.size());
      if (expect == "ok") {
        e.expectOk = true;
      } else if (startsWith(expect, "reject ")) {
        e.expectOk = false;
        e.expectDiag = expect.substr(std::string("reject ").size());
      } else {
        throw std::runtime_error("FixtureLoader: " + e.id + ": bad expect line: " + expect);
      }
      ++i;
    }
    std::ostringstream src;
    while (i < lines.size() && !startsWith(lines[i], "### ")) {
      src << lines[i] << "\n";
      ++i;
    }
    e.src = src.str();
    if (e.src.find_first_not_of(" \t\n") == std::string::npos)
      throw std::runtime_error("FixtureLoader: " + e.id + ": empty source body");
    entries.push_back(std::move(e));
  }
  if (entries.empty()) throw std::runtime_error("FixtureLoader: no cases found in " + path);
  return entries;
}

// Returns the raw .tns source for the single entry `id` in
// `fixturePath`. For hand-written CASE(...) bodies (e.g.
// tests/semantic/test_semantic.cpp, tests/semantic/test_graph.cpp,
// tests/parser/test_parser.cpp) that need a real C++ assertion on the
// parsed/analyzed result -- not just the ok/reject-plus-DiagCode check
// registerFixtureCases automates -- so the case body stays hand-written,
// but its .tns source still lives as one real Tensora program in a
// fixture file, not as an escaped C++ string literal.
inline std::string loadFixtureSource(const std::string& fixturePath, const std::string& id) {
  for (FixtureEntry& e : loadFixtureFile(fixturePath)) {
    if (e.id == id) return e.src;
  }
  throw std::runtime_error("FixtureLoader: no entry '" + id + "' in " + fixturePath);
}

// Registers every case in `fixturePath` into the shared harness registry,
// exactly like TEST_CASE(...) does for a hand-written case. `runPhase`
// runs one phase (parse, or parse+analyze) over a fixture entry's source
// and returns {accepted, firstDiagCodeName} -- the second element is
// only inspected when accepted is false.
inline void registerFixtureCases(const std::string& fixturePath,
                                  std::function<std::pair<bool, std::string>(const std::string&)> runPhase) {
  for (FixtureEntry& e : loadFixtureFile(fixturePath)) {
    if (!e.hasExpect) throw std::runtime_error("FixtureLoader: " + e.id + ": missing '### expect:' line");
    Category category = categoryOf(e.id);
    registry().push_back({e.id, e.desc, category, [e, runPhase](Reporter& R) {
                             auto [accepted, diag] = runPhase(e.src);
                             if (e.expectOk) {
                               R.check(accepted, "accepted as expected");
                             } else {
                               R.check(!accepted, "rejected as expected");
                               R.checkEqual(diag == e.expectDiag, "reports " + e.expectDiag, diag, e.expectDiag);
                             }
                           }});
  }
}

}  // namespace tensora::testing
