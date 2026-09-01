#pragma once
// Shared harness for the hand-rolled test executables. This uses no
// test framework. Each rule is a named case with a stable ID (matching
// docs/testing/TEST_MATRIX.md). You can run each case on its own. So
// both the standalone binary and CTest (one test per ID) report which
// rule failed, instead of one pass/fail result for the whole binary.
// Run with no arguments to run every registered case. Run with a case
// ID to run just that one case (this is how CMakeLists.txt registers
// one CTest test per ID).
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace tensora::testing {

// One of these five categories, matching docs/testing/08_TEST_PLAN.md
// Section 2. The category comes from the case ID's own naming
// convention, already used across every test file and TEST_MATRIX.md.
// So a test case never needs to state its own category by hand.
enum class Category { Unit, Smoke, Boundary, Integration, Performance };

inline const char* categoryName(Category c) {
  switch (c) {
    case Category::Unit: return "UNIT";
    case Category::Smoke: return "SMOKE";
    case Category::Boundary: return "BOUNDARY";
    case Category::Integration: return "INTEGRATION";
    case Category::Performance: return "PERFORMANCE";
  }
  return "UNIT";
}

inline bool startsWith(const std::string& s, const char* prefix) {
  return s.rfind(prefix, 0) == 0;
}

// BND-* is always Boundary, no matter which suite it belongs to
// (BND-LEX-*, BND-PAR-*, BND-MM-*, and so on). SMOKE-*/INT-* are named
// for their category directly. E2E-*/LOW-*/NATIVE-*/LOOP-*/LIVE-* all
// run real phases against real MLIR/LLVM, with no mocks. This matches
// the Integration definition in 08_TEST_PLAN.md Section 9a. Every other
// case is a Unit case: one rule, one grammar production, or one pass,
// checked on its own.
inline Category categoryOf(const std::string& id) {
  if (startsWith(id, "BND-")) return Category::Boundary;
  if (startsWith(id, "SMOKE")) return Category::Smoke;
  if (startsWith(id, "INT-")) return Category::Integration;
  if (startsWith(id, "E2E-") || startsWith(id, "LOW-") || startsWith(id, "NATIVE-") ||
      startsWith(id, "LOOP-") || startsWith(id, "LIVE-")) {
    return Category::Integration;
  }
  return Category::Unit;
}

class Reporter {
 public:
  explicit Reporter(std::string caseId) : caseId_(std::move(caseId)) {}

  void check(bool cond, const std::string& what) {
    total_++;
    if (!cond) {
      failed_++;
      std::cout << "  [FAIL] " << caseId_ << ": " << what << "\n";
    }
  }

  // Same as check(), but on failure also prints what the case actually
  // got vs. what it expected. Use this instead of a bare check() when a
  // mismatch is more useful to see than to guess at (a diagnostic code,
  // a rendered type, an emitted line) -- this is the "what did it
  // generate vs. what did I expect" case a plain pass/fail hides.
  void checkEqual(bool cond, const std::string& what, const std::string& actual, const std::string& expected) {
    total_++;
    if (!cond) {
      failed_++;
      std::cout << "  [FAIL] " << caseId_ << ": " << what << "\n"
                 << "    expected: " << expected << "\n"
                 << "    actual:   " << actual << "\n";
    }
  }

  int total() const { return total_; }
  int failed() const { return failed_; }

  // True when the case is running under `--verbose` (see
  // runRegisteredCases). A case body may use this to print extra
  // detail -- the token stream, the AST, the typed graph, the emitted
  // MLIR -- that would be too noisy to print on every run but is
  // exactly what you want when debugging one case by hand:
  //   ./test_semantic SEM-DECL-001 --verbose
  bool verbose() const { return verbose_; }
  void setVerbose(bool v) { verbose_ = v; }

 private:
  std::string caseId_;
  int total_ = 0;
  int failed_ = 0;
  bool verbose_ = false;
};

struct Case {
  std::string id;
  std::string description;
  Category category;
  std::function<void(Reporter&)> run;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Register {
  Register(std::string id, std::string desc, std::function<void(Reporter&)> fn) {
    Category category = categoryOf(id);
    registry().push_back({std::move(id), std::move(desc), category, std::move(fn)});
  }
};

// Defines case_<idSuffix>(Reporter&), registers it under the stable ID
// `idString`, and opens the function body.
#define TEST_CASE(idSuffix, idString, desc)                                     \
  void case_##idSuffix(::tensora::testing::Reporter&);                          \
  ::tensora::testing::Register reg_##idSuffix(idString, desc, case_##idSuffix); \
  void case_##idSuffix(::tensora::testing::Reporter& R)

// Runs every registered case, or one case by ID, or every case in one
// category (`--category=BOUNDARY`, matching Category above). A trailing
// `--verbose` flag (in any position) sets Reporter::verbose() for every
// case that runs, so a case can print extra debug detail -- this is
// meant for a single-case debug run
// (`./test_semantic SEM-DECL-001 --verbose`), not routine CI runs, so
// it is off by default and never changes what is checked, only what is
// printed. Prints a PASS/FAIL line for each case (with its category),
// then a summary line. Returns the total failed-check count as the
// process exit code (0 means every check passed).
inline int runRegisteredCases(int argc, char** argv) {
  std::vector<Case*> toRun;
  std::string wantCategory;
  bool verbose = false;
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else {
      args.push_back(arg);
    }
  }

  if (!args.empty()) {
    const std::string& arg = args.front();
    if (arg == "--list") {
      // Print every registered case ID, one per line. This is the
      // authoritative case list (the registry itself); CMake validates
      // its discovery against it (see INT-DISCOVERY-SYNC).
      for (Case& c : registry()) std::cout << c.id << "\n";
      return 0;
    }
    if (startsWith(arg, "--category=")) {
      wantCategory = arg.substr(std::string("--category=").size());
      for (Case& c : registry()) {
        if (categoryName(c.category) == wantCategory) toRun.push_back(&c);
      }
      if (toRun.empty()) {
        std::cerr << "no cases in category: " << wantCategory << "\n";
        return 2;
      }
    } else {
      for (Case& c : registry()) {
        if (c.id == arg) toRun.push_back(&c);
      }
      if (toRun.empty()) {
        std::cerr << "no such test case: " << arg << "\n";
        return 2;
      }
    }
  } else {
    for (Case& c : registry()) toRun.push_back(&c);
  }

  int totalChecks = 0;
  int totalFailedChecks = 0;
  int casesFailed = 0;
  for (Case* c : toRun) {
    Reporter r(c->id);
    r.setVerbose(verbose);
    c->run(r);
    totalChecks += r.total();
    totalFailedChecks += r.failed();
    bool ok = r.failed() == 0;
    if (!ok) casesFailed++;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << c->id << " [" << categoryName(c->category) << "]: "
              << c->description << " (" << r.total() << (r.total() == 1 ? " check" : " checks") << ")\n";
  }

  std::cout << "\n" << (static_cast<int>(toRun.size()) - casesFailed) << "/" << toRun.size()
            << " cases passed, " << (totalChecks - totalFailedChecks) << "/" << totalChecks
            << " checks passed\n";
  return totalFailedChecks;
}

}  // namespace tensora::testing
