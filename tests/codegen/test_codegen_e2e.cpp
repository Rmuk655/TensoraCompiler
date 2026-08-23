// Consolidated codegen end-to-end driver.
//
// Each codegen e2e check has its own numeric-JIT implementation in its own
// translation unit (test_*_e2e.cpp), exposed as run_*(). They used to be
// eight separate executables with their own main(); here they are one
// binary, registered as harness CASEs so they share the same
// CASE-discovery/registration and per-case CTest isolation as every other
// test (ctest runs `test_codegen_e2e <ID>`). Each run_* prints its own
// diagnostics and returns 0 on success; the CASE just asserts that.
#include "support/TestHarness.h"

int run_mlp_e2e();
int run_tiling_e2e();
int run_liveness_tiling_e2e();
int run_new_ops_e2e();
int run_rank_e2e();
int run_cnn_e2e();
int run_attention_e2e();
int run_ffn_e2e();

using tensora::testing::Reporter;
#define CASE TEST_CASE

CASE(LOW_E2E_MLP, "LOW-E2E-MLP", "MLIR lowering of matmul/linear/relu, JIT-executed") {
  R.check(run_mlp_e2e() == 0, "MLP end-to-end numeric result matches the reference");
}
CASE(LOOP_002, "LOOP-002", "Level-2 matmul tiling pass: structure + numeric equivalence + timing") {
  R.check(run_tiling_e2e() == 0, "tiled and untiled matmul agree; tiled has scf.for loops");
}
CASE(LOOP_005, "LOOP-005", "liveness-driven tiling: MlirLowerer's own LivenessAnalysis picks the tile size") {
  R.check(run_liveness_tiling_e2e() == 0, "memory pressure selects the conservative tile size through real lowering");
}
CASE(LOW_NEWOPS, "LOW-NEWOPS", "transpose/flatten/softmax/conv2d(+padding)/sum/mean/maxpool lowering, JIT-executed") {
  R.check(run_new_ops_e2e() == 0, "each op's lowered result matches the reference");
}
CASE(RANK_E2E, "RANK-E2E", "rank-1/rank>=3 linear and rank>=2 batched matmul lowering, JIT-executed") {
  R.check(run_rank_e2e() == 0, "rank/batched lowering results match the reference");
}
CASE(E2E_002, "E2E-002", "CNN classifier (conv2d->relu->flatten->linear->softmax) vs reference") {
  R.check(run_cnn_e2e() == 0, "CNN class probability matches the hand-computed reference");
}
CASE(E2E_004, "E2E-004", "single-head scaled dot-product attention vs reference") {
  R.check(run_attention_e2e() == 0, "attention output matches the hand-computed reference");
}
CASE(E2E_003, "E2E-003", "Transformer FFN sublayer (linear->relu->linear) vs reference") {
  R.check(run_ffn_e2e() == 0, "FFN output matches the reference");
}

int main(int argc, char** argv) { return tensora::testing::runRegisteredCases(argc, argv); }
