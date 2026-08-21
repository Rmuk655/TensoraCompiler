/* NATIVE-001: a small C driver, with no other dependency, linked
 * directly against mlp.o (the native object file codegen_mlp_native
 * builds through LLVM's TargetMachine). It uses no MLIR/LLVM headers,
 * no JIT, and no ExecutionEngine. This is a real ahead-of-time-compiled
 * executable. It calls a plain C-ABI function ("tensora_entry": zero
 * args, a scalar float return value, no memref or ciface marshalling
 * needed) built by Tensora's backend.
 */
#include <math.h>
#include <stdio.h>

extern float tensora_entry(void);

int main(void) {
  float result = tensora_entry();
  printf("NATIVE-001: tensora_entry() = %f (expected 3.0)\n", result);
  if (fabsf(result - 3.0f) > 1e-6f) {
    fprintf(stderr, "NATIVE-001 FAIL: expected 3.0, got %f\n", result);
    return 1;
  }
  printf("NATIVE-001 PASS: MLIR/LLVM-compiled native executable computes MLP(linear->relu->linear) = %f\n", result);
  return 0;
}
