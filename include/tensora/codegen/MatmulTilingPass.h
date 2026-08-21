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
