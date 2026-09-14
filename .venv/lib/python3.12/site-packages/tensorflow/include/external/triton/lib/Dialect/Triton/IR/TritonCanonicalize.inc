/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Canonicalize.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    external/triton/lib/Dialect/Triton/IR/Canonicalize.td:13
*/
struct BroadcastBroadcastPattern : public ::mlir::RewritePattern {
  BroadcastBroadcastPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tt.broadcast", 2, context, {"tt.broadcast"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::triton::BroadcastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::triton::BroadcastOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::triton::BroadcastOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::triton::BroadcastOp tblgen_BroadcastOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::triton::BroadcastOp::Properties tblgen_props; (void)tblgen_props;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_BroadcastOp_0 = ::mlir::triton::BroadcastOp::create(rewriter, odsLoc, tblgen_types, tblgen_values, tblgen_props);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_BroadcastOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/triton/lib/Dialect/Triton/IR/Canonicalize.td:8
*/
struct BroadcastSplatPattern : public ::mlir::RewritePattern {
  BroadcastSplatPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tt.broadcast", 2, context, {"tt.splat"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range x(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::triton::BroadcastOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::triton::SplatOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::triton::SplatOp type";
        });
      }
      x = castedOp1.getODSOperands(0);
      tblgen_ops.push_back(op1);
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::triton::SplatOp tblgen_SplatOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::triton::SplatOp::Properties tblgen_props; (void)tblgen_props;
      tblgen_values.push_back((*x.begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_SplatOp_0 = ::mlir::triton::SplatOp::create(rewriter, odsLoc, tblgen_types, tblgen_values, tblgen_props);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_SplatOp_0.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

[[maybe_unused]] void populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<BroadcastBroadcastPattern>(patterns.getContext());
  patterns.add<BroadcastSplatPattern>(patterns.getContext());
}
