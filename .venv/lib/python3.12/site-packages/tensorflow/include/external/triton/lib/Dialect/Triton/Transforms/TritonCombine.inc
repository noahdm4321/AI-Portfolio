/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Combine.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    external/triton/lib/Dialect/Triton/Transforms/Combine.td:17
*/
struct CombineAddPtrPattern : public ::mlir::RewritePattern {
  CombineAddPtrPattern(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tt.addptr", 2, context, {"arith.addi", "tt.addptr"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range idx1(op0->getOperands());
    ::mlir::Operation::operand_range idx0(op0->getOperands());
    ::mlir::Operation::operand_range ptr(op0->getOperands());
    ::mlir::triton::AddPtrOp src;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::triton::AddPtrOp>(op0); (void)castedOp0;
    src = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      if (!(op1)){
        return rewriter.notifyMatchFailure(castedOp0, [&](::mlir::Diagnostic &diag) {
          diag << "There's no operation that defines operand 0 of castedOp0";
        });
      }
      auto castedOp1 = ::llvm::dyn_cast<::mlir::triton::AddPtrOp>(op1); (void)castedOp1;
      if (!(castedOp1)){
        return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
          diag << "castedOp1 is not ::mlir::triton::AddPtrOp type";
        });
      }
      ptr = castedOp1.getODSOperands(0);
      idx0 = castedOp1.getODSOperands(1);
      tblgen_ops.push_back(op1);
    }
    idx1 = castedOp0.getODSOperands(1);
    if (!((isAddPtrOffsetCombinable((*idx0.begin()), (*idx1.begin()))))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'idx0, idx1' failed to satisfy constraint: ''";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::arith::AddIOp tblgen_AddIOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::arith::AddIOp::Properties tblgen_props; (void)tblgen_props;
      tblgen_values.push_back((*idx0.begin()));
      tblgen_values.push_back((*idx1.begin()));
      tblgen_props.overflowFlags = ::llvm::dyn_cast_if_present<decltype(tblgen_props.overflowFlags)>(::mlir::arith::IntegerOverflowFlagsAttr::get(rewriter.getContext(), ::mlir::arith::IntegerOverflowFlags::none));
      tblgen_AddIOp_0 = ::mlir::arith::AddIOp::create(rewriter, odsLoc, tblgen_values, tblgen_props);
    }
    ::mlir::triton::AddPtrOp dest;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::triton::AddPtrOp::Properties tblgen_props; (void)tblgen_props;
      tblgen_values.push_back((*ptr.begin()));
      tblgen_values.push_back((*tblgen_AddIOp_0.getODSResults(0).begin()));
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      dest = ::mlir::triton::AddPtrOp::create(rewriter, odsLoc, tblgen_types, tblgen_values, tblgen_props);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ dest.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }
    (*dest.getODSResults(0).begin()).getOwner()->setDiscardableAttrs(triton::filterDiscardableAttrs((*src.getODSResults(0).begin()).getOwner(), {"tt.divisibility", "tt.contiguity", "tt.constancy"}));

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

[[maybe_unused]] void populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<CombineAddPtrPattern>(patterns.getContext());
}
