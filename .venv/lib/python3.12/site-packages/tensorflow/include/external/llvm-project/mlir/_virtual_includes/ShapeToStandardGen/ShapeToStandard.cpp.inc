/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ShapeToStandard.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    external/llvm-project/mlir/lib/Conversion/ShapeToStandard/ShapeToStandard.td:23
*/
struct CstrBroadcastableToRequire : public ::mlir::RewritePattern {
  CstrBroadcastableToRequire(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.cstr_broadcastable", 1, context, {"shape.cstr_require", "shape.is_broadcastable"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range shapes(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::shape::CstrBroadcastableOp>(op0); (void)castedOp0;
    shapes = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::shape::IsBroadcastableOp tblgen_IsBroadcastableOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values_0;
      for (auto v: shapes) {
        tblgen_values_0.push_back(v);
      }
      tblgen_IsBroadcastableOp_0 = ::mlir::shape::IsBroadcastableOp::create(rewriter, odsLoc,
        /*shapes=*/tblgen_values_0
      );
    }
    auto nativeVar_1 = 
      rewriter.getStringAttr("required broadcastable shapes")
    ; (void)nativeVar_1;
    ::mlir::shape::CstrRequireOp tblgen_CstrRequireOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::shape::CstrRequireOp::Properties tblgen_props; (void)tblgen_props;
      tblgen_values.push_back((*tblgen_IsBroadcastableOp_0.getODSResults(0).begin()));
      tblgen_props.msg = ::llvm::dyn_cast_if_present<decltype(tblgen_props.msg)>(nativeVar_1);
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CstrRequireOp_2 = ::mlir::shape::CstrRequireOp::create(rewriter, odsLoc, tblgen_types, tblgen_values, tblgen_props);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CstrRequireOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

/* Generated from:
    external/llvm-project/mlir/lib/Conversion/ShapeToStandard/ShapeToStandard.td:32
*/
struct CstrEqToRequire : public ::mlir::RewritePattern {
  CstrEqToRequire(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("shape.cstr_eq", 1, context, {"shape.cstr_require", "shape.shape_eq"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Operation::operand_range shapes(op0->getOperands());
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::shape::CstrEqOp>(op0); (void)castedOp0;
    shapes = castedOp0.getODSOperands(0);

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::shape::ShapeEqOp tblgen_ShapeEqOp_0;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values_0;
      for (auto v: shapes) {
        tblgen_values_0.push_back(v);
      }
      tblgen_ShapeEqOp_0 = ::mlir::shape::ShapeEqOp::create(rewriter, odsLoc,
        /*shapes=*/tblgen_values_0
      );
    }
    auto nativeVar_1 = 
      rewriter.getStringAttr("required equal shapes")
    ; (void)nativeVar_1;
    ::mlir::shape::CstrRequireOp tblgen_CstrRequireOp_2;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::shape::CstrRequireOp::Properties tblgen_props; (void)tblgen_props;
      tblgen_values.push_back((*tblgen_ShapeEqOp_0.getODSResults(0).begin()));
      tblgen_props.msg = ::llvm::dyn_cast_if_present<decltype(tblgen_props.msg)>(nativeVar_1);
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_CstrRequireOp_2 = ::mlir::shape::CstrRequireOp::create(rewriter, odsLoc, tblgen_types, tblgen_values, tblgen_props);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_CstrRequireOp_2.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  }
};

[[maybe_unused]] void populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<CstrBroadcastableToRequire>(patterns.getContext());
  patterns.add<CstrEqToRequire>(patterns.getContext());
}
