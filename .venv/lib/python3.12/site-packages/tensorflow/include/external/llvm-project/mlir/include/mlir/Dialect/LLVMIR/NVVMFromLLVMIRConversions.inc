if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_aggr_smem_size) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::AggrSmemSize::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockDimXOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockDimYOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockDimZOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockIdXOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockIdYOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockIdZOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_x) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockInClusterIdXOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_y) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockInClusterIdYOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_z) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::BlockInClusterIdZOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_clock64) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::Clock64Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_clock) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClockOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_cluster_nctarank) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterDim::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_x) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterDimBlocksXOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_y) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterDimBlocksYOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_z) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterDimBlocksZOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_nclusterid_x) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterDimXOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_nclusterid_y) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterDimYOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_nclusterid_z) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterDimZOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_cluster_ctarank) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterId::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_clusterid_x) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterIdXOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_clusterid_y) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterIdYOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_clusterid_z) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ClusterIdZOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_dynamic_smem_size) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::DynamicSmemSize::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg0) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg0Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg1) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg1Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg2) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg2Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg3) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg3Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg4) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg4Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg5) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg5Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg6) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg6Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg7) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg7Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg8) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg8Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg9) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg9Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg10) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg10Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg11) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg11Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg12) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg12Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg13) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg13Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg14) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg14Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg15) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg15Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg16) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg16Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg17) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg17Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg18) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg18Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg19) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg19Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg20) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg20Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg21) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg21Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg22) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg22Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg23) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg23Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg24) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg24Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg25) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg25Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg26) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg26Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg27) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg27Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg28) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg28Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg29) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg29Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg30) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg30Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_envreg31) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::EnvReg31Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_globaltimer_lo) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::GlobalTimerLoOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_globaltimer) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::GlobalTimerOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::GridDimXOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_y) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::GridDimYOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_z) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::GridDimZOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_gridid) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::GridIdOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_laneid) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::LaneIdOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_eq) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::LaneMaskEqOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_ge) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::LaneMaskGeOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_gt) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::LaneMaskGtOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_le) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::LaneMaskLeOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_lanemask_lt) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::LaneMaskLtOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_rcp_approx_ftz_f) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::RcpApproxFtzF32Op::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_nsmid) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::SmDimOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_smid) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::SmIdOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ThreadIdXOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ThreadIdYOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::ThreadIdZOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_total_smem_size) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::TotalSmemSize::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_nwarpid) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::WarpDimOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_warpid) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::WarpIdOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
if (intrinsicID == llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize) {


    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        false,
        {},
        {},
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    {moduleImport.convertType(inst->getType())};
    auto op = ::mlir::NVVM::WarpSizeOp::create(odsBuilder,
      moduleImport.translateLoc(inst->getDebugLoc()), resultTypes, mlirOperands, mlirAttrs);
    
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get(odsBuilder.getContext(), value.getLower(), value.getUpper()));
    }
  moduleImport.mapValue(inst) = op;
  return success();
}
