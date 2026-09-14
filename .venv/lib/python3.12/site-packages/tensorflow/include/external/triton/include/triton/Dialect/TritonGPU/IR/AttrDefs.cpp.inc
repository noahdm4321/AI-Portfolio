/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::triton::gpu::CGAEncodingAttr,
::mlir::triton::gpu::SwizzledSharedEncodingAttr,
::mlir::triton::gpu::PaddedSharedEncodingAttr,
::mlir::triton::gpu::SharedLinearEncodingAttr,
::mlir::triton::gpu::NVMMASharedEncodingAttr,
::mlir::triton::gpu::AMDRotatingSharedEncodingAttr,
::mlir::triton::gpu::LinearEncodingAttr,
::mlir::triton::gpu::BlockedEncodingAttr,
::mlir::triton::gpu::AMDMfmaEncodingAttr,
::mlir::triton::gpu::AMDWmmaEncodingAttr,
::mlir::triton::gpu::NvidiaMmaEncodingAttr,
::mlir::triton::gpu::SliceEncodingAttr,
::mlir::triton::gpu::DotOperandEncodingAttr,
::mlir::triton::gpu::SharedMemorySpaceAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::gpu::SwizzledSharedEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::SwizzledSharedEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::PaddedSharedEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::PaddedSharedEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::SharedLinearEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::SharedLinearEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::NVMMASharedEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::NVMMASharedEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::AMDRotatingSharedEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::AMDRotatingSharedEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::LinearEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::LinearEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::BlockedEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::BlockedEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::AMDMfmaEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::AMDMfmaEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::AMDWmmaEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::AMDWmmaEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::NvidiaMmaEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::NvidiaMmaEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::SliceEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::SliceEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::DotOperandEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::DotOperandEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gpu::SharedMemorySpaceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gpu::SharedMemorySpaceAttr::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::triton::gpu::SwizzledSharedEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::SwizzledSharedEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::PaddedSharedEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::PaddedSharedEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::SharedLinearEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::SharedLinearEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::NVMMASharedEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::NVMMASharedEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::AMDRotatingSharedEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::AMDRotatingSharedEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::LinearEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::LinearEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::BlockedEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::BlockedEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::AMDMfmaEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::AMDMfmaEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::AMDWmmaEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::AMDWmmaEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::NvidiaMmaEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::NvidiaMmaEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::SliceEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::SliceEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::DotOperandEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::DotOperandEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::gpu::SharedMemorySpaceAttr>([&](auto t) {
      printer << ::mlir::triton::gpu::SharedMemorySpaceAttr::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::triton::gpu {

namespace detail {

struct CGAEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<LinearLayout>;
  CGAEncodingAttrStorage(LinearLayout linearLayout) : linearLayout(std::move(linearLayout)) {}

  KeyTy getAsKey() const {
    return KeyTy(linearLayout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (linearLayout == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CGAEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto linearLayout = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CGAEncodingAttrStorage>()) CGAEncodingAttrStorage(std::move(linearLayout));
  }

  LinearLayout linearLayout;
};

} // namespace detail
CGAEncodingAttr CGAEncodingAttr::get(::mlir::MLIRContext *context, LinearLayout linearLayout) {
  return Base::get(context, std::move(linearLayout));
}

CGAEncodingAttr CGAEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, LinearLayout linearLayout) {
  return Base::getChecked(emitError, context, std::move(linearLayout));
}

::llvm::LogicalResult CGAEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout) {
  if (::mlir::failed(verify(emitError, linearLayout)))
    return ::mlir::failure();
  return ::mlir::success();
}

const LinearLayout &CGAEncodingAttr::getLinearLayout() const {
  return getImpl()->linearLayout;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::CGAEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct SwizzledSharedEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, unsigned, unsigned, ::llvm::ArrayRef<unsigned>, CGAEncodingAttr>;
  SwizzledSharedEncodingAttrStorage(unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) : vec(std::move(vec)), perPhase(std::move(perPhase)), maxPhase(std::move(maxPhase)), order(std::move(order)), CGALayout(std::move(CGALayout)) {}

  KeyTy getAsKey() const {
    return KeyTy(vec, perPhase, maxPhase, order, CGALayout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (vec == std::get<0>(tblgenKey)) && (perPhase == std::get<1>(tblgenKey)) && (maxPhase == std::get<2>(tblgenKey)) && (order == std::get<3>(tblgenKey)) && (CGALayout == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static SwizzledSharedEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto vec = std::move(std::get<0>(tblgenKey));
    auto perPhase = std::move(std::get<1>(tblgenKey));
    auto maxPhase = std::move(std::get<2>(tblgenKey));
    auto order = std::move(std::get<3>(tblgenKey));
    auto CGALayout = std::move(std::get<4>(tblgenKey));
    order = allocator.copyInto(order);
    return new (allocator.allocate<SwizzledSharedEncodingAttrStorage>()) SwizzledSharedEncodingAttrStorage(std::move(vec), std::move(perPhase), std::move(maxPhase), std::move(order), std::move(CGALayout));
  }

  unsigned vec;
  unsigned perPhase;
  unsigned maxPhase;
  ::llvm::ArrayRef<unsigned> order;
  CGAEncodingAttr CGALayout;
};

} // namespace detail
SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::get(::mlir::MLIRContext *context, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) {
  return Base::get(context, std::move(vec), std::move(perPhase), std::move(maxPhase), std::move(order), std::move(CGALayout));
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) {
  return Base::getChecked(emitError, context, std::move(vec), std::move(perPhase), std::move(maxPhase), std::move(order), std::move(CGALayout));
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::get(::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned typeWidthInBit) {
  bool needTrans = false; // default value
  return get(context, dotOpEnc, shape, order, CGALayout, typeWidthInBit, needTrans);
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned typeWidthInBit) {
  bool needTrans = false; // default value
  return get(context, dotOpEnc, shape, order, CGALayout, typeWidthInBit, needTrans);
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::get(::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned typeWidthInBit, bool needTrans) {
  // ---- begin MFMA ----
  if (auto mfmaEnc = mlir::dyn_cast<AMDMfmaEncodingAttr>(dotOpEnc.getParent())) {
    return mfmaEnc.composeSharedLayoutForOperand(
        CGALayout, dotOpEnc.getOpIdx(), shape, order, dotOpEnc.getKWidth(),
        typeWidthInBit, needTrans);
  }

  // ---- begin WMMA ----
  if (auto wmmaEnc = mlir::dyn_cast<AMDWmmaEncodingAttr>(dotOpEnc.getParent())) {
    return wmmaEnc.composeSharedLayoutForOperand(
        CGALayout, dotOpEnc.getOpIdx(), shape, order, dotOpEnc.getKWidth(),
        typeWidthInBit, needTrans);
  }


  auto mmaEnc = mlir::dyn_cast<NvidiaMmaEncodingAttr>(dotOpEnc.getParent());

  if(!mmaEnc)
    return get(context, 1, 1, 1, order, CGALayout);

  // ---- begin Ampere & Hopper ----
  if (mmaEnc.isAmpere() || mmaEnc.isHopper()) {
    return get(context, dotOpEnc.getOpIdx(), dotOpEnc.getKWidth(), shape, order, CGALayout, typeWidthInBit, needTrans);
  }

  // ---- not implemented ----
  llvm_unreachable("unsupported swizzling for provided MMA version");
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned typeWidthInBit, bool needTrans) {
  // ---- begin MFMA ----
  if (auto mfmaEnc = mlir::dyn_cast<AMDMfmaEncodingAttr>(dotOpEnc.getParent())) {
    return mfmaEnc.composeSharedLayoutForOperand(
        CGALayout, dotOpEnc.getOpIdx(), shape, order, dotOpEnc.getKWidth(),
        typeWidthInBit, needTrans);
  }

  // ---- begin WMMA ----
  if (auto wmmaEnc = mlir::dyn_cast<AMDWmmaEncodingAttr>(dotOpEnc.getParent())) {
    return wmmaEnc.composeSharedLayoutForOperand(
        CGALayout, dotOpEnc.getOpIdx(), shape, order, dotOpEnc.getKWidth(),
        typeWidthInBit, needTrans);
  }


  auto mmaEnc = mlir::dyn_cast<NvidiaMmaEncodingAttr>(dotOpEnc.getParent());

  if(!mmaEnc)
    return get(context, 1, 1, 1, order, CGALayout);

  // ---- begin Ampere & Hopper ----
  if (mmaEnc.isAmpere() || mmaEnc.isHopper()) {
    return get(context, dotOpEnc.getOpIdx(), dotOpEnc.getKWidth(), shape, order, CGALayout, typeWidthInBit, needTrans);
  }

  // ---- not implemented ----
  llvm_unreachable("unsupported swizzling for provided MMA version");
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::get(::mlir::MLIRContext *context, int opIdx, unsigned kWidth, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned bitwidth, bool needTrans) {
  int K =  getShapePerCTA(CGALayout.getCTASplitNum(), shape)[order[0]];
  // Elems necessary to cover all the banks divided by the inner dimension
  // This packs a few rows together for small K
  int perPhase = std::max<int>(1024 / (bitwidth * K), 1);

  int mmaStride = 8;
  int vec = 4 * kWidth;
  // needsTrans is equiv. to flipping the opIdx
  if (needTrans)
    std::swap(vec, mmaStride);
  assert(opIdx == 0 || opIdx == 1);
  int rank = order.size();
  int kDim = opIdx == 0 ? rank-1 : rank-2;
  if (order[0] != kDim)
    std::swap(vec, mmaStride);
  // Count how many vec elements are needed to cover all the banks
  int maxPhase = std::max(std::min<int>(mmaStride, 1024 / (vec * bitwidth)), 1);
  // Account for the row packing from perPhase: mmaStride / perPhase
  maxPhase = std::max(maxPhase / perPhase, 1);
  return get(context, vec, perPhase, maxPhase, order, CGALayout);
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int opIdx, unsigned kWidth, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned bitwidth, bool needTrans) {
  int K =  getShapePerCTA(CGALayout.getCTASplitNum(), shape)[order[0]];
  // Elems necessary to cover all the banks divided by the inner dimension
  // This packs a few rows together for small K
  int perPhase = std::max<int>(1024 / (bitwidth * K), 1);

  int mmaStride = 8;
  int vec = 4 * kWidth;
  // needsTrans is equiv. to flipping the opIdx
  if (needTrans)
    std::swap(vec, mmaStride);
  assert(opIdx == 0 || opIdx == 1);
  int rank = order.size();
  int kDim = opIdx == 0 ? rank-1 : rank-2;
  if (order[0] != kDim)
    std::swap(vec, mmaStride);
  // Count how many vec elements are needed to cover all the banks
  int maxPhase = std::max(std::min<int>(mmaStride, 1024 / (vec * bitwidth)), 1);
  // Account for the row packing from perPhase: mmaStride / perPhase
  maxPhase = std::max(maxPhase / perPhase, 1);
  return get(context, vec, perPhase, maxPhase, order, CGALayout);
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::get(::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy) {
  unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
  return get(context, dotOpEnc, shape, order, CGALayout, bitwidth);
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy) {
  unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
  return get(context, dotOpEnc, shape, order, CGALayout, bitwidth);
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::get(::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy, bool needTrans) {
  unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
  return get(context, dotOpEnc, shape, order, CGALayout, bitwidth, needTrans);
}

SwizzledSharedEncodingAttr SwizzledSharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy, bool needTrans) {
  unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
  return get(context, dotOpEnc, shape, order, CGALayout, bitwidth, needTrans);
}

::llvm::LogicalResult SwizzledSharedEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) {
  if (::mlir::failed(verify(emitError, vec, perPhase, maxPhase, order, CGALayout)))
    return ::mlir::failure();
  return ::mlir::success();
}

unsigned SwizzledSharedEncodingAttr::getVec() const {
  return getImpl()->vec;
}

unsigned SwizzledSharedEncodingAttr::getPerPhase() const {
  return getImpl()->perPhase;
}

unsigned SwizzledSharedEncodingAttr::getMaxPhase() const {
  return getImpl()->maxPhase;
}

::llvm::ArrayRef<unsigned> SwizzledSharedEncodingAttr::getOrder() const {
  return getImpl()->order;
}

CGAEncodingAttr SwizzledSharedEncodingAttr::getCGALayout() const {
  return getImpl()->CGALayout;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::SwizzledSharedEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct PaddedSharedEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<unsigned>, ::llvm::ArrayRef<unsigned>, LinearLayout>;
  PaddedSharedEncodingAttrStorage(::llvm::ArrayRef<unsigned> intervals, ::llvm::ArrayRef<unsigned> paddings, LinearLayout linearComponent) : intervals(std::move(intervals)), paddings(std::move(paddings)), linearComponent(std::move(linearComponent)) {}

  KeyTy getAsKey() const {
    return KeyTy(intervals, paddings, linearComponent);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (intervals == std::get<0>(tblgenKey)) && (paddings == std::get<1>(tblgenKey)) && (linearComponent == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static PaddedSharedEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto intervals = std::move(std::get<0>(tblgenKey));
    auto paddings = std::move(std::get<1>(tblgenKey));
    auto linearComponent = std::move(std::get<2>(tblgenKey));
    intervals = allocator.copyInto(intervals);
    paddings = allocator.copyInto(paddings);
    return new (allocator.allocate<PaddedSharedEncodingAttrStorage>()) PaddedSharedEncodingAttrStorage(std::move(intervals), std::move(paddings), std::move(linearComponent));
  }

  ::llvm::ArrayRef<unsigned> intervals;
  ::llvm::ArrayRef<unsigned> paddings;
  LinearLayout linearComponent;
};

} // namespace detail
PaddedSharedEncodingAttr PaddedSharedEncodingAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<unsigned> intervals, ::llvm::ArrayRef<unsigned> paddings, LinearLayout linearComponent) {
  return Base::get(context, std::move(intervals), std::move(paddings), std::move(linearComponent));
}

PaddedSharedEncodingAttr PaddedSharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<unsigned> intervals, ::llvm::ArrayRef<unsigned> paddings, LinearLayout linearComponent) {
  return Base::getChecked(emitError, context, std::move(intervals), std::move(paddings), std::move(linearComponent));
}

::llvm::LogicalResult PaddedSharedEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<unsigned> intervals, ::llvm::ArrayRef<unsigned> paddings, LinearLayout linearComponent) {
  if (::mlir::failed(verify(emitError, intervals, paddings, linearComponent)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<unsigned> PaddedSharedEncodingAttr::getIntervals() const {
  return getImpl()->intervals;
}

::llvm::ArrayRef<unsigned> PaddedSharedEncodingAttr::getPaddings() const {
  return getImpl()->paddings;
}

const LinearLayout &PaddedSharedEncodingAttr::getLinearComponent() const {
  return getImpl()->linearComponent;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::PaddedSharedEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct SharedLinearEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<LinearLayout, unsigned>;
  SharedLinearEncodingAttrStorage(LinearLayout linearLayout, unsigned layoutAlignment) : linearLayout(std::move(linearLayout)), layoutAlignment(std::move(layoutAlignment)) {}

  KeyTy getAsKey() const {
    return KeyTy(linearLayout, layoutAlignment);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (linearLayout == std::get<0>(tblgenKey)) && (layoutAlignment == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static SharedLinearEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto linearLayout = std::move(std::get<0>(tblgenKey));
    auto layoutAlignment = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<SharedLinearEncodingAttrStorage>()) SharedLinearEncodingAttrStorage(std::move(linearLayout), std::move(layoutAlignment));
  }

  LinearLayout linearLayout;
  unsigned layoutAlignment;
};

} // namespace detail
SharedLinearEncodingAttr SharedLinearEncodingAttr::get(::mlir::MLIRContext *context, LinearLayout linearLayout, unsigned layoutAlignment) {
  return Base::get(context, std::move(linearLayout), std::move(layoutAlignment));
}

SharedLinearEncodingAttr SharedLinearEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, LinearLayout linearLayout, unsigned layoutAlignment) {
  return Base::getChecked(emitError, context, std::move(linearLayout), std::move(layoutAlignment));
}

::llvm::LogicalResult SharedLinearEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout, unsigned layoutAlignment) {
  if (::mlir::failed(verify(emitError, linearLayout, layoutAlignment)))
    return ::mlir::failure();
  return ::mlir::success();
}

const LinearLayout &SharedLinearEncodingAttr::getLinearLayout() const {
  return getImpl()->linearLayout;
}

unsigned SharedLinearEncodingAttr::getLayoutAlignment() const {
  return getImpl()->layoutAlignment;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::SharedLinearEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct NVMMASharedEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, bool, unsigned, bool, CGAEncodingAttr>;
  NVMMASharedEncodingAttrStorage(unsigned swizzlingByteWidth, bool transposed, unsigned elementBitWidth, bool fp4Padded, CGAEncodingAttr CGALayout) : swizzlingByteWidth(std::move(swizzlingByteWidth)), transposed(std::move(transposed)), elementBitWidth(std::move(elementBitWidth)), fp4Padded(std::move(fp4Padded)), CGALayout(std::move(CGALayout)) {}

  KeyTy getAsKey() const {
    return KeyTy(swizzlingByteWidth, transposed, elementBitWidth, fp4Padded, CGALayout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (swizzlingByteWidth == std::get<0>(tblgenKey)) && (transposed == std::get<1>(tblgenKey)) && (elementBitWidth == std::get<2>(tblgenKey)) && (fp4Padded == std::get<3>(tblgenKey)) && (CGALayout == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static NVMMASharedEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto swizzlingByteWidth = std::move(std::get<0>(tblgenKey));
    auto transposed = std::move(std::get<1>(tblgenKey));
    auto elementBitWidth = std::move(std::get<2>(tblgenKey));
    auto fp4Padded = std::move(std::get<3>(tblgenKey));
    auto CGALayout = std::move(std::get<4>(tblgenKey));
    return new (allocator.allocate<NVMMASharedEncodingAttrStorage>()) NVMMASharedEncodingAttrStorage(std::move(swizzlingByteWidth), std::move(transposed), std::move(elementBitWidth), std::move(fp4Padded), std::move(CGALayout));
  }

  unsigned swizzlingByteWidth;
  bool transposed;
  unsigned elementBitWidth;
  bool fp4Padded;
  CGAEncodingAttr CGALayout;
};

} // namespace detail
NVMMASharedEncodingAttr NVMMASharedEncodingAttr::get(::mlir::MLIRContext *context, unsigned swizzlingByteWidth, bool transposed, unsigned elementBitWidth, bool fp4Padded, CGAEncodingAttr CGALayout) {
  return Base::get(context, std::move(swizzlingByteWidth), std::move(transposed), std::move(elementBitWidth), std::move(fp4Padded), std::move(CGALayout));
}

NVMMASharedEncodingAttr NVMMASharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned swizzlingByteWidth, bool transposed, unsigned elementBitWidth, bool fp4Padded, CGAEncodingAttr CGALayout) {
  return Base::getChecked(emitError, context, std::move(swizzlingByteWidth), std::move(transposed), std::move(elementBitWidth), std::move(fp4Padded), std::move(CGALayout));
}

NVMMASharedEncodingAttr NVMMASharedEncodingAttr::get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy, bool fp4Padded) {
  auto shapePerCTA = getShapePerCTA(CGALayout.getCTASplitNum(), shape);
  int32_t swizzlingByteWidth = 0;
  unsigned eleBitWidth = eltTy.getIntOrFloatBitWidth();
  int packingFactor = fp4Padded ? 2 : 1;

  // get proper shared memory swizzling mode from the contiguous dimension
  // size of the origin blocked layout.
  auto contigDimSizeInByte = shapePerCTA[order[0]] * packingFactor * eleBitWidth / 8;
  if (contigDimSizeInByte >= 128 && contigDimSizeInByte % 128 == 0) {
    swizzlingByteWidth = 128;
  } else if (contigDimSizeInByte >= 64 && contigDimSizeInByte % 64 == 0) {
    swizzlingByteWidth = 64;
  } else if (contigDimSizeInByte >= 32 && contigDimSizeInByte % 32 == 0) {
    swizzlingByteWidth = 32;
  } else {
    swizzlingByteWidth = 0;
  }
  int flattenOutterDim = 1;
  for (int i = 1; i < shapePerCTA.size(); i++) {
    flattenOutterDim *= shapePerCTA[order[i]];
  }
  if (shapePerCTA.size() < 2 || flattenOutterDim < 8) {
    swizzlingByteWidth = 0;
  }
  bool transposed = order.size() > 1 && order[0] == 0;
  return Base::get(context, swizzlingByteWidth, transposed, eleBitWidth, fp4Padded, CGALayout);
}

NVMMASharedEncodingAttr NVMMASharedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy, bool fp4Padded) {
  auto shapePerCTA = getShapePerCTA(CGALayout.getCTASplitNum(), shape);
  int32_t swizzlingByteWidth = 0;
  unsigned eleBitWidth = eltTy.getIntOrFloatBitWidth();
  int packingFactor = fp4Padded ? 2 : 1;

  // get proper shared memory swizzling mode from the contiguous dimension
  // size of the origin blocked layout.
  auto contigDimSizeInByte = shapePerCTA[order[0]] * packingFactor * eleBitWidth / 8;
  if (contigDimSizeInByte >= 128 && contigDimSizeInByte % 128 == 0) {
    swizzlingByteWidth = 128;
  } else if (contigDimSizeInByte >= 64 && contigDimSizeInByte % 64 == 0) {
    swizzlingByteWidth = 64;
  } else if (contigDimSizeInByte >= 32 && contigDimSizeInByte % 32 == 0) {
    swizzlingByteWidth = 32;
  } else {
    swizzlingByteWidth = 0;
  }
  int flattenOutterDim = 1;
  for (int i = 1; i < shapePerCTA.size(); i++) {
    flattenOutterDim *= shapePerCTA[order[i]];
  }
  if (shapePerCTA.size() < 2 || flattenOutterDim < 8) {
    swizzlingByteWidth = 0;
  }
  bool transposed = order.size() > 1 && order[0] == 0;
  return Base::getChecked(emitError, context, swizzlingByteWidth, transposed, eleBitWidth, fp4Padded, CGALayout);
}

::llvm::LogicalResult NVMMASharedEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned swizzlingByteWidth, bool transposed, unsigned elementBitWidth, bool fp4Padded, CGAEncodingAttr CGALayout) {
  if (::mlir::failed(verify(emitError, swizzlingByteWidth, transposed, elementBitWidth, fp4Padded, CGALayout)))
    return ::mlir::failure();
  return ::mlir::success();
}

unsigned NVMMASharedEncodingAttr::getSwizzlingByteWidth() const {
  return getImpl()->swizzlingByteWidth;
}

bool NVMMASharedEncodingAttr::getTransposed() const {
  return getImpl()->transposed;
}

unsigned NVMMASharedEncodingAttr::getElementBitWidth() const {
  return getImpl()->elementBitWidth;
}

bool NVMMASharedEncodingAttr::getFp4Padded() const {
  return getImpl()->fp4Padded;
}

CGAEncodingAttr NVMMASharedEncodingAttr::getCGALayout() const {
  return getImpl()->CGALayout;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::NVMMASharedEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct AMDRotatingSharedEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, unsigned, unsigned, ::llvm::ArrayRef<unsigned>, CGAEncodingAttr>;
  AMDRotatingSharedEncodingAttrStorage(unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) : vec(std::move(vec)), perPhase(std::move(perPhase)), maxPhase(std::move(maxPhase)), order(std::move(order)), CGALayout(std::move(CGALayout)) {}

  KeyTy getAsKey() const {
    return KeyTy(vec, perPhase, maxPhase, order, CGALayout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (vec == std::get<0>(tblgenKey)) && (perPhase == std::get<1>(tblgenKey)) && (maxPhase == std::get<2>(tblgenKey)) && (order == std::get<3>(tblgenKey)) && (CGALayout == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static AMDRotatingSharedEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto vec = std::move(std::get<0>(tblgenKey));
    auto perPhase = std::move(std::get<1>(tblgenKey));
    auto maxPhase = std::move(std::get<2>(tblgenKey));
    auto order = std::move(std::get<3>(tblgenKey));
    auto CGALayout = std::move(std::get<4>(tblgenKey));
    order = allocator.copyInto(order);
    return new (allocator.allocate<AMDRotatingSharedEncodingAttrStorage>()) AMDRotatingSharedEncodingAttrStorage(std::move(vec), std::move(perPhase), std::move(maxPhase), std::move(order), std::move(CGALayout));
  }

  unsigned vec;
  unsigned perPhase;
  unsigned maxPhase;
  ::llvm::ArrayRef<unsigned> order;
  CGAEncodingAttr CGALayout;
};

} // namespace detail
AMDRotatingSharedEncodingAttr AMDRotatingSharedEncodingAttr::get(::mlir::MLIRContext *context, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) {
  return Base::get(context, std::move(vec), std::move(perPhase), std::move(maxPhase), std::move(order), std::move(CGALayout));
}

unsigned AMDRotatingSharedEncodingAttr::getVec() const {
  return getImpl()->vec;
}

unsigned AMDRotatingSharedEncodingAttr::getPerPhase() const {
  return getImpl()->perPhase;
}

unsigned AMDRotatingSharedEncodingAttr::getMaxPhase() const {
  return getImpl()->maxPhase;
}

::llvm::ArrayRef<unsigned> AMDRotatingSharedEncodingAttr::getOrder() const {
  return getImpl()->order;
}

CGAEncodingAttr AMDRotatingSharedEncodingAttr::getCGALayout() const {
  return getImpl()->CGALayout;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::AMDRotatingSharedEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct LinearEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<LinearLayout>;
  LinearEncodingAttrStorage(LinearLayout linearLayout) : linearLayout(std::move(linearLayout)) {}

  KeyTy getAsKey() const {
    return KeyTy(linearLayout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (linearLayout == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static LinearEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto linearLayout = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<LinearEncodingAttrStorage>()) LinearEncodingAttrStorage(std::move(linearLayout));
  }

  LinearLayout linearLayout;
};

} // namespace detail
LinearEncodingAttr LinearEncodingAttr::get(::mlir::MLIRContext *context, LinearLayout linearLayout) {
  return Base::get(context, std::move(linearLayout));
}

LinearEncodingAttr LinearEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, LinearLayout linearLayout) {
  return Base::getChecked(emitError, context, std::move(linearLayout));
}

::llvm::LogicalResult LinearEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout) {
  if (::mlir::failed(verify(emitError, linearLayout)))
    return ::mlir::failure();
  return ::mlir::success();
}

const LinearLayout &LinearEncodingAttr::getLinearLayout() const {
  return getImpl()->linearLayout;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::LinearEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct BlockedEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<unsigned>, ::llvm::ArrayRef<unsigned>, ::llvm::ArrayRef<unsigned>, ::llvm::ArrayRef<unsigned>, CGAEncodingAttr>;
  BlockedEncodingAttrStorage(::llvm::ArrayRef<unsigned> sizePerThread, ::llvm::ArrayRef<unsigned> threadsPerWarp, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) : sizePerThread(std::move(sizePerThread)), threadsPerWarp(std::move(threadsPerWarp)), warpsPerCTA(std::move(warpsPerCTA)), order(std::move(order)), CGALayout(std::move(CGALayout)) {}

  KeyTy getAsKey() const {
    return KeyTy(sizePerThread, threadsPerWarp, warpsPerCTA, order, CGALayout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (sizePerThread == std::get<0>(tblgenKey)) && (threadsPerWarp == std::get<1>(tblgenKey)) && (warpsPerCTA == std::get<2>(tblgenKey)) && (order == std::get<3>(tblgenKey)) && (CGALayout == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static BlockedEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto sizePerThread = std::move(std::get<0>(tblgenKey));
    auto threadsPerWarp = std::move(std::get<1>(tblgenKey));
    auto warpsPerCTA = std::move(std::get<2>(tblgenKey));
    auto order = std::move(std::get<3>(tblgenKey));
    auto CGALayout = std::move(std::get<4>(tblgenKey));
    sizePerThread = allocator.copyInto(sizePerThread);
    threadsPerWarp = allocator.copyInto(threadsPerWarp);
    warpsPerCTA = allocator.copyInto(warpsPerCTA);
    order = allocator.copyInto(order);
    return new (allocator.allocate<BlockedEncodingAttrStorage>()) BlockedEncodingAttrStorage(std::move(sizePerThread), std::move(threadsPerWarp), std::move(warpsPerCTA), std::move(order), std::move(CGALayout));
  }

  ::llvm::ArrayRef<unsigned> sizePerThread;
  ::llvm::ArrayRef<unsigned> threadsPerWarp;
  ::llvm::ArrayRef<unsigned> warpsPerCTA;
  ::llvm::ArrayRef<unsigned> order;
  CGAEncodingAttr CGALayout;
};

} // namespace detail
BlockedEncodingAttr BlockedEncodingAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<unsigned> sizePerThread, ::llvm::ArrayRef<unsigned> threadsPerWarp, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) {
  return Base::get(context, std::move(sizePerThread), std::move(threadsPerWarp), std::move(warpsPerCTA), std::move(order), std::move(CGALayout));
}

BlockedEncodingAttr BlockedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<unsigned> sizePerThread, ::llvm::ArrayRef<unsigned> threadsPerWarp, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) {
  return Base::getChecked(emitError, context, std::move(sizePerThread), std::move(threadsPerWarp), std::move(warpsPerCTA), std::move(order), std::move(CGALayout));
}

BlockedEncodingAttr BlockedEncodingAttr::get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order, unsigned numWarps, unsigned numThreadsPerWarp, CGAEncodingAttr CGALayout) {
  unsigned rank = sizePerThread.size();
  SmallVector<unsigned, 4> threadsPerWarp(rank);
  SmallVector<unsigned, 4> warpsPerCTA(rank);
  SmallVector<int64_t> shapePerCTA = getShapePerCTA(CGALayout.getCTASplitNum(), shape);

  unsigned remainingLanes = numThreadsPerWarp;
  unsigned remainingThreads = numWarps * numThreadsPerWarp;
  unsigned remainingWarps = numWarps;
  unsigned prevLanes = 1;
  unsigned prevWarps = 1;

  // starting from the contiguous dimension
  for (unsigned d = 0; d < rank - 1; ++d) {
    unsigned i = order[d];
    unsigned threadsPerCTA = std::clamp<unsigned>(remainingThreads, 1, std::max<unsigned>(1, shapePerCTA[i] / sizePerThread[i]));
    threadsPerWarp[i] = std::clamp<unsigned>(threadsPerCTA, 1, remainingLanes);
    warpsPerCTA[i] = std::clamp<unsigned>(threadsPerCTA / threadsPerWarp[i], 1, remainingWarps);
    remainingWarps /= warpsPerCTA[i];
    remainingLanes /= threadsPerWarp[i];
    remainingThreads /= threadsPerCTA;
    prevLanes *= threadsPerWarp[i];
    prevWarps *= warpsPerCTA[i];
  }

  // Expand the last dimension to fill the remaining lanes and warps
  threadsPerWarp[order[rank - 1]] = numThreadsPerWarp / prevLanes;
  warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

  return Base::get(context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CGALayout);
}

BlockedEncodingAttr BlockedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order, unsigned numWarps, unsigned numThreadsPerWarp, CGAEncodingAttr CGALayout) {
  unsigned rank = sizePerThread.size();
  SmallVector<unsigned, 4> threadsPerWarp(rank);
  SmallVector<unsigned, 4> warpsPerCTA(rank);
  SmallVector<int64_t> shapePerCTA = getShapePerCTA(CGALayout.getCTASplitNum(), shape);

  unsigned remainingLanes = numThreadsPerWarp;
  unsigned remainingThreads = numWarps * numThreadsPerWarp;
  unsigned remainingWarps = numWarps;
  unsigned prevLanes = 1;
  unsigned prevWarps = 1;

  // starting from the contiguous dimension
  for (unsigned d = 0; d < rank - 1; ++d) {
    unsigned i = order[d];
    unsigned threadsPerCTA = std::clamp<unsigned>(remainingThreads, 1, std::max<unsigned>(1, shapePerCTA[i] / sizePerThread[i]));
    threadsPerWarp[i] = std::clamp<unsigned>(threadsPerCTA, 1, remainingLanes);
    warpsPerCTA[i] = std::clamp<unsigned>(threadsPerCTA / threadsPerWarp[i], 1, remainingWarps);
    remainingWarps /= warpsPerCTA[i];
    remainingLanes /= threadsPerWarp[i];
    remainingThreads /= threadsPerCTA;
    prevLanes *= threadsPerWarp[i];
    prevWarps *= warpsPerCTA[i];
  }

  // Expand the last dimension to fill the remaining lanes and warps
  threadsPerWarp[order[rank - 1]] = numThreadsPerWarp / prevLanes;
  warpsPerCTA[order[rank - 1]] = numWarps / prevWarps;

  return Base::getChecked(emitError, context, sizePerThread, threadsPerWarp, warpsPerCTA, order, CGALayout);
}

BlockedEncodingAttr BlockedEncodingAttr::get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order, unsigned numWarps, unsigned numThreadsPerWarp, unsigned numCTAs) {
  unsigned rank = sizePerThread.size();
  SmallVector<unsigned, 4> CTAsPerCGA(rank);
  SmallVector<unsigned, 4> CTASplitNum(rank);
  ArrayRef<unsigned> CTAOrder = order;

  unsigned remainingCTAs = numCTAs;

  // starting from the most strided dimension
  for (int d = rank - 1; d >= 0; --d) {
    unsigned i = order[d];
    CTAsPerCGA[i] = std::clamp<unsigned>(remainingCTAs, 1, std::max<unsigned>(1, shape[i] / sizePerThread[i]));
    CTASplitNum[i] = CTAsPerCGA[i];
    remainingCTAs /= CTAsPerCGA[i];
  }

  CTAsPerCGA[rank - 1] *= remainingCTAs; // wrap at CTA level

  CGAEncodingAttr CGALayout = CGAEncodingAttr::fromSplitParams(context, CTAsPerCGA, CTASplitNum, CTAOrder);
  return get(context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, CGALayout);
}

BlockedEncodingAttr BlockedEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order, unsigned numWarps, unsigned numThreadsPerWarp, unsigned numCTAs) {
  unsigned rank = sizePerThread.size();
  SmallVector<unsigned, 4> CTAsPerCGA(rank);
  SmallVector<unsigned, 4> CTASplitNum(rank);
  ArrayRef<unsigned> CTAOrder = order;

  unsigned remainingCTAs = numCTAs;

  // starting from the most strided dimension
  for (int d = rank - 1; d >= 0; --d) {
    unsigned i = order[d];
    CTAsPerCGA[i] = std::clamp<unsigned>(remainingCTAs, 1, std::max<unsigned>(1, shape[i] / sizePerThread[i]));
    CTASplitNum[i] = CTAsPerCGA[i];
    remainingCTAs /= CTAsPerCGA[i];
  }

  CTAsPerCGA[rank - 1] *= remainingCTAs; // wrap at CTA level

  CGAEncodingAttr CGALayout = CGAEncodingAttr::fromSplitParams(context, CTAsPerCGA, CTASplitNum, CTAOrder);
  return get(context, shape, sizePerThread, order, numWarps, numThreadsPerWarp, CGALayout);
}

::llvm::LogicalResult BlockedEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<unsigned> sizePerThread, ::llvm::ArrayRef<unsigned> threadsPerWarp, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout) {
  if (::mlir::failed(verify(emitError, sizePerThread, threadsPerWarp, warpsPerCTA, order, CGALayout)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<unsigned> BlockedEncodingAttr::getSizePerThread() const {
  return getImpl()->sizePerThread;
}

::llvm::ArrayRef<unsigned> BlockedEncodingAttr::getThreadsPerWarp() const {
  return getImpl()->threadsPerWarp;
}

::llvm::ArrayRef<unsigned> BlockedEncodingAttr::getWarpsPerCTA() const {
  return getImpl()->warpsPerCTA;
}

::llvm::ArrayRef<unsigned> BlockedEncodingAttr::getOrder() const {
  return getImpl()->order;
}

CGAEncodingAttr BlockedEncodingAttr::getCGALayout() const {
  return getImpl()->CGALayout;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::BlockedEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct AMDMfmaEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, ::llvm::ArrayRef<unsigned>, ::llvm::ArrayRef<unsigned>, bool, CGAEncodingAttr, ::llvm::ArrayRef<unsigned>, unsigned>;
  AMDMfmaEncodingAttrStorage(unsigned version, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> instrShape, bool isTransposed, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> tilesPerWarp, unsigned elementBitWidth) : version(std::move(version)), warpsPerCTA(std::move(warpsPerCTA)), instrShape(std::move(instrShape)), isTransposed(std::move(isTransposed)), CGALayout(std::move(CGALayout)), tilesPerWarp(std::move(tilesPerWarp)), elementBitWidth(std::move(elementBitWidth)) {}

  KeyTy getAsKey() const {
    return KeyTy(version, warpsPerCTA, instrShape, isTransposed, CGALayout, tilesPerWarp, elementBitWidth);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (version == std::get<0>(tblgenKey)) && (warpsPerCTA == std::get<1>(tblgenKey)) && (instrShape == std::get<2>(tblgenKey)) && (isTransposed == std::get<3>(tblgenKey)) && (CGALayout == std::get<4>(tblgenKey)) && (tilesPerWarp == std::get<5>(tblgenKey)) && (elementBitWidth == std::get<6>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey));
  }

  static AMDMfmaEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto version = std::move(std::get<0>(tblgenKey));
    auto warpsPerCTA = std::move(std::get<1>(tblgenKey));
    auto instrShape = std::move(std::get<2>(tblgenKey));
    auto isTransposed = std::move(std::get<3>(tblgenKey));
    auto CGALayout = std::move(std::get<4>(tblgenKey));
    auto tilesPerWarp = std::move(std::get<5>(tblgenKey));
    auto elementBitWidth = std::move(std::get<6>(tblgenKey));
    warpsPerCTA = allocator.copyInto(warpsPerCTA);
    instrShape = allocator.copyInto(instrShape);
    tilesPerWarp = allocator.copyInto(tilesPerWarp);
    return new (allocator.allocate<AMDMfmaEncodingAttrStorage>()) AMDMfmaEncodingAttrStorage(std::move(version), std::move(warpsPerCTA), std::move(instrShape), std::move(isTransposed), std::move(CGALayout), std::move(tilesPerWarp), std::move(elementBitWidth));
  }

  unsigned version;
  ::llvm::ArrayRef<unsigned> warpsPerCTA;
  ::llvm::ArrayRef<unsigned> instrShape;
  bool isTransposed;
  CGAEncodingAttr CGALayout;
  ::llvm::ArrayRef<unsigned> tilesPerWarp;
  unsigned elementBitWidth;
};

} // namespace detail
AMDMfmaEncodingAttr AMDMfmaEncodingAttr::get(::mlir::MLIRContext *context, unsigned version, ArrayRef<unsigned> warpsPerCTA, ArrayRef<unsigned> instrShape, bool isTransposed, CGAEncodingAttr CGALayout, ArrayRef<unsigned> tpw, unsigned elementBitWidth) {
  SmallVector<unsigned> tilesPerWarp(tpw);
  if (tilesPerWarp.empty())
    tilesPerWarp = SmallVector<unsigned>(warpsPerCTA.size(), 1);
  if (elementBitWidth == 0)
    elementBitWidth = 32;
  return Base::get(context, version, warpsPerCTA, instrShape, isTransposed, CGALayout, tilesPerWarp, elementBitWidth);
}

AMDMfmaEncodingAttr AMDMfmaEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned version, ArrayRef<unsigned> warpsPerCTA, ArrayRef<unsigned> instrShape, bool isTransposed, CGAEncodingAttr CGALayout, ArrayRef<unsigned> tpw, unsigned elementBitWidth) {
  SmallVector<unsigned> tilesPerWarp(tpw);
  if (tilesPerWarp.empty())
    tilesPerWarp = SmallVector<unsigned>(warpsPerCTA.size(), 1);
  if (elementBitWidth == 0)
    elementBitWidth = 32;
  return Base::getChecked(emitError, context, version, warpsPerCTA, instrShape, isTransposed, CGALayout, tilesPerWarp, elementBitWidth);
}

::llvm::LogicalResult AMDMfmaEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned version, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> instrShape, bool isTransposed, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> tilesPerWarp, unsigned elementBitWidth) {
  if (::mlir::failed(verify(emitError, version, warpsPerCTA, instrShape, isTransposed, CGALayout, tilesPerWarp, elementBitWidth)))
    return ::mlir::failure();
  return ::mlir::success();
}

unsigned AMDMfmaEncodingAttr::getVersion() const {
  return getImpl()->version;
}

::llvm::ArrayRef<unsigned> AMDMfmaEncodingAttr::getWarpsPerCTA() const {
  return getImpl()->warpsPerCTA;
}

::llvm::ArrayRef<unsigned> AMDMfmaEncodingAttr::getInstrShape() const {
  return getImpl()->instrShape;
}

bool AMDMfmaEncodingAttr::getIsTransposed() const {
  return getImpl()->isTransposed;
}

CGAEncodingAttr AMDMfmaEncodingAttr::getCGALayout() const {
  return getImpl()->CGALayout;
}

::llvm::ArrayRef<unsigned> AMDMfmaEncodingAttr::getTilesPerWarp() const {
  return getImpl()->tilesPerWarp;
}

unsigned AMDMfmaEncodingAttr::getElementBitWidth() const {
  return getImpl()->elementBitWidth;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::AMDMfmaEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct AMDWmmaEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, bool, ::llvm::ArrayRef<unsigned>, ::llvm::ArrayRef<unsigned>, CGAEncodingAttr, ::llvm::ArrayRef<unsigned>>;
  AMDWmmaEncodingAttrStorage(unsigned version, bool isTransposed, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> tilesPerWarp, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape) : version(std::move(version)), isTransposed(std::move(isTransposed)), warpsPerCTA(std::move(warpsPerCTA)), tilesPerWarp(std::move(tilesPerWarp)), CGALayout(std::move(CGALayout)), instrShape(std::move(instrShape)) {}

  KeyTy getAsKey() const {
    return KeyTy(version, isTransposed, warpsPerCTA, tilesPerWarp, CGALayout, instrShape);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (version == std::get<0>(tblgenKey)) && (isTransposed == std::get<1>(tblgenKey)) && (warpsPerCTA == std::get<2>(tblgenKey)) && (tilesPerWarp == std::get<3>(tblgenKey)) && (CGALayout == std::get<4>(tblgenKey)) && (instrShape == std::get<5>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey));
  }

  static AMDWmmaEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto version = std::move(std::get<0>(tblgenKey));
    auto isTransposed = std::move(std::get<1>(tblgenKey));
    auto warpsPerCTA = std::move(std::get<2>(tblgenKey));
    auto tilesPerWarp = std::move(std::get<3>(tblgenKey));
    auto CGALayout = std::move(std::get<4>(tblgenKey));
    auto instrShape = std::move(std::get<5>(tblgenKey));
    warpsPerCTA = allocator.copyInto(warpsPerCTA);
    tilesPerWarp = allocator.copyInto(tilesPerWarp);
    instrShape = allocator.copyInto(instrShape);
    return new (allocator.allocate<AMDWmmaEncodingAttrStorage>()) AMDWmmaEncodingAttrStorage(std::move(version), std::move(isTransposed), std::move(warpsPerCTA), std::move(tilesPerWarp), std::move(CGALayout), std::move(instrShape));
  }

  unsigned version;
  bool isTransposed;
  ::llvm::ArrayRef<unsigned> warpsPerCTA;
  ::llvm::ArrayRef<unsigned> tilesPerWarp;
  CGAEncodingAttr CGALayout;
  ::llvm::ArrayRef<unsigned> instrShape;
};

} // namespace detail
AMDWmmaEncodingAttr AMDWmmaEncodingAttr::get(::mlir::MLIRContext *context, unsigned version, bool isTransposed, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> tilesPerWarp, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape) {
  return Base::get(context, std::move(version), std::move(isTransposed), std::move(warpsPerCTA), std::move(tilesPerWarp), std::move(CGALayout), std::move(instrShape));
}

AMDWmmaEncodingAttr AMDWmmaEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned version, bool isTransposed, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> tilesPerWarp, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape) {
  return Base::getChecked(emitError, context, std::move(version), std::move(isTransposed), std::move(warpsPerCTA), std::move(tilesPerWarp), std::move(CGALayout), std::move(instrShape));
}

AMDWmmaEncodingAttr AMDWmmaEncodingAttr::get(::mlir::MLIRContext *context, unsigned version, bool isTransposed, ArrayRef<unsigned> warpsPerCTA, CGAEncodingAttr CGALayout, ArrayRef<unsigned> instrShape) {
  SmallVector<unsigned> tilesPerWarp(warpsPerCTA.size(), 1);
  return Base::get(context, version, isTransposed, warpsPerCTA, tilesPerWarp, CGALayout, instrShape);
}

AMDWmmaEncodingAttr AMDWmmaEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned version, bool isTransposed, ArrayRef<unsigned> warpsPerCTA, CGAEncodingAttr CGALayout, ArrayRef<unsigned> instrShape) {
  SmallVector<unsigned> tilesPerWarp(warpsPerCTA.size(), 1);
  return Base::getChecked(emitError, context, version, isTransposed, warpsPerCTA, tilesPerWarp, CGALayout, instrShape);
}

::llvm::LogicalResult AMDWmmaEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned version, bool isTransposed, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> tilesPerWarp, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape) {
  if (::mlir::failed(verify(emitError, version, isTransposed, warpsPerCTA, tilesPerWarp, CGALayout, instrShape)))
    return ::mlir::failure();
  return ::mlir::success();
}

unsigned AMDWmmaEncodingAttr::getVersion() const {
  return getImpl()->version;
}

bool AMDWmmaEncodingAttr::getIsTransposed() const {
  return getImpl()->isTransposed;
}

::llvm::ArrayRef<unsigned> AMDWmmaEncodingAttr::getWarpsPerCTA() const {
  return getImpl()->warpsPerCTA;
}

::llvm::ArrayRef<unsigned> AMDWmmaEncodingAttr::getTilesPerWarp() const {
  return getImpl()->tilesPerWarp;
}

CGAEncodingAttr AMDWmmaEncodingAttr::getCGALayout() const {
  return getImpl()->CGALayout;
}

::llvm::ArrayRef<unsigned> AMDWmmaEncodingAttr::getInstrShape() const {
  return getImpl()->instrShape;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::AMDWmmaEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct NvidiaMmaEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, unsigned, ::llvm::ArrayRef<unsigned>, CGAEncodingAttr, ::llvm::ArrayRef<unsigned>>;
  NvidiaMmaEncodingAttrStorage(unsigned versionMajor, unsigned versionMinor, ::llvm::ArrayRef<unsigned> warpsPerCTA, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape) : versionMajor(std::move(versionMajor)), versionMinor(std::move(versionMinor)), warpsPerCTA(std::move(warpsPerCTA)), CGALayout(std::move(CGALayout)), instrShape(std::move(instrShape)) {}

  KeyTy getAsKey() const {
    return KeyTy(versionMajor, versionMinor, warpsPerCTA, CGALayout, instrShape);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (versionMajor == std::get<0>(tblgenKey)) && (versionMinor == std::get<1>(tblgenKey)) && (warpsPerCTA == std::get<2>(tblgenKey)) && (CGALayout == std::get<3>(tblgenKey)) && (instrShape == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static NvidiaMmaEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto versionMajor = std::move(std::get<0>(tblgenKey));
    auto versionMinor = std::move(std::get<1>(tblgenKey));
    auto warpsPerCTA = std::move(std::get<2>(tblgenKey));
    auto CGALayout = std::move(std::get<3>(tblgenKey));
    auto instrShape = std::move(std::get<4>(tblgenKey));
    warpsPerCTA = allocator.copyInto(warpsPerCTA);
    instrShape = allocator.copyInto(instrShape);
    return new (allocator.allocate<NvidiaMmaEncodingAttrStorage>()) NvidiaMmaEncodingAttrStorage(std::move(versionMajor), std::move(versionMinor), std::move(warpsPerCTA), std::move(CGALayout), std::move(instrShape));
  }

  unsigned versionMajor;
  unsigned versionMinor;
  ::llvm::ArrayRef<unsigned> warpsPerCTA;
  CGAEncodingAttr CGALayout;
  ::llvm::ArrayRef<unsigned> instrShape;
};

} // namespace detail
NvidiaMmaEncodingAttr NvidiaMmaEncodingAttr::get(::mlir::MLIRContext *context, unsigned versionMajor, unsigned versionMinor, ::llvm::ArrayRef<unsigned> warpsPerCTA, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape) {
  return Base::get(context, std::move(versionMajor), std::move(versionMinor), std::move(warpsPerCTA), std::move(CGALayout), std::move(instrShape));
}

unsigned NvidiaMmaEncodingAttr::getVersionMajor() const {
  return getImpl()->versionMajor;
}

unsigned NvidiaMmaEncodingAttr::getVersionMinor() const {
  return getImpl()->versionMinor;
}

::llvm::ArrayRef<unsigned> NvidiaMmaEncodingAttr::getWarpsPerCTA() const {
  return getImpl()->warpsPerCTA;
}

CGAEncodingAttr NvidiaMmaEncodingAttr::getCGALayout() const {
  return getImpl()->CGALayout;
}

::llvm::ArrayRef<unsigned> NvidiaMmaEncodingAttr::getInstrShape() const {
  return getImpl()->instrShape;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::NvidiaMmaEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct SliceEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, DistributedEncodingTrait>;
  SliceEncodingAttrStorage(unsigned dim, DistributedEncodingTrait parent) : dim(std::move(dim)), parent(std::move(parent)) {}

  KeyTy getAsKey() const {
    return KeyTy(dim, parent);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (dim == std::get<0>(tblgenKey)) && (parent == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static SliceEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto dim = std::move(std::get<0>(tblgenKey));
    auto parent = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<SliceEncodingAttrStorage>()) SliceEncodingAttrStorage(std::move(dim), std::move(parent));
  }

  unsigned dim;
  DistributedEncodingTrait parent;
};

} // namespace detail
SliceEncodingAttr SliceEncodingAttr::get(::mlir::MLIRContext *context, unsigned dim, DistributedEncodingTrait parent) {
  return Base::get(context, std::move(dim), std::move(parent));
}

SliceEncodingAttr SliceEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned dim, DistributedEncodingTrait parent) {
  return Base::getChecked(emitError, context, std::move(dim), std::move(parent));
}

::llvm::LogicalResult SliceEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned dim, DistributedEncodingTrait parent) {
  if (::mlir::failed(verify(emitError, dim, parent)))
    return ::mlir::failure();
  return ::mlir::success();
}

unsigned SliceEncodingAttr::getDim() const {
  return getImpl()->dim;
}

DistributedEncodingTrait SliceEncodingAttr::getParent() const {
  return getImpl()->parent;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::SliceEncodingAttr)
namespace mlir::triton::gpu {

namespace detail {

struct DotOperandEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, Attribute, unsigned>;
  DotOperandEncodingAttrStorage(unsigned opIdx, Attribute parent, unsigned kWidth) : opIdx(std::move(opIdx)), parent(std::move(parent)), kWidth(std::move(kWidth)) {}

  KeyTy getAsKey() const {
    return KeyTy(opIdx, parent, kWidth);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (opIdx == std::get<0>(tblgenKey)) && (parent == std::get<1>(tblgenKey)) && (kWidth == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static DotOperandEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto opIdx = std::move(std::get<0>(tblgenKey));
    auto parent = std::move(std::get<1>(tblgenKey));
    auto kWidth = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<DotOperandEncodingAttrStorage>()) DotOperandEncodingAttrStorage(std::move(opIdx), std::move(parent), std::move(kWidth));
  }

  unsigned opIdx;
  Attribute parent;
  unsigned kWidth;
};

} // namespace detail
DotOperandEncodingAttr DotOperandEncodingAttr::get(::mlir::MLIRContext *context, unsigned opIdx, Attribute parent, unsigned kWidth) {
  return Base::get(context, std::move(opIdx), std::move(parent), std::move(kWidth));
}

DotOperandEncodingAttr DotOperandEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned opIdx, Attribute parent, unsigned kWidth) {
  return Base::getChecked(emitError, context, std::move(opIdx), std::move(parent), std::move(kWidth));
}

DotOperandEncodingAttr DotOperandEncodingAttr::get(::mlir::MLIRContext *context, unsigned opIdx, Attribute parent, Type eltTy) {
  NvidiaMmaEncodingAttr parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent);
  if (!parentAttr || (!parentAttr.isAmpere() && !parentAttr.isHopper()))
    return Base::get(context, opIdx, parent, 0);
  // For MMAV2 and V3
  unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
  unsigned kWidth = std::max(32 / bitwidth, 1u);
  return Base::get(context, opIdx, parent, kWidth);
}

DotOperandEncodingAttr DotOperandEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned opIdx, Attribute parent, Type eltTy) {
  NvidiaMmaEncodingAttr parentAttr = mlir::dyn_cast<NvidiaMmaEncodingAttr>(parent);
  if (!parentAttr || (!parentAttr.isAmpere() && !parentAttr.isHopper()))
    return Base::getChecked(emitError, context, opIdx, parent, 0);
  // For MMAV2 and V3
  unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
  unsigned kWidth = std::max(32 / bitwidth, 1u);
  return Base::getChecked(emitError, context, opIdx, parent, kWidth);
}

::llvm::LogicalResult DotOperandEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned opIdx, Attribute parent, unsigned kWidth) {
  if (::mlir::failed(verify(emitError, opIdx, parent, kWidth)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute DotOperandEncodingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<unsigned> _result_opIdx;
  ::mlir::FailureOr<Attribute> _result_parent;
  ::mlir::FailureOr<unsigned> _result_kWidth;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal '{'
  if (odsParser.parseLBrace()) return {};
  // Parse parameter struct
  bool _seen_opIdx = false;
  bool _seen_parent = false;
  bool _seen_kWidth = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_opIdx && _paramKey == "opIdx") {
        _seen_opIdx = true;

        // Parse variable 'opIdx'
        _result_opIdx = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_opIdx)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DotOperandEncodingAttr parameter 'opIdx' which is to be a `unsigned`");
          return {};
        }
      } else if (!_seen_parent && _paramKey == "parent") {
        _seen_parent = true;

        // Parse variable 'parent'
        _result_parent = ::mlir::FieldParser<Attribute>::parse(odsParser);
        if (::mlir::failed(_result_parent)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DotOperandEncodingAttr parameter 'parent' which is to be a `Attribute`");
          return {};
        }
      } else if (!_seen_kWidth && _paramKey == "kWidth") {
        _seen_kWidth = true;

        // Parse variable 'kWidth'
        _result_kWidth = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_kWidth)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DotOperandEncodingAttr parameter 'kWidth' which is to be a `unsigned`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    do {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
    } while(!odsParser.parseOptionalComma());
    if (!_seen_opIdx) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "opIdx";
      return {};
    }
    if (!_seen_parent) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "parent";
      return {};
    }
  }
  // Parse literal '}'
  if (odsParser.parseRBrace()) return {};
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_opIdx));
  assert(::mlir::succeeded(_result_parent));
  return odsParser.getChecked<DotOperandEncodingAttr>(odsLoc, odsParser.getContext(),
      unsigned((*_result_opIdx)),
      Attribute((*_result_parent)),
      unsigned((_result_kWidth.value_or(0))));
}

void DotOperandEncodingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "{";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "opIdx = ";
    odsPrinter.printStrippedAttrOrType(getOpIdx());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "parent = ";
    odsPrinter.printStrippedAttrOrType(getParent());
    if (!(getKWidth() == 0)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "kWidth = ";
      if (!(getKWidth() == 0)) {
        odsPrinter.printStrippedAttrOrType(getKWidth());
      }
    }
  }
  odsPrinter << "}";
  odsPrinter << ">";
}

unsigned DotOperandEncodingAttr::getOpIdx() const {
  return getImpl()->opIdx;
}

Attribute DotOperandEncodingAttr::getParent() const {
  return getImpl()->parent;
}

unsigned DotOperandEncodingAttr::getKWidth() const {
  return getImpl()->kWidth;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::DotOperandEncodingAttr)
namespace mlir::triton::gpu {


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::SharedMemorySpaceAttr)
namespace mlir::triton::gpu {


/// Parse an attribute registered to this dialect.
::mlir::Attribute TritonGPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void TritonGPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::triton::gpu

#endif // GET_ATTRDEF_CLASSES

