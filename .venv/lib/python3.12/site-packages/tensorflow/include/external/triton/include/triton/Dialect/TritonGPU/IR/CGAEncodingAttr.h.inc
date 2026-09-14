/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::triton::gpu {

/// Describes how blocks (CTAs) in a cooperative thread array (CGA) map onto logical
/// tensor dimensions. The `LinearLayout` maps from `block` into `dim0`, `dim1`...
class CGAEncodingAttr;
namespace detail {

struct CGAEncodingAttrStorage;

} // namespace detail
class CGAEncodingAttr : public ::mlir::Attribute::AttrBase<CGAEncodingAttr, ::mlir::Attribute, detail::CGAEncodingAttrStorage> {
public:
  using Base::Base;
  static CGAEncodingAttr getDefault(MLIRContext *context, int rank);
  // Legacy, we should kill this! Note that it is not true in general that
  // fromSplitParams(enc.getCTAsPerCGA(), enc.getCTASplitNum(), enc.getCTAOrder()) == enc!!
  static CGAEncodingAttr fromSplitParams(MLIRContext *context,
                                         ArrayRef<unsigned> CTAsPerCGA,
                                         ArrayRef<unsigned> CTASplitNum,
                                         ArrayRef<unsigned> CTAOrder);

  unsigned getRank() const { return getLinearLayout().getNumOutDims(); }
  SmallVector<unsigned> getCTAsPerCGA() const;
  SmallVector<unsigned> getCTASplitNum() const;
  SmallVector<unsigned> getCTAOrder() const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.cga_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  static CGAEncodingAttr get(::mlir::MLIRContext *context, LinearLayout linearLayout);
  static CGAEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, LinearLayout linearLayout);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout);
  const LinearLayout &getLinearLayout() const;
};

} // namespace mlir::triton::gpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::CGAEncodingAttr)

#endif // GET_ATTRDEF_CLASSES

