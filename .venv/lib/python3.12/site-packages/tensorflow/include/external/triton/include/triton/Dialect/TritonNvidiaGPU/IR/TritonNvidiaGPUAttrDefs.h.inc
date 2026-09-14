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
namespace mlir::triton::nvidia_gpu {

/// Attribute to indicate that the memory descriptor points to tensor memory.
///     The memory is laid out in blocks of size blockM x blockN. Each block is distributed
///     across TMEM 128 rows.
/// 
///     Blocks are distributed along M dimension first and then N dimension. This is an arbitrary
///     convention that needs to be followed by operations reading/writing to TMEM.
/// 
///     a tensor <128x128xf32> with blockM = 64 and blockN = 32 will be distributed as follows:
/// 
///         \ col    0        1            31         32            64            96           127
///     rows: 0  ( 0,  0) ( 0,  1) ... ( 0,  31)  ( 0,  32) ... ( 0,  64) ... ( 0,  96) ... ( 0,  127)
///           1
///          ...
///           15 (15,  0) (15,  1) ... (15,  31)  (15,  32) ... (15,  64) ... (15,  96) ... (15,  127)
///           16 (64,  0) (64,  1) ... (64,  31)  (64,  32) ... (64,  64) ... (64,  96) ... (64,  127)
///          ...
///           31 (79,  0) (79,  1) ... (79,  31)  (79,  32) ... (79,  64) ... (79,  96) ... (79,  127)
///           32 (16,  0) (16,  1) ... (16,  31)  (16,  32) ... (16,  64) ... (16,  96) ... (16,  127)
///          ..
///          127 (127, 0) (127, 1) ... (127, 31) (127, 32) ... (127, 64) ... (127, 96) ... (127, 127)
class TensorMemorySpaceAttr;
/// An encoding to represent the different way the tensor memory is laid out.
///     `colStride` describes the stride in elements along the column dimension,
///     that is, the stride between two elements in the same row.
///     When colStride is 1 the tensor memory is packed. When colStride > 1, the
///     tensor memory between elements is undefined.
///     `twoCTAs` indicates that the tensor memory is laid out for twoCTA mode,
///     i.e., `cta_group::2`.
class TensorMemoryEncodingAttr;
/// An encoding to represent the layout of tensor memory scales.
///     As described in the PTX doc, blocked scales in TMEM must be in a special layout. They are organized
///     as a multiple copies of "chunk", each of which having the size 32x4x4B. Moreover, such chunks are duplicated
///     over 4 warps to fill entire 128 rows of TMEM. This encoding indicates that a tensor in TMEM is in such a special
///     layout.
class TensorMemoryScalesEncodingAttr;
class TensorMemorySpaceAttr : public ::mlir::Attribute::AttrBase<TensorMemorySpaceAttr, ::mlir::Attribute, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ttng.tensor_memory";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory"};
  }

};
namespace detail {

struct TensorMemoryEncodingAttrStorage;

} // namespace detail
class TensorMemoryEncodingAttr : public ::mlir::Attribute::AttrBase<TensorMemoryEncodingAttr, ::mlir::Attribute, detail::TensorMemoryEncodingAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.tensor_memory_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  using Base::getChecked;
  static TensorMemoryEncodingAttr get(::mlir::MLIRContext *context, unsigned blockM, unsigned blockN, unsigned colStride, unsigned CTASplitM, unsigned CTASplitN, bool twoCTAs);
  static TensorMemoryEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned blockM, unsigned blockN, unsigned colStride, unsigned CTASplitM, unsigned CTASplitN, bool twoCTAs);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned blockM, unsigned blockN, unsigned colStride, unsigned CTASplitM, unsigned CTASplitN, bool twoCTAs);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned blockM, unsigned blockN, unsigned colStride, unsigned CTASplitM, unsigned CTASplitN, bool twoCTAs);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory_encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getBlockM() const;
  unsigned getBlockN() const;
  unsigned getColStride() const;
  unsigned getCTASplitM() const;
  unsigned getCTASplitN() const;
  bool getTwoCTAs() const;
};
namespace detail {

struct TensorMemoryScalesEncodingAttrStorage;

} // namespace detail
class TensorMemoryScalesEncodingAttr : public ::mlir::Attribute::AttrBase<TensorMemoryScalesEncodingAttr, ::mlir::Attribute, detail::TensorMemoryScalesEncodingAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.tensor_memory_scales_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttng";
  static TensorMemoryScalesEncodingAttr get(::mlir::MLIRContext *context, unsigned CTASplitM, unsigned CTASplitN);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_memory_scales_encoding"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getCTASplitM() const;
  unsigned getCTASplitN() const;
};

} // namespace mlir::triton::nvidia_gpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr)

#endif // GET_ATTRDEF_CLASSES

