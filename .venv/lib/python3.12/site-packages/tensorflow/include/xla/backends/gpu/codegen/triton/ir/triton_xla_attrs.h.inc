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
namespace mlir::triton::xla {

/// tensor map swizzle mode
class SwizzleModeAttr;
/// Contains the necessary data to generate TMA descriptors.
class TmaDescriptorAttr;
/// A comparison operator for instructions.
/// Enum cases:
/// * equal_to (`EQ`)
/// * less_than (`LT`)
class ComparatorAttr;
namespace detail {

struct SwizzleModeAttrStorage;

} // namespace detail
class SwizzleModeAttr : public ::mlir::Attribute::AttrBase<SwizzleModeAttr, ::mlir::Attribute, detail::SwizzleModeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton_xla.swizzle_mode";
  static constexpr ::llvm::StringLiteral dialectName = "triton_xla";
  static SwizzleModeAttr get(::mlir::MLIRContext *context, ::mlir::triton::xla::SwizzleMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"swizzle_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::xla::SwizzleMode getValue() const;
};
namespace detail {

struct TmaDescriptorAttrStorage;

} // namespace detail
class TmaDescriptorAttr : public ::mlir::Attribute::AttrBase<TmaDescriptorAttr, ::mlir::Attribute, detail::TmaDescriptorAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton_xla.tma_descriptor";
  static constexpr ::llvm::StringLiteral dialectName = "triton_xla";
  static TmaDescriptorAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> global_shape, ::llvm::ArrayRef<int64_t> tile_shape, ::llvm::ArrayRef<int64_t> tile_strides, ::llvm::ArrayRef<int64_t> layout, int element_byte_size, SwizzleModeAttr swizzle_mode);
  static TmaDescriptorAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> global_shape, ArrayRef<int64_t> tile_shape, ArrayRef<int64_t> tile_strides, ArrayRef<int64_t> layout, int element_byte_size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tma_descriptor"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getGlobalShape() const;
  ::llvm::ArrayRef<int64_t> getTileShape() const;
  ::llvm::ArrayRef<int64_t> getTileStrides() const;
  ::llvm::ArrayRef<int64_t> getLayout() const;
  int getElementByteSize() const;
  SwizzleModeAttr getSwizzleMode() const;
};
namespace detail {

struct ComparatorAttrStorage;

} // namespace detail
class ComparatorAttr : public ::mlir::Attribute::AttrBase<ComparatorAttr, ::mlir::Attribute, detail::ComparatorAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton_xla.";
  static constexpr ::llvm::StringLiteral dialectName = "triton_xla";
  static ComparatorAttr get(::mlir::MLIRContext *context, ::mlir::triton::xla::Comparator value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {""};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::xla::Comparator getValue() const;
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::SwizzleModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::TmaDescriptorAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::ComparatorAttr)

#endif // GET_ATTRDEF_CLASSES

