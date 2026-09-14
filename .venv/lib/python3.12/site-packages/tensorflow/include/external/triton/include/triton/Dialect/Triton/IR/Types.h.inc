/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::triton {

/// Pointer type (`::mlir::triton::PointerType`) in Triton IR type system
/// Pointer type in Triton IR type system, which could be pointing to scalars or tensors.
class PointerType;
/// Tensor descriptor type (`::mlir::triton::TensorDescType`) in Triton IR type system
/// A portable abstraction for nvidia-TMA descriptors.
class TensorDescType;
namespace detail {

struct PointerTypeStorage;

} // namespace detail
class PointerType : public ::mlir::Type::TypeBase<PointerType, ::mlir::Type, detail::PointerTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tt.ptr";
  static constexpr ::llvm::StringLiteral dialectName = "tt";
  static PointerType get(Type pointeeType, int addressSpace);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ptr"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getPointeeType() const;
  int getAddressSpace() const;
};
namespace detail {

struct TensorDescTypeStorage;

} // namespace detail
class TensorDescType : public ::mlir::Type::TypeBase<TensorDescType, ::mlir::Type, detail::TensorDescTypeStorage> {
public:
  using Base::Base;
  RankedTensorType getSignlessBlockType() const {
    auto resTy = getBlockType();
    if (auto intTy = llvm::dyn_cast<IntegerType>(resTy.getElementType())) {
      auto width = resTy.getElementTypeBitWidth();
      auto signlessTy = IntegerType::get(getContext(), width);
      resTy = resTy.clone(signlessTy);
    }
    return resTy;
  }
  static constexpr ::llvm::StringLiteral name = "tt.tensordesc";
  static constexpr ::llvm::StringLiteral dialectName = "tt";
  static TensorDescType get(::mlir::MLIRContext *context, RankedTensorType blockType);
  static TensorDescType get(::mlir::MLIRContext *context, RankedTensorType blockType, bool isSigned);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensordesc"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  RankedTensorType getBlockType() const;
};

} // namespace mlir::triton
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::PointerType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::TensorDescType)

#endif // GET_TYPEDEF_CLASSES

