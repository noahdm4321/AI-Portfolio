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
namespace mlir::acc {

/// Type for representing acc data clause bounds information
class DataBoundsType;
/// declare token type
/// `acc.declare_token` is a type returned by a `declare_enter` operation and
///     can be passed to a `declare_exit` operation to represent an implicit
///     data region.
class DeclareTokenType;
class DataBoundsType : public ::mlir::Type::TypeBase<DataBoundsType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.data_bounds_ty";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_bounds_ty"};
  }

};
class DeclareTokenType : public ::mlir::Type::TypeBase<DeclareTokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.declare_token";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"declare_token"};
  }

};

} // namespace mlir::acc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DataBoundsType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DeclareTokenType)

#endif // GET_TYPEDEF_CLASSES

