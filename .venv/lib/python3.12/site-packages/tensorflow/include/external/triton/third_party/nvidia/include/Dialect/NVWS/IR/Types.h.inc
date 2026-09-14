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
namespace mlir::triton::nvws {

/// Asynchronous Reference
/// A meta-type that holds an asynchronous reference to an underlying Type.
/// 
///         Can wrap multiple underlying values simultaneously.
/// 
///         Useful for syncing asynchronous operations while doing transformations such
///         as pipelining and warp specialization. Lowers to the underlying type, and
///         operations that use this should insert appropriate barriers during lowering.
class ArefType;
class TokenType;
namespace detail {

struct ArefTypeStorage;

} // namespace detail
class ArefType : public ::mlir::Type::TypeBase<ArefType, ::mlir::Type, detail::ArefTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvws.aref";
  static constexpr ::llvm::StringLiteral dialectName = "nvws";
  static ArefType get(::mlir::MLIRContext *context, TypeArrayAttr baseType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"aref"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  TypeArrayAttr getBaseType() const;
};
class TokenType : public ::mlir::Type::TypeBase<TokenType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvws.token";
  static constexpr ::llvm::StringLiteral dialectName = "nvws";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"token"};
  }

};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ArefType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::TokenType)

#endif // GET_TYPEDEF_CLASSES

