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
namespace mlir::tosa {

/// INT8 type as defined by OCP-MX
/// 8-bit integer format with an implicit 1/64 scale defined by OCP-MX.
class mxint8Type;
/// Shape with static rank and Index element type
/// Syntax:
/// 
///     ``` shape - type :: = `shape` `<` rank `>`
///     ``` Values with shape type represents a shape with a fixed rank and a list
///                                of dimensions
///                                    .Rank must be zero or a positive integer
///                                    .Each dimension is represented by the builtin
///                                        Index type.
/// 
///     Examples:
/// 
///     ```mlir
///      // Shape with rank of four, for example, [1, 1, 8, 16]:
///      !tosa
///          .shape<4>
/// 
///      // Shape with rank of one, for example, [16]:
///      !tosa
///          .shape<1>
/// 
///      // Shape with rank zero, for example, [] (i.e., shape of scalar values):
///      !tosa.shape<0>
///     ```
class shapeType;
class mxint8Type : public ::mlir::Type::TypeBase<mxint8Type, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.mxint8";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mxint8"};
  }

};
namespace detail {

struct shapeTypeStorage;

} // namespace detail
class shapeType : public ::mlir::Type::TypeBase<shapeType, ::mlir::Type, detail::shapeTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.shape";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  using Base::getChecked;
  static shapeType get(::mlir::MLIRContext *context, int rank);
  static shapeType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int rank);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int rank);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int rank);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shape"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getRank() const;
};

} // namespace mlir::tosa
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::mxint8Type)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::shapeType)

#endif // GET_TYPEDEF_CLASSES

