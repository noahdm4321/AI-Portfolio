/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TosaOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::tosa {

/// The Tensor Operator Set Architecture (TOSA) dialect.
/// 
///     This dialect implements the TOSA standard described at
///     https://www.mlplatform.org/tosa/tosa_spec.html .
/// 
///     Tensor Operator Set Architecture (TOSA) provides a set of whole-tensor
///     operations commonly employed by Deep Neural Networks. The intent is to
///     enable a variety of implementations running on a diverse range of
///     processors, with the results at the TOSA level consistent across those
///     implementations. Applications or frameworks which target TOSA can therefore
///     be deployed on a wide range of different processors, such as CPUs or GPUs,
///     with defined accuracy and compatibility constraints. Most operators from the
///     common ML frameworks should be expressible in TOSA. It is expected that
///     there will be tools to lower from the ML frameworks into TOSA.
class TosaDialect : public ::mlir::Dialect {
  explicit TosaDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TosaDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("tosa");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc) override;
};

} // namespace mlir::tosa
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::TosaDialect)
