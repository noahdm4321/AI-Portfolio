/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: GluonDialect.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::gluon {

/// Gluon dialect.
class GluonDialect : public ::mlir::Dialect {
  explicit GluonDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~GluonDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("gluon");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;
};

} // namespace mlir::triton::gluon
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gluon::GluonDialect)
