/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: triton_xla_dialect.td                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::xla {

/// This dialect contains ops included in the xla extension point for Triton.
class XlaTritonDialect : public ::mlir::Dialect {
  explicit XlaTritonDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~XlaTritonDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("triton_xla");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::XlaTritonDialect)
