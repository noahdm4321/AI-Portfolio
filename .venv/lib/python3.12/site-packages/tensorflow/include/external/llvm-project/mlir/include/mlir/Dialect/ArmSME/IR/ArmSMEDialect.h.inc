/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ArmSME.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::arm_sme {

/// Basic dialect to target Arm SME architectures
/// This dialect contains the definitions necessary to target Arm SME
///     scalable matrix operations.
/// 
///     Sources:
///     https://developer.arm.com/documentation/ddi0616
///     https://developer.arm.com/documentation/ddi0602/2023-03/SME-Instructions
class ArmSMEDialect : public ::mlir::Dialect {
  explicit ArmSMEDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ArmSMEDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("arm_sme");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;
};

} // namespace mlir::arm_sme
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::arm_sme::ArmSMEDialect)
