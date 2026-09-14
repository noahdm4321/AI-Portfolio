/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonAMDGPUDialect.td                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::amdgpu {

/// TritonAMDGPU Dialect hosts AMD specific ops at TritonGPU abstraction level.
class TritonAMDGPUDialect : public ::mlir::Dialect {
  explicit TritonAMDGPUDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TritonAMDGPUDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("amdg");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;
};

} // namespace mlir::triton::amdgpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::amdgpu::TritonAMDGPUDialect)
