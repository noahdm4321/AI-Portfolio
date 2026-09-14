/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVGPUDialect.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvgpu {

/// NVGPU Dialect.
class NVGPUDialect : public ::mlir::Dialect {
  explicit NVGPUDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~NVGPUDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("nvg");
  }
};

} // namespace mlir::triton::nvgpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::NVGPUDialect)
