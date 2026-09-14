/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ProtonOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::proton {

/// Proton Dialect provides core ops for building third-party compiler-based
///     performance profiling and analysis tools.
class ProtonDialect : public ::mlir::Dialect {
  explicit ProtonDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ProtonDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("proton");
  }
};

} // namespace mlir::triton::proton
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::proton::ProtonDialect)
