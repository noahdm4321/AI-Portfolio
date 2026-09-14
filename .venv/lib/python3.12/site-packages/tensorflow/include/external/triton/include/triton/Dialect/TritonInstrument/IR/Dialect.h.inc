/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonInstrumentDialect.td                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::instrument {


class TritonInstrumentDialect : public ::mlir::Dialect {
  explicit TritonInstrumentDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TritonInstrumentDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("tti");
  }
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::TritonInstrumentDialect)
