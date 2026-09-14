/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonGPUDialect.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::gpu {

/// Triton GPU Dialect.
class TritonGPUDialect : public ::mlir::Dialect {
  explicit TritonGPUDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TritonGPUDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("ttg");
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

    /// Provides a hook for verifying dialect attributes attached to the given
    /// op.
    ::llvm::LogicalResult verifyOperationAttribute(
        ::mlir::Operation *op, ::mlir::NamedAttribute attribute) override;

    void registerTypes();

    LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout);
    LinearEncodingAttr toLinearEncoding(ArrayRef<int64_t> shape, Attribute layout);

    static int getNumCTAs(ModuleOp mod);
    static int getThreadsPerWarp(ModuleOp mod);

    private:
      LinearLayoutCache llCache;
      LinearEncodingCache leCache;
  };

} // namespace mlir::triton::gpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::TritonGPUDialect)
