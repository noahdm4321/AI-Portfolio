/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: tf_framework_ops.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::kernel_gen::tf_framework {

/// Types and operations for tf_framework dialect
/// This dialect contains operations and types for that correspond to
///     TensorFlow C++ Framework.
class TFFrameworkDialect : public ::mlir::Dialect {
  explicit TFFrameworkDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TFFrameworkDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("tf_framework");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

    static constexpr StringRef kTFEntryAttrName = "tf_entry";
    static constexpr size_t kAllocationAlignment = 16;
  };

} // namespace mlir::kernel_gen::tf_framework
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::kernel_gen::tf_framework::TFFrameworkDialect)
