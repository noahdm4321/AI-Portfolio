/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: XeVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::xevm {

/// The XeVM dialect that extends LLVM dialect and models Intel GPU's hardware features.
/// The XeVM dialect is extension to the LLVM dialect that models hardware
///     features of Intel GPUs. The dialect is designed to work with the Xe
///     architecture for Intel GPUs, supporting advanced operations like 2D block
///     loads, stores, prefetch and matrix multiply-add (MMA) operations.
class XeVMDialect : public ::mlir::Dialect {
  explicit XeVMDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~XeVMDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("xevm");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

    /// Get the name for the attribute used to specify cache control
    /// decorations.
    static constexpr ::llvm::StringRef getCacheControlsAttrName() {
      return ::llvm::StringLiteral("xevm.DecorationCacheControl");
    }
  };

} // namespace mlir::xevm
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xevm::XeVMDialect)
