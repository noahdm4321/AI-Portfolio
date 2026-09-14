/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: XeGPU.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::xegpu {

/// The XeGPU dialect that models Intel GPU's ISA
/// The XeGPU dialect closely models a subset of the Xe GPU's ISA, providing an
///       abstraction to support high-performance GEMM code generation. It serves as a
///       bridge dialect in the MLIR gradual lowering process, working with MLIR memref
///       and vector types, and complements the Arith, Math, Vector, and Memref dialects.
///       XeGPU operations are introduced for special Xe instructions not modeled by the
///       LLVM/SPIR-V dialect, such as DPAS and 2D block load and store.
/// 
///       It supports a tile-based programming model, decomposing the GEMM kernel into
///       large predefined tile sizes at the subgroup and workgroup levels. XeGPU allows
///       the high-level GEMM algorithm to be easily expressed. Underneath, it uses
///       target-specific recipes and hardware features to achieve optimal performance
///       on specific hardware. By decomposing GEMM at submatrix granularity and mapping it
///       to registers, it naturally supports optimizations like fusing with neighboring
///       operations.
class XeGPUDialect : public ::mlir::Dialect {
  explicit XeGPUDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~XeGPUDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("xegpu");
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

      /// Checks if the given shape can be evenly distributed based on the layout
      /// and data factors provided by the LayoutAttr.
      static bool isEvenlyDistributable(llvm::ArrayRef<int64_t> shape, xegpu::DistributeLayoutAttr attr);

      /// drops/slices the shape in the specified dims, and return the rest. e.g.,
      /// for shape = [32, 64, 8], dims = [0, 2], it will return [64]
      template<typename T, typename U>
      static llvm::SmallVector<T> slice(llvm::ArrayRef<T> shape, llvm::ArrayRef<U> dims) {
        llvm::SmallVector<T> result;
        for (auto [i, v]: llvm::enumerate(shape)) {
          if (!llvm::is_contained(dims, i))
            result.push_back(v);
        }
        return result;
      }
    };

} // namespace mlir::xegpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::XeGPUDialect)
