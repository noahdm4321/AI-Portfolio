/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MPI.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::mpi {

/// This dialect models the Message Passing Interface (MPI), version 
///     4.0. It is meant to serve as an interfacing dialect that is targeted
///     by higher-level dialects. The MPI dialect itself can be lowered to 
///     multiple MPI implementations and hide differences in ABI. The dialect
///     models the functions of the MPI specification as close to 1:1 as possible
///     while preserving SSA value semantics where it makes sense, and uses 
///     `memref` types instead of bare pointers.
/// 
///     This dialect is under active development, and while stability is an
///     eventual goal, it is not guaranteed at this juncture. Given the early 
///     state, it is recommended to inquire further prior to using this dialect.
/// 
///     For an in-depth documentation of the MPI library interface, please refer 
///     to official documentation such as the 
///     [OpenMPI online documentation](https://www.open-mpi.org/doc/current/).
class MPIDialect : public ::mlir::Dialect {
  explicit MPIDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~MPIDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("mpi");
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
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::MPIDialect)
