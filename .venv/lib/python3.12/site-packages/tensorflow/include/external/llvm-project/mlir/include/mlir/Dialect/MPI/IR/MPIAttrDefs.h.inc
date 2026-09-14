/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::mpi {

/// MPI error class name
class MPI_ErrorClassEnumAttr;
namespace detail {

struct MPI_ErrorClassEnumAttrStorage;

} // namespace detail
class MPI_ErrorClassEnumAttr : public ::mlir::Attribute::AttrBase<MPI_ErrorClassEnumAttr, ::mlir::Attribute, detail::MPI_ErrorClassEnumAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "mpi.errclass";
  static constexpr ::llvm::StringLiteral dialectName = "mpi";
  static MPI_ErrorClassEnumAttr get(::mlir::MLIRContext *context, ::mlir::mpi::MPI_ErrorClassEnum value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"errclass"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::mpi::MPI_ErrorClassEnum getValue() const;
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::MPI_ErrorClassEnumAttr)

#endif // GET_ATTRDEF_CLASSES

