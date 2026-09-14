/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::mpi {

/// MPI function call return value (!mpi.retval)
/// This type represents a return value from an MPI function call.
///     This value can be MPI_SUCCESS, MPI_ERR_IN_STATUS, or any error code.
/// 
///     This return value can be compared agains the known MPI error classes
///     represented by `#mpi.errclass` using the `mpi.retval_check` operation.
class RetvalType;
/// MPI communicator handler
/// This type represents a handler for the MPI communicator.
class CommType;
/// MPI asynchronous request handler
/// This type represents a handler to an asynchronous request.
class RequestType;
/// MPI reception operation status type
/// This type represents the status of a reception operation.
class StatusType;
class RetvalType : public ::mlir::Type::TypeBase<RetvalType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "mpi.retval";
  static constexpr ::llvm::StringLiteral dialectName = "mpi";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"retval"};
  }

};
class CommType : public ::mlir::Type::TypeBase<CommType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "mpi.comm";
  static constexpr ::llvm::StringLiteral dialectName = "mpi";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"comm"};
  }

};
class RequestType : public ::mlir::Type::TypeBase<RequestType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "mpi.request";
  static constexpr ::llvm::StringLiteral dialectName = "mpi";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"request"};
  }

};
class StatusType : public ::mlir::Type::TypeBase<StatusType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "mpi.status";
  static constexpr ::llvm::StringLiteral dialectName = "mpi";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"status"};
  }

};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::RetvalType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::CommType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::RequestType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::StatusType)

#endif // GET_TYPEDEF_CLASSES

