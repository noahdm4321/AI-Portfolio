/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OpenACCOps.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::OpenACCDialect)
namespace mlir::acc {


OpenACCDialect::OpenACCDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<OpenACCDialect>())
    
     {
  getContext()->loadDialect<::mlir::memref::MemRefDialect>();
  getContext()->loadDialect<::mlir::LLVM::LLVMDialect>();
  getContext()->loadDialect<::mlir::gpu::GPUDialect>();
  initialize();
}

OpenACCDialect::~OpenACCDialect() = default;


} // namespace mlir::acc
