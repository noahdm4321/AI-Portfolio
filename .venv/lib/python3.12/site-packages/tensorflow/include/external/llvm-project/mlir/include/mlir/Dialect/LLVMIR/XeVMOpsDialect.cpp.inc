/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: XeVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xevm::XeVMDialect)
namespace mlir::xevm {


XeVMDialect::XeVMDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<XeVMDialect>())
    
     {
  getContext()->loadDialect<LLVM::LLVMDialect>();
  initialize();
}

XeVMDialect::~XeVMDialect() = default;


} // namespace mlir::xevm
