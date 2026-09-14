/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ProtonOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::proton::ProtonDialect)
namespace mlir::triton::proton {


ProtonDialect::ProtonDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<ProtonDialect>())
    
     {
  
  initialize();
}

ProtonDialect::~ProtonDialect() = default;


} // namespace mlir::triton::proton
