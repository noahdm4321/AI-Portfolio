/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonInstrumentDialect.td                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::TritonInstrumentDialect)
namespace mlir::triton::instrument {


TritonInstrumentDialect::TritonInstrumentDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TritonInstrumentDialect>())
    
     {
  
  initialize();
}

TritonInstrumentDialect::~TritonInstrumentDialect() = default;


} // namespace mlir::triton::instrument
