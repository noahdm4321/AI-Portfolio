/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVWSDialect.td                                                       *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::NVWSDialect)
namespace mlir::triton::nvws {


NVWSDialect::NVWSDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<NVWSDialect>())
    
     {
  getContext()->loadDialect<triton::TritonDialect>();
  getContext()->loadDialect<triton::gpu::TritonGPUDialect>();
  initialize();
}

NVWSDialect::~NVWSDialect() = default;


} // namespace mlir::triton::nvws
