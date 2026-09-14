/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonGPUDialect.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::TritonGPUDialect)
namespace mlir::triton::gpu {


TritonGPUDialect::TritonGPUDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TritonGPUDialect>())
    
     {
  getContext()->loadDialect<triton::TritonDialect>();
  getContext()->loadDialect<mlir::gpu::GPUDialect>();
  initialize();
}

TritonGPUDialect::~TritonGPUDialect() = default;


} // namespace mlir::triton::gpu
