/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVGPUDialect.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::NVGPUDialect)
namespace mlir::triton::nvgpu {


NVGPUDialect::NVGPUDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<NVGPUDialect>())
    
     {
  getContext()->loadDialect<mlir::LLVM::LLVMDialect>();
  initialize();
}

NVGPUDialect::~NVGPUDialect() = default;


} // namespace mlir::triton::nvgpu
