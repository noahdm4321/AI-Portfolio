/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: GluonDialect.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gluon::GluonDialect)
namespace mlir::triton::gluon {


GluonDialect::GluonDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<GluonDialect>())
    
     {
  getContext()->loadDialect<triton::TritonDialect>();
  getContext()->loadDialect<triton::gpu::TritonGPUDialect>();
  getContext()->loadDialect<mlir::gpu::GPUDialect>();
  initialize();
}

GluonDialect::~GluonDialect() = default;


} // namespace mlir::triton::gluon
