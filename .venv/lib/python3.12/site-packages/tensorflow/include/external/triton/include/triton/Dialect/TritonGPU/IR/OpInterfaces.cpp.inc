/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Infer destination encoding
mlir::Attribute mlir::triton::gpu::UpcastFpOpInterface::inferDstEncoding(unsigned opIdx, mlir::Attribute srcEnc) {
      return getImpl()->inferDstEncoding(getImpl(), getOperation(), opIdx, srcEnc);
  }

/// Infer operand encoding from dst encoding
mlir::Attribute mlir::triton::gpu::UpcastFpOpInterface::inferSrcEncoding(unsigned opIdx, mlir::Attribute dstEnc) {
      return getImpl()->inferSrcEncoding(getImpl(), getOperation(), opIdx, dstEnc);
  }
