/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Get operation base ptr.
::mlir::TypedValue<::mlir::triton::PointerType> mlir::triton::amdgpu::BufferOpInterface::getPtr() {
      return getImpl()->getPtr(getImpl(), getOperation());
  }

/// Get mutable operation base ptr.
::mlir::OpOperand &mlir::triton::amdgpu::BufferOpInterface::getPtrMutable() {
      return getImpl()->getPtrMutable(getImpl(), getOperation());
  }

/// Get operation offset tensor.
::mlir::TypedValue<::mlir::TensorType> mlir::triton::amdgpu::BufferOpInterface::getOffsets() {
      return getImpl()->getOffsets(getImpl(), getOperation());
  }

/// Get mutable operation offset tensor.
::mlir::OpOperand &mlir::triton::amdgpu::BufferOpInterface::getOffsetsMutable() {
      return getImpl()->getOffsetsMutable(getImpl(), getOperation());
  }

/// Get operation stride.
::mlir::TypedValue<::mlir::IntegerType> mlir::triton::amdgpu::BufferOpInterface::getStride() {
      return getImpl()->getStride(getImpl(), getOperation());
  }

/// Get mutable operation stride.
::mlir::MutableOperandRange mlir::triton::amdgpu::BufferOpInterface::getStrideMutable() {
      return getImpl()->getStrideMutable(getImpl(), getOperation());
  }
