/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Get the LHS A tensor
::mlir::Value mlir::triton::DotOpInterface::getA() {
      return getImpl()->getA(getImpl(), getOperation());
  }

/// Get the RHS B tensor
::mlir::Value mlir::triton::DotOpInterface::getB() {
      return getImpl()->getB(getImpl(), getOperation());
  }

/// Get the output tensor
::mlir::Value mlir::triton::DotOpInterface::getD() {
      return getImpl()->getD(getImpl(), getOperation());
  }

/// Verify the dimensions of the A and B DotOp operands.
bool mlir::triton::DotOpInterface::verifyDims() {
      return getImpl()->verifyDims(getImpl(), getOperation());
  }

/// Verify the dimensions of the DotOp output.
bool mlir::triton::DotOpInterface::verifyOutputDims() {
      return getImpl()->verifyOutputDims(getImpl(), getOperation());
  }

/// Get the descriptor
::mlir::TypedValue<mlir::triton::TensorDescType> mlir::triton::DescriptorOpInterface::getDesc() {
      return getImpl()->getDesc(getImpl(), getOperation());
  }

/// Get Source tensor
::mlir::TypedValue<mlir::RankedTensorType> mlir::triton::DescriptorStoreLikeOpInterface::getSrc() {
      return getImpl()->getSrc(getImpl(), getOperation());
  }

/// Get mutable source tensor
::mlir::OpOperand&mlir::triton::DescriptorStoreLikeOpInterface::getSrcMutable() {
      return getImpl()->getSrcMutable(getImpl(), getOperation());
  }

/// Get the descriptor
::mlir::TypedValue<mlir::triton::TensorDescType> mlir::triton::DescriptorStoreLikeOpInterface::getDesc() {
      return getImpl()->implDescriptorOpInterface->getDesc(getImpl()->implDescriptorOpInterface, getOperation());
  }

/// Get the source operand of the transposition.
::mlir::Value mlir::triton::TransposeOpInterface::getSrc() {
      return getImpl()->getSrc(getImpl(), getOperation());
  }

/// Get the order of the transposition.
::mlir::ArrayRef<int32_t> mlir::triton::TransposeOpInterface::getOrder() {
      return getImpl()->getOrder(getImpl(), getOperation());
  }
