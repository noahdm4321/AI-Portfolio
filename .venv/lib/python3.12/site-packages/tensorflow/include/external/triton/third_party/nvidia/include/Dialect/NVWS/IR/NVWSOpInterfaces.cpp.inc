/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Return aref stage
::mlir::Value mlir::triton::nvws::ArefStageInterface::getStage() {
      return getImpl()->getStage(getImpl(), getOperation());
  }

/// Set aref stage
void mlir::triton::nvws::ArefStageInterface::setStage(::mlir::Value stage) {
      return getImpl()->setStage(getImpl(), getOperation(), stage);
  }

/// Get the transaction counts
int mlir::triton::nvws::DescriptorLoadOpInterface::getTxCount() {
      return getImpl()->getTxCount(getImpl(), getOperation());
  }

/// Get the descriptor
::mlir::TypedValue<mlir::triton::TensorDescType> mlir::triton::nvws::DescriptorLoadOpInterface::getDesc() {
      return getImpl()->implDescriptorOpInterface->getDesc(getImpl()->implDescriptorOpInterface, getOperation());
  }
