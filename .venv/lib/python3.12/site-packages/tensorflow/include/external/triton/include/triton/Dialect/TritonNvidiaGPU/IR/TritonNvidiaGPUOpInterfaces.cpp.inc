/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Return the A operand.
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> mlir::triton::nvidia_gpu::MMAv5OpInterface::getA() {
      return getImpl()->getA(getImpl(), getOperation());
  }

/// Return the B operand.
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> mlir::triton::nvidia_gpu::MMAv5OpInterface::getB() {
      return getImpl()->getB(getImpl(), getOperation());
  }

/// Return the accumulator init flag.
::mlir::Value mlir::triton::nvidia_gpu::MMAv5OpInterface::useAccumulator() {
      return getImpl()->useAccumulator(getImpl(), getOperation());
  }

/// Set the accumulator init flag.
void mlir::triton::nvidia_gpu::MMAv5OpInterface::setUseAccumulator(::mlir::Value flag) {
      return getImpl()->setUseAccumulator(getImpl(), getOperation(), flag);
  }

/// Return the completion barriers of this MMAv5 op.
::mlir::ValueRange mlir::triton::nvidia_gpu::MMAv5OpInterface::getCompletionBarriers() {
      return getImpl()->getCompletionBarriers(getImpl(), getOperation());
  }

/// Return the completion barrier predicates of this MMAv5 op.
::mlir::ValueRange mlir::triton::nvidia_gpu::MMAv5OpInterface::getCompletionBarrierPreds() {
      return getImpl()->getCompletionBarrierPreds(getImpl(), getOperation());
  }

/// Associate a new completion barrier to this MMAv5 op.
void mlir::triton::nvidia_gpu::MMAv5OpInterface::addCompletionBarrier(::mlir::Value barrier, ::mlir::Value pred) {
      return getImpl()->addCompletionBarrier(getImpl(), getOperation(), barrier, pred);
  }

/// Return the accumulator.
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> mlir::triton::nvidia_gpu::MMAv5OpInterface::getAccumulator() {
      return getImpl()->getAccumulator(getImpl(), getOperation());
  }

/// Set the accumulator.
void mlir::triton::nvidia_gpu::MMAv5OpInterface::setAccumulator(::mlir::Value accum) {
      return getImpl()->setAccumulator(getImpl(), getOperation(), accum);
  }

/// Return the predicate of this op.
::mlir::Value mlir::triton::nvidia_gpu::MMAv5OpInterface::getPredicate() {
      return getImpl()->getPredicate(getImpl(), getOperation());
  }

/// Set the predicate of this op.
void mlir::triton::nvidia_gpu::MMAv5OpInterface::setPredicate(::mlir::Value pred) {
      return getImpl()->setPredicate(getImpl(), getOperation(), pred);
  }

/// Get the memory dependencies of the accumulator.
::mlir::Value mlir::triton::nvidia_gpu::MMAv5OpInterface::getAccDep() {
      return getImpl()->getAccDep(getImpl(), getOperation());
  }

/// Get the mutable memory dependencies of the accumulator.
::mlir::MutableOperandRange mlir::triton::nvidia_gpu::MMAv5OpInterface::getAccDepMutable() {
      return getImpl()->getAccDepMutable(getImpl(), getOperation());
  }

/// Get the produced write dependency of the accumulator.
::mlir::Value mlir::triton::nvidia_gpu::MMAv5OpInterface::getToken() {
      return getImpl()->getToken(getImpl(), getOperation());
  }

/// Indicate that this MMA op executes asynchronously.
void mlir::triton::nvidia_gpu::MMAv5OpInterface::setIsAsync(bool isAsync) {
      return getImpl()->setIsAsync(getImpl(), getOperation(), isAsync);
  }

/// Return true if this MMA op executes asynchronously.
bool mlir::triton::nvidia_gpu::MMAv5OpInterface::isAsync() {
      return getImpl()->isAsync(getImpl(), getOperation());
  }
