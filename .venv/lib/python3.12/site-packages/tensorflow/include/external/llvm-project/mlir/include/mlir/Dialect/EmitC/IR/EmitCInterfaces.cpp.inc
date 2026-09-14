/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Check whether operation has side effects that may affect the expression
/// evaluation.
/// 
/// By default operation is marked as having side effects.
/// 
/// ```c++
/// class ConcreteOp ... {
/// public:
///   bool hasSideEffects() {
///     // That way we can override the default implementation.
///     return false;
///   }
/// };
/// ```
bool mlir::emitc::CExpressionInterface::hasSideEffects() {
      return getImpl()->hasSideEffects(getImpl(), getOperation());
  }
