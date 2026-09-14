/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Availability Interface Definitions                                         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TosaOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> mlir::tosa::QueryProfileInterface::getProfiles() {
  return getImpl()->getProfiles(getImpl(), getOperation());
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> mlir::tosa::QueryExtensionInterface::getExtensions() {
  return getImpl()->getExtensions(getImpl(), getOperation());
}
