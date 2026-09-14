/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Get the anchor layout attribute.
xegpu::DistributeLayoutAttr mlir::xegpu::AnchorLayoutInterface::getAnchorLayout() {
      return getImpl()->getAnchorLayout(getImpl(), getOperation());
  }

/// Set the anchor layout attribute.
void mlir::xegpu::AnchorLayoutInterface::setAnchorLayout(xegpu::DistributeLayoutAttr layout) {
      return getImpl()->setAnchorLayout(getImpl(), getOperation(), layout);
  }
