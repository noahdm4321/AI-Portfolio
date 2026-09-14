/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first
SmallVector<unsigned> mlir::triton::gpu::DistributedEncodingTrait::getRepOrder() const {
      return getImpl()->getRepOrder(getImpl(), *this);
  }

/// Return total element size per thread.
unsigned mlir::triton::gpu::DistributedEncodingTrait::getTotalElemsPerThread(ArrayRef<int64_t> shape) const {
      return getImpl()->getTotalElemsPerThread(getImpl(), *this, shape);
  }

/// Return element size per thread in each dimension.
SmallVector<unsigned> mlir::triton::gpu::DistributedEncodingTrait::getElemsPerThread(ArrayRef<int64_t> shape) const {
      return getImpl()->getElemsPerThread(getImpl(), *this, shape);
  }

/// Convert to LinearLayout.
LinearLayout mlir::triton::gpu::DistributedEncodingTrait::toLinearLayout(ArrayRef<int64_t> shape) const {
      return getImpl()->toLinearLayout(getImpl(), *this, shape);
  }

/// Get the CGA layout backing this encoding.
CGAEncodingAttr mlir::triton::gpu::LayoutEncodingTrait::getCGALayout() const {
      return getImpl()->getCGALayout(getImpl(), *this);
  }

/// Get the rank of the layout.
unsigned mlir::triton::gpu::LayoutEncodingTrait::getRank() const {
      return getImpl()->getRank(getImpl(), *this);
  }

/// Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first
SmallVector<unsigned> mlir::triton::gpu::MmaEncodingTrait::getRepOrderForOperand(int opIdx) const {
      return getImpl()->getRepOrderForOperand(getImpl(), *this, opIdx);
  }

/// Return the default alignment for the layout.
int32_t mlir::triton::gpu::SharedEncodingTrait::getAlignment() const {
      return getImpl()->getAlignment(getImpl(), *this);
  }
