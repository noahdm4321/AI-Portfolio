/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Check the availability of workgroup level layouts
bool mlir::xegpu::DistributeLayoutAttr::isForWorkgroup() const {
      return getImpl()->isForWorkgroup(getImpl(), *this);
  }

/// Check the availability of subgroup level layouts
bool mlir::xegpu::DistributeLayoutAttr::isForSubgroup() const {
      return getImpl()->isForSubgroup(getImpl(), *this);
  }

/// Get the rank of attribute
int64_t mlir::xegpu::DistributeLayoutAttr::getRank() const {
      return getImpl()->getRank(getImpl(), *this);
  }

/// Get the num of effective subgroups
int64_t mlir::xegpu::DistributeLayoutAttr::getNumSubgroups() const {
      return getImpl()->getNumSubgroups(getImpl(), *this);
  }

/// Get the order of the layout attribute
DenseI32ArrayAttr mlir::xegpu::DistributeLayoutAttr::getOrder() const {
      return getImpl()->getOrder(getImpl(), *this);
  }

/// Get the effective SgLayout of the layout attribute as integer array
SmallVector<int64_t> mlir::xegpu::DistributeLayoutAttr::getEffectiveSgLayoutAsInt() const {
      return getImpl()->getEffectiveSgLayoutAsInt(getImpl(), *this);
  }

/// Get the effective SgData of the layout attribute as integer array
SmallVector<int64_t> mlir::xegpu::DistributeLayoutAttr::getEffectiveSgDataAsInt() const {
      return getImpl()->getEffectiveSgDataAsInt(getImpl(), *this);
  }

/// Get the effective InstData of the layout attribute as integer array
SmallVector<int64_t> mlir::xegpu::DistributeLayoutAttr::getEffectiveInstDataAsInt() const {
      return getImpl()->getEffectiveInstDataAsInt(getImpl(), *this);
  }

/// Get the effective LaneLayout of the layout attribute as integer array
SmallVector<int64_t> mlir::xegpu::DistributeLayoutAttr::getEffectiveLaneLayoutAsInt() const {
      return getImpl()->getEffectiveLaneLayoutAsInt(getImpl(), *this);
  }

/// Get the effective LaneData of the layout attribute as integer array
SmallVector<int64_t> mlir::xegpu::DistributeLayoutAttr::getEffectiveLaneDataAsInt() const {
      return getImpl()->getEffectiveLaneDataAsInt(getImpl(), *this);
  }

/// Derive a new layout by dropping sgLayout and sgData
xegpu::DistributeLayoutAttr mlir::xegpu::DistributeLayoutAttr::dropSgLayoutAndData() const {
      return getImpl()->dropSgLayoutAndData(getImpl(), *this);
  }

/// Derive a new layout by dropping InstData
xegpu::DistributeLayoutAttr mlir::xegpu::DistributeLayoutAttr::dropInstData() const {
      return getImpl()->dropInstData(getImpl(), *this);
  }

/// Derive a new layout with sg_data, inst_data and lane_data set to 1 for the specified unit dims
xegpu::DistributeLayoutAttr mlir::xegpu::DistributeLayoutAttr::setUnitDimData(const llvm::SetVector<int64_t> unitDims) const {
      return getImpl()->setUnitDimData(getImpl(), *this, unitDims);
  }

/// Derive a new layout with sg_lane and lane_layout set to 1 for the specified unit dims
xegpu::DistributeLayoutAttr mlir::xegpu::DistributeLayoutAttr::setUnitDimLayout(const llvm::SetVector<int64_t> unitDims) const {
      return getImpl()->setUnitDimLayout(getImpl(), *this, unitDims);
  }

/// Delinearizes a linear ID into its multidimensional
///                       indices based on the effective layout level.
FailureOr<SmallVector<Value>> mlir::xegpu::DistributeLayoutAttr::delinearizeId(OpBuilder & builder, Location loc, Value linearId) const {
      return getImpl()->delinearizeId(getImpl(), *this, builder, loc, linearId);
  }

/// Generates instructions to compute multidimensional coordinates for dist units
///                       assigned to a level identified by linearId. The shape parameter
///                       represents the higher-level problem size. Each level may access
///                       multiple blocks according to round-robin distribution rules.
FailureOr<SmallVector<SmallVector<Value>>> mlir::xegpu::DistributeLayoutAttr::computeDistributedCoords(OpBuilder & builder, Location loc, Value linearId, ArrayRef<int64_t> shape) const {
      return getImpl()->computeDistributedCoords(getImpl(), *this, builder, loc, linearId, shape);
  }

/// Check if this layout can be achieved by applying a transpose
///                      to some other layout according to given permutation of (0...n-1).
bool mlir::xegpu::DistributeLayoutAttr::isTransposeOf(const xegpu::DistributeLayoutAttr& other, ArrayRef<int64_t> perm) const {
      return getImpl()->isTransposeOf(getImpl(), *this, other, perm);
  }

/// Check if this layout is a slice of another layout.
bool mlir::xegpu::DistributeLayoutAttr::isSliceOf(const xegpu::DistributeLayoutAttr& other) const {
      return getImpl()->isSliceOf(getImpl(), *this, other);
  }

/// Check if this layout is identical to another layout.
bool mlir::xegpu::DistributeLayoutAttr::isEqualTo(const xegpu::DistributeLayoutAttr& other) const {
      return getImpl()->isEqualTo(getImpl(), *this, other);
  }
