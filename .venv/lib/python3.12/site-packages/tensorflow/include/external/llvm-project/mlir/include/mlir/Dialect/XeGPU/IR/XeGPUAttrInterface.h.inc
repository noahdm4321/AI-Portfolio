/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::xegpu {

/// Common trait for all XeGPU layouts.
class DistributeLayoutAttr;

} // namespace mlir::xegpu
namespace mlir::xegpu {

/// Common trait for all XeGPU layouts.
namespace detail {
struct DistributeLayoutAttrInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*isForWorkgroup)(const Concept *impl, ::mlir::Attribute );
    bool (*isForSubgroup)(const Concept *impl, ::mlir::Attribute );
    int64_t (*getRank)(const Concept *impl, ::mlir::Attribute );
    int64_t (*getNumSubgroups)(const Concept *impl, ::mlir::Attribute );
    DenseI32ArrayAttr (*getOrder)(const Concept *impl, ::mlir::Attribute );
    SmallVector<int64_t> (*getEffectiveSgLayoutAsInt)(const Concept *impl, ::mlir::Attribute );
    SmallVector<int64_t> (*getEffectiveSgDataAsInt)(const Concept *impl, ::mlir::Attribute );
    SmallVector<int64_t> (*getEffectiveInstDataAsInt)(const Concept *impl, ::mlir::Attribute );
    SmallVector<int64_t> (*getEffectiveLaneLayoutAsInt)(const Concept *impl, ::mlir::Attribute );
    SmallVector<int64_t> (*getEffectiveLaneDataAsInt)(const Concept *impl, ::mlir::Attribute );
    xegpu::DistributeLayoutAttr (*dropSgLayoutAndData)(const Concept *impl, ::mlir::Attribute );
    xegpu::DistributeLayoutAttr (*dropInstData)(const Concept *impl, ::mlir::Attribute );
    xegpu::DistributeLayoutAttr (*setUnitDimData)(const Concept *impl, ::mlir::Attribute , const llvm::SetVector<int64_t>);
    xegpu::DistributeLayoutAttr (*setUnitDimLayout)(const Concept *impl, ::mlir::Attribute , const llvm::SetVector<int64_t>);
    FailureOr<SmallVector<Value>> (*delinearizeId)(const Concept *impl, ::mlir::Attribute , OpBuilder &, Location, Value);
    FailureOr<SmallVector<SmallVector<Value>>> (*computeDistributedCoords)(const Concept *impl, ::mlir::Attribute , OpBuilder &, Location, Value, ArrayRef<int64_t>);
    bool (*isTransposeOf)(const Concept *impl, ::mlir::Attribute , const xegpu::DistributeLayoutAttr&, ArrayRef<int64_t>);
    bool (*isSliceOf)(const Concept *impl, ::mlir::Attribute , const xegpu::DistributeLayoutAttr&);
    bool (*isEqualTo)(const Concept *impl, ::mlir::Attribute , const xegpu::DistributeLayoutAttr&);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::xegpu::DistributeLayoutAttr;
    Model() : Concept{isForWorkgroup, isForSubgroup, getRank, getNumSubgroups, getOrder, getEffectiveSgLayoutAsInt, getEffectiveSgDataAsInt, getEffectiveInstDataAsInt, getEffectiveLaneLayoutAsInt, getEffectiveLaneDataAsInt, dropSgLayoutAndData, dropInstData, setUnitDimData, setUnitDimLayout, delinearizeId, computeDistributedCoords, isTransposeOf, isSliceOf, isEqualTo} {}

    static inline bool isForWorkgroup(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline bool isForSubgroup(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline int64_t getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline int64_t getNumSubgroups(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline DenseI32ArrayAttr getOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveSgLayoutAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveSgDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveInstDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveLaneLayoutAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveLaneDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline xegpu::DistributeLayoutAttr dropSgLayoutAndData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline xegpu::DistributeLayoutAttr dropInstData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline xegpu::DistributeLayoutAttr setUnitDimData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const llvm::SetVector<int64_t> unitDims);
    static inline xegpu::DistributeLayoutAttr setUnitDimLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const llvm::SetVector<int64_t> unitDims);
    static inline FailureOr<SmallVector<Value>> delinearizeId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, OpBuilder & builder, Location loc, Value linearId);
    static inline FailureOr<SmallVector<SmallVector<Value>>> computeDistributedCoords(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, OpBuilder & builder, Location loc, Value linearId, ArrayRef<int64_t> shape);
    static inline bool isTransposeOf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other, ArrayRef<int64_t> perm);
    static inline bool isSliceOf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other);
    static inline bool isEqualTo(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::xegpu::DistributeLayoutAttr;
    FallbackModel() : Concept{isForWorkgroup, isForSubgroup, getRank, getNumSubgroups, getOrder, getEffectiveSgLayoutAsInt, getEffectiveSgDataAsInt, getEffectiveInstDataAsInt, getEffectiveLaneLayoutAsInt, getEffectiveLaneDataAsInt, dropSgLayoutAndData, dropInstData, setUnitDimData, setUnitDimLayout, delinearizeId, computeDistributedCoords, isTransposeOf, isSliceOf, isEqualTo} {}

    static inline bool isForWorkgroup(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline bool isForSubgroup(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline int64_t getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline int64_t getNumSubgroups(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline DenseI32ArrayAttr getOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveSgLayoutAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveSgDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveInstDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveLaneLayoutAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline SmallVector<int64_t> getEffectiveLaneDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline xegpu::DistributeLayoutAttr dropSgLayoutAndData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline xegpu::DistributeLayoutAttr dropInstData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline xegpu::DistributeLayoutAttr setUnitDimData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const llvm::SetVector<int64_t> unitDims);
    static inline xegpu::DistributeLayoutAttr setUnitDimLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const llvm::SetVector<int64_t> unitDims);
    static inline FailureOr<SmallVector<Value>> delinearizeId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, OpBuilder & builder, Location loc, Value linearId);
    static inline FailureOr<SmallVector<SmallVector<Value>>> computeDistributedCoords(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, OpBuilder & builder, Location loc, Value linearId, ArrayRef<int64_t> shape);
    static inline bool isTransposeOf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other, ArrayRef<int64_t> perm);
    static inline bool isSliceOf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other);
    static inline bool isEqualTo(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct DistributeLayoutAttrTrait;

} // namespace detail
class DistributeLayoutAttr : public ::mlir::AttributeInterface<DistributeLayoutAttr, detail::DistributeLayoutAttrInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DistributeLayoutAttr, detail::DistributeLayoutAttrInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DistributeLayoutAttrTrait<ConcreteAttr> {};

  /// Check the availability of workgroup level layouts
  bool isForWorkgroup() const;

  /// Check the availability of subgroup level layouts
  bool isForSubgroup() const;

  /// Get the rank of attribute
  int64_t getRank() const;

  /// Get the num of effective subgroups
  int64_t getNumSubgroups() const;

  /// Get the order of the layout attribute
  DenseI32ArrayAttr getOrder() const;

  /// Get the effective SgLayout of the layout attribute as integer array
  SmallVector<int64_t> getEffectiveSgLayoutAsInt() const;

  /// Get the effective SgData of the layout attribute as integer array
  SmallVector<int64_t> getEffectiveSgDataAsInt() const;

  /// Get the effective InstData of the layout attribute as integer array
  SmallVector<int64_t> getEffectiveInstDataAsInt() const;

  /// Get the effective LaneLayout of the layout attribute as integer array
  SmallVector<int64_t> getEffectiveLaneLayoutAsInt() const;

  /// Get the effective LaneData of the layout attribute as integer array
  SmallVector<int64_t> getEffectiveLaneDataAsInt() const;

  /// Derive a new layout by dropping sgLayout and sgData
  xegpu::DistributeLayoutAttr dropSgLayoutAndData() const;

  /// Derive a new layout by dropping InstData
  xegpu::DistributeLayoutAttr dropInstData() const;

  /// Derive a new layout with sg_data, inst_data and lane_data set to 1 for the specified unit dims
  xegpu::DistributeLayoutAttr setUnitDimData(const llvm::SetVector<int64_t> unitDims) const;

  /// Derive a new layout with sg_lane and lane_layout set to 1 for the specified unit dims
  xegpu::DistributeLayoutAttr setUnitDimLayout(const llvm::SetVector<int64_t> unitDims) const;

  /// Delinearizes a linear ID into its multidimensional
  ///                       indices based on the effective layout level.
  FailureOr<SmallVector<Value>> delinearizeId(OpBuilder & builder, Location loc, Value linearId) const;

  /// Generates instructions to compute multidimensional coordinates for dist units
  ///                       assigned to a level identified by linearId. The shape parameter
  ///                       represents the higher-level problem size. Each level may access
  ///                       multiple blocks according to round-robin distribution rules.
  FailureOr<SmallVector<SmallVector<Value>>> computeDistributedCoords(OpBuilder & builder, Location loc, Value linearId, ArrayRef<int64_t> shape) const;

  /// Check if this layout can be achieved by applying a transpose
  ///                      to some other layout according to given permutation of (0...n-1).
  bool isTransposeOf(const xegpu::DistributeLayoutAttr& other, ArrayRef<int64_t> perm) const;

  /// Check if this layout is a slice of another layout.
  bool isSliceOf(const xegpu::DistributeLayoutAttr& other) const;

  /// Check if this layout is identical to another layout.
  bool isEqualTo(const xegpu::DistributeLayoutAttr& other) const;
};

} // namespace mlir::xegpu
namespace mlir::xegpu::detail {

  template <typename ConcreteAttr>
  struct DistributeLayoutAttrTrait : public ::mlir::AttributeInterface<DistributeLayoutAttr, detail::DistributeLayoutAttrInterfaceTraits>::Trait<ConcreteAttr> {
  };

} // namespace mlir::xegpu::detail
namespace mlir::xegpu {

template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::isForWorkgroup(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).isForWorkgroup();
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::isForSubgroup(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).isForSubgroup();
}
template<typename ConcreteAttr>
int64_t detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRank();
}
template<typename ConcreteAttr>
int64_t detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::getNumSubgroups(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  std::optional<SmallVector<int64_t>> sgLayout = llvm::cast<ConcreteAttr>(tablegen_opaque_val).getEffectiveSgLayoutAsInt();
                        if (sgLayout.has_value())
                          return computeProduct(*sgLayout);
                        return 0;
}
template<typename ConcreteAttr>
DenseI32ArrayAttr detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::getOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder();
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::getEffectiveSgLayoutAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveSgLayoutAsInt();
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::getEffectiveSgDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveSgDataAsInt();
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::getEffectiveInstDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveInstDataAsInt();
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::getEffectiveLaneLayoutAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveLaneLayoutAsInt();
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::getEffectiveLaneDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveLaneDataAsInt();
}
template<typename ConcreteAttr>
xegpu::DistributeLayoutAttr detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::dropSgLayoutAndData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).dropSgLayoutAndData();
}
template<typename ConcreteAttr>
xegpu::DistributeLayoutAttr detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::dropInstData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).dropInstData();
}
template<typename ConcreteAttr>
xegpu::DistributeLayoutAttr detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::setUnitDimData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const llvm::SetVector<int64_t> unitDims) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).setUnitDimData(unitDims);
}
template<typename ConcreteAttr>
xegpu::DistributeLayoutAttr detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::setUnitDimLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const llvm::SetVector<int64_t> unitDims) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).setUnitDimLayout(unitDims);
}
template<typename ConcreteAttr>
FailureOr<SmallVector<Value>> detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::delinearizeId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, OpBuilder & builder, Location loc, Value linearId) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).delinearizeId(builder, loc, linearId);
}
template<typename ConcreteAttr>
FailureOr<SmallVector<SmallVector<Value>>> detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::computeDistributedCoords(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, OpBuilder & builder, Location loc, Value linearId, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).computeDistributedCoords(builder, loc, linearId, shape);
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::isTransposeOf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other, ArrayRef<int64_t> perm) {
  if (!other)
                        return false;
                      if ((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRank() != other.getRank() || perm.size() != static_cast<size_t>((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRank()))
                        return false;
                      // Check if the permutation is valid
                      if (!isPermutationVector(perm))
                        return false;
                      auto checkTranspose = [](ArrayRef<int64_t> dst, ArrayRef<int64_t> src, ArrayRef<int64_t> perm) {
                        // If both `dst` and `src` are empty, conservatively return true
                        // here because some layout fields can be empty.
                        if (dst.empty() && src.empty())
                          return true;
                        for (const auto &ta : llvm::enumerate(perm)) {
                          if (src[ta.index()] != dst[ta.value()])
                            return false;
                        }
                        return true;
                      };
                      // Check sgLayout
                      if (!checkTranspose((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveSgLayoutAsInt(), other.getEffectiveSgLayoutAsInt(), perm))
                        return false;
                      // Check sgData
                      if (!checkTranspose((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveSgDataAsInt(), other.getEffectiveSgDataAsInt(), perm))
                        return false;
                      // Check instData
                      if (!checkTranspose((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveInstDataAsInt(), other.getEffectiveInstDataAsInt(), perm))
                        return false;
                      // Check laneLayout
                      if (!checkTranspose((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveLaneLayoutAsInt(), other.getEffectiveLaneLayoutAsInt(), perm))
                        return false;
                      // Check laneData
                      if (!checkTranspose((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getEffectiveLaneDataAsInt(), other.getEffectiveLaneDataAsInt(), perm))
                        return false;
                      // Check order if both sides have order field.
                      if ((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder() && other.getOrder()) {
                        auto thisOrderAsInt = llvm::to_vector_of<int64_t>((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getOrder().asArrayRef());
                        auto otherOrderAsInt = llvm::to_vector_of<int64_t>(other.getOrder().asArrayRef());
                        if (!checkTranspose(thisOrderAsInt, otherOrderAsInt, perm))
                          return false;
                      }
                      return true;
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::isSliceOf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).isSliceOf(other);
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::Model<ConcreteAttr>::isEqualTo(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).isEqualTo(other);
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::isForWorkgroup(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->isForWorkgroup(tablegen_opaque_val);
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::isForSubgroup(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->isForSubgroup(tablegen_opaque_val);
}
template<typename ConcreteAttr>
int64_t detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRank(tablegen_opaque_val);
}
template<typename ConcreteAttr>
int64_t detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getNumSubgroups(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getNumSubgroups(tablegen_opaque_val);
}
template<typename ConcreteAttr>
DenseI32ArrayAttr detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getOrder(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getEffectiveSgLayoutAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEffectiveSgLayoutAsInt(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getEffectiveSgDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEffectiveSgDataAsInt(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getEffectiveInstDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEffectiveInstDataAsInt(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getEffectiveLaneLayoutAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEffectiveLaneLayoutAsInt(tablegen_opaque_val);
}
template<typename ConcreteAttr>
SmallVector<int64_t> detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::getEffectiveLaneDataAsInt(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getEffectiveLaneDataAsInt(tablegen_opaque_val);
}
template<typename ConcreteAttr>
xegpu::DistributeLayoutAttr detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::dropSgLayoutAndData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->dropSgLayoutAndData(tablegen_opaque_val);
}
template<typename ConcreteAttr>
xegpu::DistributeLayoutAttr detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::dropInstData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->dropInstData(tablegen_opaque_val);
}
template<typename ConcreteAttr>
xegpu::DistributeLayoutAttr detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::setUnitDimData(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const llvm::SetVector<int64_t> unitDims) {
  return static_cast<const ConcreteAttr *>(impl)->setUnitDimData(tablegen_opaque_val, unitDims);
}
template<typename ConcreteAttr>
xegpu::DistributeLayoutAttr detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::setUnitDimLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const llvm::SetVector<int64_t> unitDims) {
  return static_cast<const ConcreteAttr *>(impl)->setUnitDimLayout(tablegen_opaque_val, unitDims);
}
template<typename ConcreteAttr>
FailureOr<SmallVector<Value>> detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::delinearizeId(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, OpBuilder & builder, Location loc, Value linearId) {
  return static_cast<const ConcreteAttr *>(impl)->delinearizeId(tablegen_opaque_val, builder, loc, linearId);
}
template<typename ConcreteAttr>
FailureOr<SmallVector<SmallVector<Value>>> detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::computeDistributedCoords(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, OpBuilder & builder, Location loc, Value linearId, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->computeDistributedCoords(tablegen_opaque_val, builder, loc, linearId, shape);
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::isTransposeOf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other, ArrayRef<int64_t> perm) {
  return static_cast<const ConcreteAttr *>(impl)->isTransposeOf(tablegen_opaque_val, other, perm);
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::isSliceOf(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other) {
  return static_cast<const ConcreteAttr *>(impl)->isSliceOf(tablegen_opaque_val, other);
}
template<typename ConcreteAttr>
bool detail::DistributeLayoutAttrInterfaceTraits::FallbackModel<ConcreteAttr>::isEqualTo(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, const xegpu::DistributeLayoutAttr& other) {
  return static_cast<const ConcreteAttr *>(impl)->isEqualTo(tablegen_opaque_val, other);
}

} // namespace mlir::xegpu
