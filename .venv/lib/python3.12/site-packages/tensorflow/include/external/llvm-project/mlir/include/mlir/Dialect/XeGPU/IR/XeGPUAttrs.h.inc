/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::xegpu {

/// a composite attribute for `TensorDescType`
/// `BlockTensorDesc` (or `block_tdesc_attr`) is a composite
///     attribute defined for `TensorDescType` for describing following
///     properties of a `TensorDesc`.
///     1. `memory_space`: It describes where the data block described by the
///         TensorDesc is located, `Global` device memory or `Shared` local memory.
///         It is default to `Global`.
///     2. `array_length`: It describes how many horizontally consecutive blocks
///         will be loaded by a hardware load instruction. If the TensorDesc shape
///         is 8x16, with array_length = 2. The loaded block shape will be actually
///         8x32. Its default value is 1.
///     3. `boundary_check`: It is used to indicates the hardware whether to do
///         out-of-boundary check. The default value is true.
class BlockTensorDescAttr;
/// a composite attribute for `TensorDescType`
/// `ScatterTensorDesc` is a composite attribute defined for `TensorDescType`
///     for describing following properties of a `TensorDesc`:
/// 
///     1. `memory_space`: It describes where the data block described by the
///         TensorDesc is located, `Global` device memory or `Shared` local memory.
///         It is default to `Global`.
/// 
///     2. `chunk_size`: Specifies the number of contiguous elements accessed per offset.
///       The default value is 1.
class ScatterTensorDescAttr;
/// Describe the location of data described by a `TensorDesc`:
///                  Global device memory (`Global`) or Shared local memory (`SLM`).
class MemorySpaceAttr;
/// Describe the cache settings for prefetch/load/store operators
class CachePolicyAttr;
/// Describes the scope of fence.
///                     "workgroup" means that the scope is within each work group.
///                     "gpu" means the scope is across work groups within the gpu.
class FenceScopeAttr;
/// Describes the data distribution to subgroups and work-items for a tensor
///     specified by the tensor descriptor.
/// XeGPU operations use `LayoutAttr` to define how data is distributed across subgroups and work-items.
///     This attribute is specified in tensor descriptors during tensor description creation. `LayoutAttr`
///     includes the following parameters:
/// 
///     * `sg_layout`: Specifies the total number of subgroups and their layout within a workgroup.
///       It is mandatory for workgroup-level programming. Its presence implies workgroup-level code.
///     * `sg_data`: Defines the data size accessed per subgroup. It is optionally used with `sg_layout`
///       for workgroup-level programming. When it is left empty, the size accessed per subgroup can be
///       derived from the tensor shape and `sg_layout` using the formula:
///       `sg_data[i] = tensor_shape[i] / sg_layout[i]`.
///     * `inst_data`: Specifies the data size that is processed by an instruction. It is optionally
///       used with lane_layout. When it is left empty, the data size per instruction is equivalent to
///       the sg_data for workgroup-level programming or equivalent to tensor shape for subgroup-level
///       programming.
///     * `lane_layout` : Specifies the total number of work-items and their arrangement within a subgroup.
///       It is mandatory for subgroup-level programming and optional for workgroup-level programming.
///     * `lane_data` : Specifies the shape of the tensor fragment that each lane accesses. It defines a single,
///       minimal distribution unit. Processing the entire tensor may require one or more distribution units per
///       hardware instruction.
///     * `order`: Specifies the dimension order used to linearize n-dimensional sg_layout and lane_layout to
///       1-dimensional layout. The first dimension in the order list is the fastest-changing dimension. If it
///       is not present, the default value is [1, 0].
/// 
///     Examples:
/// 
///     1. Subgroup level layout:
///     ```mlir
///     #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1]>
///     ```
///     In this example, there are 16 work-items per subgroup, and is organized as
///     [[0, 1, 2, .., 7],[8, 9, .., 15]]. The distribution unit is 1x1.
/// 
///     2. Subgroup level layout with order:
///     ```mlir
///     #xegpu.layout<lane_layout = [2, 8], lane_data = [1, 1], order = [0, 1]>
///     ```
///     In this example, there are 16 work-items per subgroup, and is organized as
///     [[0, 2, 4, ..., 14], [1, 3, 5, ..., 15]]. The distribution unit is 1x1.
/// 
///     3. Subgroup level layout with inst_data
///     ```mlir
///     #xegpu.layout<inst_data = [8, 16], lane_layout = [2, 8], lane_data = [2, 2]>
///     ```
///     In this example, the original problem size is partitioned into smaller subproblems of dimensions [8, 16],
///     which are then distributed among 16 work-items arranged as [[0, 1, 2, ..., 7], [8, 9, ..., 15]]. Each
///     work-item is assigned four 2x2 blocks in a round-robin manner.
/// 
///     4. Workgroup level layout:
///     ```mlir
///     #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [2, 8], lane_data = [1, 1]>
///     ```
///     In this example, the layout represents a workgroup distribution. A workgroup consists of 8 subgroups
///     arranged as [[0, 1, 2, 3], [4, 5, 6, 7]]. Each subgroup accesses a 16x16 block per instruction, which
///     is further distributed to 16 work items which is organized as [[0, 1, 2, .., 7],[8, 9, .., 15]].
/// 
///     5. Workgroup level layout with order:
///     ```mlir
///     #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], lane_layout = [2, 8], lane_data = [1, 1], order = [0, 1]>
///     ```
///     In this example, the layout represents a workgroup distribution. A workgroup consists of 8 subgroups
///     arranged as [[0, 2, 4, 6], [1, 3, 5, 7]]. Each subgroup accesses a 16x16 block per instruction, which
///     is further distributed to 16 work items which is organized as [[0, 2, 4, ..., 14], [1, 3, 5, ..., 15]].
/// 
///     6. Workgroup level layout with inst_data:
///     ```mlir
///     #xegpu.layout<sg_layout = [2, 4], sg_data = [16, 16], inst_data = [8, 16], lane_layout = [2, 8], lane_data = [1, 1]>
///     ```
///     This example is similar to the previous ones, but the `inst_data` parameter divides `sg_data` into two instructions,
///     each processing an 8x16 block. These blocks are further distributed across 16 work-items with a distribution unit of 1x1.
///     Unlike the 2x2 distribution unit in example 3, which results in accessing contiguous 2x2 blocks, the 1x1 distribution
///     unit may result in non-contiguous access.
class LayoutAttr;
/// Describes the data distribution and sharing among subgroups or work-items.
/// Like LayoutAttr, SliceAttr describes data distribution among subgroups or work-items.
///     However, whereas LayoutAttr requires the data to have the same rank as the attribute,
///     SliceAttr permits the data to have a lower rank. In this case, compute units in the
///     specified dimensions (given by `$dims`) share the data, provided that the remaining
///     ranks match the data rank. SliceAttr is commonly used by operations such as
///     vector.multi_reduction and vector.broadcast.
/// 
///     Example:
///     ```
///     #l = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>
///     #r = #xegpu.slice<#l, dim = [0]>
/// 
///     %exp = math.exp %input {layout_result_0 = #l}: vector<256x128xf32>
///     %red = vector.multi_reduction<add>, %exp, %acc [0] {layout_result_0 = #r}: vector<256x128xf32> to vector<128xf32>
///     %bcast = vector.broadcast %red {layout_result_0 = #l} : vector<128xf32> to vector<256x128xf32>
///     ```
///     In this example, %red is conceptually divided into 4 vectors of type vector<32xf32>, each assigned to
///     a group of subgroups. Each group consists of 8 subgroups from the same column of sg_layout, sharing a
///     single reduction result of type vector<32xf32>.
class SliceAttr;
/// Specifies a half-open range
/// `RangeAttr` is an attribute that defines a half-open range [start, end).
///     The range is inclusive of the start value and exclusive of the end value.
///     One usage of this attribute can be to specify the subgroup id range.
///     The subgroup id range can be specified using this attribute,
///     and it can be attached to a scf.if op like
///     ```mlir
///     scf.if %cond {
///       // some operations
///     } {sg_id_range = #xegpu.range<[2, 4]>}
///     ```
///     In this case, the scf.if op will only be executed for subgroup IDs 2 and 3.
class RangeAttr;
/// Specifies memory layouts with named attributes.
/// This attribute stores a collection of named attributes that describe
///     memory layout properties such as stride, block, etc.
class MemLayoutAttr;
namespace detail {

struct BlockTensorDescAttrStorage;

} // namespace detail
class BlockTensorDescAttr : public ::mlir::Attribute::AttrBase<BlockTensorDescAttr, ::mlir::Attribute, detail::BlockTensorDescAttrStorage> {
public:
  using Base::Base;
  // return true if all fields of the BlockTensorDescAttr are set with
  // default values.
  bool hasDefaultsOnly();
  static constexpr ::llvm::StringLiteral name = "xegpu.block_tdesc_attr";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static BlockTensorDescAttr get(::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr array_length, BoolAttr boundary_check);
  static BlockTensorDescAttr get(::mlir::MLIRContext *context, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, int array_length = 1, bool boundary_check = true);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"block_tdesc_attr"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  MemorySpaceAttr getMemorySpace() const;
  IntegerAttr getArrayLength() const;
  BoolAttr getBoundaryCheck() const;
};
namespace detail {

struct ScatterTensorDescAttrStorage;

} // namespace detail
class ScatterTensorDescAttr : public ::mlir::Attribute::AttrBase<ScatterTensorDescAttr, ::mlir::Attribute, detail::ScatterTensorDescAttrStorage> {
public:
  using Base::Base;
  int64_t getChunkSizeAsInt() {
    return getChunkSize().getInt();
  }
  static constexpr ::llvm::StringLiteral name = "xegpu.scatter_tdesc_attr";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  using Base::getChecked;
  static ScatterTensorDescAttr get(::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr chunk_size);
  static ScatterTensorDescAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr chunk_size);
  static ScatterTensorDescAttr get(::mlir::MLIRContext *context, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, int chunk_size = 1);
  static ScatterTensorDescAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, xegpu::MemorySpace memory_space = xegpu::MemorySpace::Global, int chunk_size = 1);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, MemorySpaceAttr memory_space, IntegerAttr chunk_size);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, MemorySpaceAttr memory_space, IntegerAttr chunk_size);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"scatter_tdesc_attr"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  MemorySpaceAttr getMemorySpace() const;
  IntegerAttr getChunkSize() const;
};
namespace detail {

struct MemorySpaceAttrStorage;

} // namespace detail
class MemorySpaceAttr : public ::mlir::Attribute::AttrBase<MemorySpaceAttr, ::mlir::Attribute, detail::MemorySpaceAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.memory_space";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static MemorySpaceAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::MemorySpace value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"memory_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::MemorySpace getValue() const;
};
namespace detail {

struct CachePolicyAttrStorage;

} // namespace detail
class CachePolicyAttr : public ::mlir::Attribute::AttrBase<CachePolicyAttr, ::mlir::Attribute, detail::CachePolicyAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.cache_hint";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static CachePolicyAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::CachePolicy value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"cache_hint"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::CachePolicy getValue() const;
};
namespace detail {

struct FenceScopeAttrStorage;

} // namespace detail
class FenceScopeAttr : public ::mlir::Attribute::AttrBase<FenceScopeAttr, ::mlir::Attribute, detail::FenceScopeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.fence_scope";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static FenceScopeAttr get(::mlir::MLIRContext *context, ::mlir::xegpu::FenceScope value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"fence_scope"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xegpu::FenceScope getValue() const;
};
namespace detail {

struct LayoutAttrStorage;

} // namespace detail
class LayoutAttr : public ::mlir::Attribute::AttrBase<LayoutAttr, ::mlir::Attribute, detail::LayoutAttrStorage, ::mlir::xegpu::DistributeLayoutAttr::Trait> {
public:
  using Base::Base;
  bool isForWorkgroup() {
    return getSgLayout() != nullptr;
  }

  bool isForSubgroup() {
    return !isForWorkgroup();
  }

  int64_t getRank() {
    if (auto attr = getSgLayout())
      return attr.size();
    if (auto attr = getInstData())
      return attr.size();
    if (auto attr = getLaneLayout())
      return attr.size();
    return 0;
  }

  LayoutAttr dropSgLayoutAndData() const{
    // avoid every field of the attribute is nullptr, which may lead to segment fault
    if (!getInstData() && !getLaneLayout())
      return nullptr;
    return LayoutAttr::get(getContext(), nullptr, nullptr, getInstData(),
                           getLaneLayout(), getLaneData(), getOrder());
  }

  LayoutAttr dropInstData() const{
    // avoid every field of the attribute is nullptr, which may lead to segment fault
    if (!getSgLayout() && !getLaneLayout())
      return nullptr;
    return LayoutAttr::get(getContext(), getSgLayout(), getSgData(), nullptr,
                           getLaneLayout(), getLaneData(), getOrder());
  }

  SmallVector<int64_t> getEffectiveSgLayoutAsInt() const {
    if (DenseI32ArrayAttr layout = getSgLayout())
      return llvm::to_vector_of<int64_t>(layout.asArrayRef());
    return {};
  }

  SmallVector<int64_t> getEffectiveSgDataAsInt() const {
    if (DenseI32ArrayAttr data = getSgData())
      return llvm::to_vector_of<int64_t>(data.asArrayRef());
    return {};
  }

  SmallVector<int64_t> getEffectiveInstDataAsInt() const {
    if (DenseI32ArrayAttr inst = getInstData())
      return llvm::to_vector_of<int64_t>(inst.asArrayRef());
    return {};
  }

  SmallVector<int64_t> getEffectiveLaneLayoutAsInt() const {
    if (DenseI32ArrayAttr layout = getLaneLayout())
      return llvm::to_vector_of<int64_t>(layout.asArrayRef());
    return {};
  }

  SmallVector<int64_t> getEffectiveLaneDataAsInt() const {
    if (DenseI32ArrayAttr data = getLaneData())
      return llvm::to_vector_of<int64_t>(data.asArrayRef());
    return {};
  }

  //set the layout for the sepcified unit dims: sg_data, inst_data and lane_data to 1
  DistributeLayoutAttr setUnitDimData(SetVector<int64_t> unitDims) const;

  //set the layout for the sepcified unit dims: sg_lane and lane_layout to 1
  DistributeLayoutAttr setUnitDimLayout(SetVector<int64_t> unitDims) const;

  /// Delinearizes a linear ID into its multidimensional indices
  /// based on the effective level of the layout.
  FailureOr<SmallVector<Value>>
  delinearizeId(OpBuilder &builder, Location loc, Value linearId);

  /// Generates instructions to compute multidimensional coordinates for dist units
  /// assigned to a level identified by linearId. The shape parameter
  /// represents the higher-level problem size. Each `level` may access
  /// multiple blocks according to round-robin distribution rules.
  FailureOr<SmallVector<SmallVector<Value>>>
  computeDistributedCoords(OpBuilder &builder, Location loc, Value linearId, ArrayRef<int64_t> shape);

  /// Check if this is slice of some other layout.
  bool isSliceOf(const xegpu::DistributeLayoutAttr &other) { return false; }

  /// Check if this is identical to some other layout.
  bool isEqualTo(const xegpu::DistributeLayoutAttr &other);

  static constexpr ::llvm::StringLiteral name = "xegpu.layout";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  using Base::getChecked;
  static LayoutAttr get(::mlir::MLIRContext *context, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static LayoutAttr get(::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> inst_data);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> inst_data);
  static LayoutAttr get(::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> inst_data, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> inst_data, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data);
  static LayoutAttr get(::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data);
  static LayoutAttr get(::mlir::MLIRContext *context, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static LayoutAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"layout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  DenseI32ArrayAttr getSgLayout() const;
  DenseI32ArrayAttr getSgData() const;
  DenseI32ArrayAttr getInstData() const;
  DenseI32ArrayAttr getLaneLayout() const;
  DenseI32ArrayAttr getLaneData() const;
  DenseI32ArrayAttr getOrder() const;
};
namespace detail {

struct SliceAttrStorage;

} // namespace detail
class SliceAttr : public ::mlir::Attribute::AttrBase<SliceAttr, ::mlir::Attribute, detail::SliceAttrStorage, ::mlir::xegpu::DistributeLayoutAttr::Trait> {
public:
  using Base::Base;
  int64_t getRank() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    return parent.getRank() - attr.getDims().size();
  }

  DenseI32ArrayAttr getOrder() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    return parent.getOrder();
  }

  bool isForWorkgroup() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    return parent.isForWorkgroup();
  }

  bool isForSubgroup() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    return parent.isForSubgroup();
  }

  /// Returns the SgLayout of the attribute, computed by applying
  /// the slice dimensions to the underlying LayoutAttr.
  SmallVector<int64_t> getEffectiveSgLayoutAsInt() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    auto layout = parent.getEffectiveSgLayoutAsInt();
    if (layout.size()) {
      ArrayRef<int64_t> dims = attr.getDims().asArrayRef();
      return XeGPUDialect::slice(ArrayRef<int64_t>(layout), dims);
    }
    return {};
  }

  /// Returns the SgData of the attribute, computed by applying
  /// the slice dimensions to the underlying LayoutAttr.
  SmallVector<int64_t> getEffectiveSgDataAsInt() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    auto data = parent.getEffectiveSgDataAsInt();
    if (data.size()) {
      ArrayRef<int64_t> dims = attr.getDims().asArrayRef();
      return XeGPUDialect::slice(ArrayRef<int64_t>(data), dims);
    }
    return {};
  }

  /// Returns the InstData of the attribute, computed by applying
  /// the slice dimensions to the underlying LayoutAttr.
  SmallVector<int64_t> getEffectiveInstDataAsInt() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    auto inst = parent.getEffectiveInstDataAsInt();
    if (inst.size()) {
      ArrayRef<int64_t> dims = attr.getDims().asArrayRef();
      return XeGPUDialect::slice(llvm::ArrayRef<int64_t>(inst), dims);
    }
    return {};
  }

  /// Returns the LaneLayout of the attribute, computed by applying
  /// the slice dimensions to the underlying LayoutAttr.
  SmallVector<int64_t> getEffectiveLaneLayoutAsInt() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    auto layout = parent.getEffectiveLaneLayoutAsInt();
    if (layout.size()) {
      ArrayRef<int64_t> dims = attr.getDims().asArrayRef();
      return XeGPUDialect::slice(llvm::ArrayRef<int64_t>(layout), dims);
    }
    return {};
  }

  /// Returns the LaneData of the attribute, computed by applying
  /// the slice dimensions to the underlying LayoutAttr.
  SmallVector<int64_t> getEffectiveLaneDataAsInt() const {
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    auto data = parent.getEffectiveLaneDataAsInt();
    if (data.size()) {
      ArrayRef<int64_t> dims = attr.getDims().asArrayRef();
      return XeGPUDialect::slice(llvm::ArrayRef<int64_t>(data), dims);
    }
    return {};
  }

  SliceAttr dropSgLayoutAndData() const{
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    parent = parent.dropSgLayoutAndData();
    if (!parent)
      return nullptr;
    return SliceAttr::get(getContext(), parent, attr.getDims());
  }

  SliceAttr dropInstData() const{
    SliceAttr attr = flatten();
    auto parent = dyn_cast<LayoutAttr>(attr.getParent());
    parent = parent.dropInstData();
    if (!parent)
      return nullptr;
    return SliceAttr::get(getContext(), parent, attr.getDims());
  }

  //set the layout for the sepcified unit dims: sg_data, inst_data and lane_data to 1
  DistributeLayoutAttr setUnitDimData(SetVector<int64_t> unitDims) const;

  //set the layout for the sepcified unit dims: sg_lane and lane_layout to 1
  DistributeLayoutAttr setUnitDimLayout(SetVector<int64_t> unitDims) const;

  /// flatten a nested SliceAttr, e.g., for 2-level nested SliceAttr
  /// #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 12]>, dims = [0]>, dims = [0]>
  /// it will coalese two slice operations and return a simplified SliceAttr
  /// #xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 12]>, dims = [0, 1]>
  SliceAttr flatten() const;

  /// Delinearizes a linear subgroup ID into its multidimensional indices
  /// based on the effective subgroup layout.
  FailureOr<SmallVector<Value>>
  delinearizeId(OpBuilder &builder, Location loc, Value linearId);

  /// Generates instructions to compute multidimensional coordinates for blocks
  /// assigned to a subgroup identified by linearId. The shape parameter
  /// represents the workgroup-level problem size. Each subgroup may access
  /// multiple blocks according to round-robin distribution rules.

  FailureOr<SmallVector<SmallVector<Value>>>
  computeDistributedCoords(OpBuilder &builder, Location loc, Value linearId, ArrayRef<int64_t> shape);

  /// Check if this is slice of some other layout.
  bool isSliceOf(const xegpu::DistributeLayoutAttr &other);

  /// Drop the slice dims to get the original layout.
  SliceAttr dropSliceDims(ArrayRef<int64_t> sliceDimsToDrop);

  /// Check if this is identical to some other layout.
  bool isEqualTo(const xegpu::DistributeLayoutAttr &other);
  static constexpr ::llvm::StringLiteral name = "xegpu.slice";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  using Base::getChecked;
  static SliceAttr get(::mlir::MLIRContext *context, xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims);
  static SliceAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"slice"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  xegpu::DistributeLayoutAttr getParent() const;
  DenseI64ArrayAttr getDims() const;
};
namespace detail {

struct RangeAttrStorage;

} // namespace detail
class RangeAttr : public ::mlir::Attribute::AttrBase<RangeAttr, ::mlir::Attribute, detail::RangeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xegpu.range";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  using Base::getChecked;
  static RangeAttr get(::mlir::MLIRContext *context, IntegerAttr start, IntegerAttr end);
  static RangeAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, IntegerAttr start, IntegerAttr end);
  static RangeAttr get(::mlir::MLIRContext *context, int start, int end);
  static RangeAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int start, int end);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, IntegerAttr start, IntegerAttr end);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, IntegerAttr start, IntegerAttr end);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"range"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  IntegerAttr getStart() const;
  IntegerAttr getEnd() const;
};
namespace detail {

struct MemLayoutAttrStorage;

} // namespace detail
class MemLayoutAttr : public ::mlir::Attribute::AttrBase<MemLayoutAttr, ::mlir::Attribute, detail::MemLayoutAttrStorage> {
public:
  using Base::Base;
  /// Get a specific attribute by name
  Attribute getAttr(StringRef name) const {
    return getAttrs().get(name);
  }

  /// Check if a specific attribute exists
  bool hasAttr(StringRef name) const {
    return getAttrs().contains(name);
  }

  ArrayAttr getStrideAttr() {
    return getAttrs().getAs<ArrayAttr>("stride");
  }

  ArrayAttr getBlockAttr() {
    return getAttrs().getAs<ArrayAttr>("block");
  }

  static constexpr ::llvm::StringLiteral name = "xegpu.mem_layout";
  static constexpr ::llvm::StringLiteral dialectName = "xegpu";
  static MemLayoutAttr get(::mlir::MLIRContext *context, DictionaryAttr attrs);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mem_layout"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  DictionaryAttr getAttrs() const;
};

} // namespace mlir::xegpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::BlockTensorDescAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::ScatterTensorDescAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::MemorySpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::CachePolicyAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::FenceScopeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::LayoutAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::SliceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::RangeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xegpu::MemLayoutAttr)

#endif // GET_ATTRDEF_CLASSES

