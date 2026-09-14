/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: triton_xla_ops.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::xla {

/// Spin-wait while reading u32 value(s) atomically.
/// Atomically read u32 value(s) from the memory location(s) specified by
///       $ptr. The memory ordering guarantees are specified by the
///       $mem_sync_(scope|semantic) attributes.
/// 
///       The loaded value is compared to $expected using the $comparator.
///       As long as the comparison returns true, the spin-wait loop continues.
///       If the comparison returns false, the spin-wait loop is finished.
/// 
///       NB: In the end only a single lane/thread within a program will
///       write to (or read from) a given memory location so the synchronization
///       happens between a single writing lane with a single reading lane. However,
///       the operation will wait until all lanes within the block have synchronized
///       so the end result is a block level memory synchronization.
/// 
///       For Memory Synchronization Scope, see comment on TTXLA_AtomicWriteOp.
///       The following memory synchronization semantics are allowed:
///         - relaxed: No memory ordering guarantees.
///         - acquire: No reads or writes in the current block/program can be
///           reordered before this wait. All writes in other programs that
///           `release` the same memory location are visible in the current block.
/// 
///       Internally this operation expands into elementwise
///       `ld.global.<semantic>.<scope>.u32` and a labelled loop.
class AtomicSpinWaitOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// Atomically write u32 value(s) to memory location(s).
/// Atomically write u32 value(s) to the memory location(s) specified by
///       $ptr. The memory ordering guarantees are specified by the
///       $mem_sync_(scope|semantic) attribute.
/// 
///       An optional $mask argument can be provided to specify which of the
///       sublanes/threads should be updated. The mask must be the same shape as
///       the $ptr.
/// 
///       Memory Synchronization Scope:
///         - cta: The write is visible to all threads within the same compute
///           thread-block.
///         - gpu: The write is visible to all threads within the same GPU.
///         - sys: The write is visible to all threads within the system.
/// 
///       The following memory synchronization semantics are allowed:
///         - relaxed: No memory ordering guarantees.
///         - release: All writes that are issued before this one will be
///           completed before the store is visible to other threads.
/// 
///       Internally this operation expands into st.global.<scope>.<semantic>.u32
class AtomicWriteOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// A block-level barrier for inter-GPU synchronization.
/// This operation implements a block-level barrier between multiple GPUs.
///     It takes a pointer of pointers to signal buffers (one from each
///     participating GPU), the rank of the current device, and a signal value to
///     write/wait for. The signal values on consequent calls to this operation must
///     increase monotonically for the same signal buffers pointer. A block on one
///     GPU synchronizes with blocks on other GPUs with the same block id.
/// 
///     *NOTE* Block orchestration order is not guaranteed across GPUs,
///     so there is a chance of a deadlock should the number of blocks be greater
///     than the number of SMs on a GPU. That is to say that if BlockN is launched
///     on GPU0 and BlockM on GPU1 and they both wait for each other to finish,
///     and there is no place to schedule more blocks on either GPU, then we
///     will have a deadlock. In general, the user of this operation should avoid
///     this by not oversubscribing the number of available SMs on a GPU. There is
///     also a secondary restriction that the number of threads per block must
///     exceed the number of participating GPUs. In the nominal case, where we are
///     within a single NV Link domain, this should not be an issue since the number
///     of participating GPUs would usually be much smaller than the number of
///     threads per block.
/// 
///     The synchronization logic is as follows:
///     1. Each GPU (identified by `device_rank`) writes the `signal` value to an
///        index `blockId * world_size + device_rank` in *every* buffer provided in
///        `$signal_buffers`.
///     2. After writing, GPU `r` polls its own buffer (`$signal_buffers[r]`).
///     3. It waits until the values at locations corresponding to all other ranks
///        (i.e., `blockId * world_size + [0..world_size)` are greater than or equal
///        to the `signal` value.
///     4. When the wait is over, all GPUs working on the same `blockId` have
///        successfully synchronized.
/// 
///     This operation ensures that GPUs in the same synchronization group do not
///     get more than one "wave" (or chunk) of data processing ahead of each other.
///     For clarity, a wave here is a set of work items that are processed together.
///     A single thread block for instance can process multiple waves. In such a
///     scenario, any given GPU can be at most 1 wave ahead of the other GPUs for
///     the same block id.
/// 
///     Internally, this operation uses TTXLA_Atomic(Write|SpinWait)Op on signal
///     buffers to synchronize all participating GPUs.
class BlockBarrierOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// Extract a tile from a pointer.
/// Offsets, strides, and sizes operands specify the result tensor tile
///     to extract. Shape and minor-to-major layout specify the layout of the source
///     pointer.
/// 
///     Assembly format:
///       from %src as memref<...> [%offsets] [%sizes] [%strides] : tensor_type
/// 
///     Example:
///       ```
///       %extracted_tensor = triton_xla.extract from %src
///         as memref<512x128xbf16, #xtile.layout<[1, 0]>>
///         [0, 0] [16, 64] [128, 1] : tensor<16x64xbf16>
///       ```
class ExtractOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// Extract the pointer to the given symmetric memory `address` on the given
///     `peer` device. An `address` should point to the memory of the given kernel
///     argument with `argument_index`. The result is calculated using the symmetric
///     memory `metadata` constructed at the runtime.
///     To calculate offsets operation also need to know the number of devices
///     participating in the collective operation (`world_size`).
class GetPeerPtrOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// Extract the device rank from the collectives metadata.
class GetRankOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// Get the thread ID within a triton kernel.
/// This operation returns the thread ID within a kernel.
///     Only the X dimension is supported. The result is a 32-bit
///     integer which in CUDA terms is threadIdx.x.
class GetTidOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// Insert a tile into a pointer.
/// Offsets, strides, and source tensor size are used to specify the tile to
///     insert. Shape and minor-to-major layout specify the layout of the
///     destination pointer.
/// 
///     Assembly format:
///       %src into %dst as memref<...> [%offsets] [%sizes] [%strides] : tensor_type
/// 
///     Example:
///       ```
///       triton_xla.insert %src into %dst
///         as memref<512x128xbf16, #xtile.layout<[1, 0]>>
///         [0, 0] [8, 8] [1, 1] : tensor<8x8xbf16>
///       ```
class InsertOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// A specialized version of unrealized_conversion_cast that converts a
///     memref to a pointer.
class MemrefToPtrOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// A specialized version of unrealized_conversion_cast that converts a
///     pointer to a memref.
class PtrToMemrefOp;

} // namespace mlir::triton::xla
namespace mlir::triton::xla {

/// Remove a size-1 dimension from a tensor.
class SqueezeDimsOp;

} // namespace mlir::triton::xla
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::AtomicSpinWaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AtomicSpinWaitOpGenericAdaptorBase {
public:
  struct Properties {
    using comparatorTy = ::mlir::triton::xla::ComparatorAttr;
    comparatorTy comparator;

    auto getComparator() const {
      auto &propStorage = this->comparator;
      return ::llvm::cast<::mlir::triton::xla::ComparatorAttr>(propStorage);
    }
    void setComparator(const ::mlir::triton::xla::ComparatorAttr &propValue) {
      this->comparator = propValue;
    }
    using mem_sync_scopeTy = ::mlir::triton::MemSyncScopeAttr;
    mem_sync_scopeTy mem_sync_scope;

    auto getMemSyncScope() const {
      auto &propStorage = this->mem_sync_scope;
      return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(propStorage);
    }
    void setMemSyncScope(const ::mlir::triton::MemSyncScopeAttr &propValue) {
      this->mem_sync_scope = propValue;
    }
    using mem_sync_semanticTy = ::mlir::triton::MemSemanticAttr;
    mem_sync_semanticTy mem_sync_semantic;

    auto getMemSyncSemantic() const {
      auto &propStorage = this->mem_sync_semantic;
      return ::llvm::cast<::mlir::triton::MemSemanticAttr>(propStorage);
    }
    void setMemSyncSemantic(const ::mlir::triton::MemSemanticAttr &propValue) {
      this->mem_sync_semantic = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.comparator == this->comparator &&
        rhs.mem_sync_scope == this->mem_sync_scope &&
        rhs.mem_sync_semantic == this->mem_sync_semantic &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AtomicSpinWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.atomic_spin_wait", odsAttrs.getContext());
  }

  AtomicSpinWaitOpGenericAdaptorBase(AtomicSpinWaitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::MemSyncScopeAttr getMemSyncScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().mem_sync_scope);
    return attr;
  }

  ::mlir::triton::MemSyncScope getMemSyncScope();
  ::mlir::triton::MemSemanticAttr getMemSyncSemanticAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().mem_sync_semantic);
    return attr;
  }

  ::mlir::triton::MemSemantic getMemSyncSemantic();
  ::mlir::triton::xla::ComparatorAttr getComparatorAttr() {
    auto attr = ::llvm::cast<::mlir::triton::xla::ComparatorAttr>(getProperties().comparator);
    return attr;
  }

  ::mlir::triton::xla::Comparator getComparator();
};

} // namespace detail
template <typename RangeT>
class AtomicSpinWaitOpGenericAdaptor : public detail::AtomicSpinWaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AtomicSpinWaitOpGenericAdaptorBase;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_expected = 1;
  static constexpr int odsIndex_mask = 2;
public:
  AtomicSpinWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AtomicSpinWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AtomicSpinWaitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AtomicSpinWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AtomicSpinWaitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  AtomicSpinWaitOpGenericAdaptor(RangeT values, const AtomicSpinWaitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AtomicSpinWaitOp, typename = std::enable_if_t<std::is_same_v<LateInst, AtomicSpinWaitOp>>>
  AtomicSpinWaitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getExpected() {
    return (*getODSOperands(1).begin());
  }

  ValueT getMask() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AtomicSpinWaitOpAdaptor : public AtomicSpinWaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AtomicSpinWaitOpGenericAdaptor::AtomicSpinWaitOpGenericAdaptor;
  AtomicSpinWaitOpAdaptor(AtomicSpinWaitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AtomicSpinWaitOp : public ::mlir::Op<AtomicSpinWaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicSpinWaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AtomicSpinWaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_expected = 1;
  static constexpr int odsIndex_mask = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("comparator"), ::llvm::StringRef("mem_sync_scope"), ::llvm::StringRef("mem_sync_semantic")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getComparatorAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getComparatorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemSyncScopeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemSyncScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getMemSyncSemanticAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getMemSyncSemanticAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.atomic_spin_wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getExpected() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getMask() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::Type>{} : ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*operands.begin());
  }

  ::mlir::OpOperand &getPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getExpectedMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getMaskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::MemSyncScopeAttr getMemSyncScopeAttr() {
    return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().mem_sync_scope);
  }

  ::mlir::triton::MemSyncScope getMemSyncScope();
  ::mlir::triton::MemSemanticAttr getMemSyncSemanticAttr() {
    return ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().mem_sync_semantic);
  }

  ::mlir::triton::MemSemantic getMemSyncSemantic();
  ::mlir::triton::xla::ComparatorAttr getComparatorAttr() {
    return ::llvm::cast<::mlir::triton::xla::ComparatorAttr>(getProperties().comparator);
  }

  ::mlir::triton::xla::Comparator getComparator();
  void setMemSyncScopeAttr(::mlir::triton::MemSyncScopeAttr attr) {
    getProperties().mem_sync_scope = attr;
  }

  void setMemSyncScope(::mlir::triton::MemSyncScope attrValue);
  void setMemSyncSemanticAttr(::mlir::triton::MemSemanticAttr attr) {
    getProperties().mem_sync_semantic = attr;
  }

  void setMemSyncSemantic(::mlir::triton::MemSemantic attrValue);
  void setComparatorAttr(::mlir::triton::xla::ComparatorAttr attr) {
    getProperties().comparator = attr;
  }

  void setComparator(::mlir::triton::xla::Comparator attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic, ::mlir::triton::xla::ComparatorAttr comparator);
  static AtomicSpinWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic, ::mlir::triton::xla::ComparatorAttr comparator);
  static AtomicSpinWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic, ::mlir::triton::xla::ComparatorAttr comparator);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic, ::mlir::triton::xla::ComparatorAttr comparator);
  static AtomicSpinWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic, ::mlir::triton::xla::ComparatorAttr comparator);
  static AtomicSpinWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic, ::mlir::triton::xla::ComparatorAttr comparator);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic, ::mlir::triton::xla::Comparator comparator);
  static AtomicSpinWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic, ::mlir::triton::xla::Comparator comparator);
  static AtomicSpinWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic, ::mlir::triton::xla::Comparator comparator);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic, ::mlir::triton::xla::Comparator comparator);
  static AtomicSpinWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic, ::mlir::triton::xla::Comparator comparator);
  static AtomicSpinWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value expected, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic, ::mlir::triton::xla::Comparator comparator);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AtomicSpinWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AtomicSpinWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AtomicSpinWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AtomicSpinWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::AtomicSpinWaitOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::AtomicWriteOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AtomicWriteOpGenericAdaptorBase {
public:
  struct Properties {
    using mem_sync_scopeTy = ::mlir::triton::MemSyncScopeAttr;
    mem_sync_scopeTy mem_sync_scope;

    auto getMemSyncScope() const {
      auto &propStorage = this->mem_sync_scope;
      return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(propStorage);
    }
    void setMemSyncScope(const ::mlir::triton::MemSyncScopeAttr &propValue) {
      this->mem_sync_scope = propValue;
    }
    using mem_sync_semanticTy = ::mlir::triton::MemSemanticAttr;
    mem_sync_semanticTy mem_sync_semantic;

    auto getMemSyncSemantic() const {
      auto &propStorage = this->mem_sync_semantic;
      return ::llvm::cast<::mlir::triton::MemSemanticAttr>(propStorage);
    }
    void setMemSyncSemantic(const ::mlir::triton::MemSemanticAttr &propValue) {
      this->mem_sync_semantic = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.mem_sync_scope == this->mem_sync_scope &&
        rhs.mem_sync_semantic == this->mem_sync_semantic &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AtomicWriteOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.atomic_write", odsAttrs.getContext());
  }

  AtomicWriteOpGenericAdaptorBase(AtomicWriteOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::MemSyncScopeAttr getMemSyncScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().mem_sync_scope);
    return attr;
  }

  ::mlir::triton::MemSyncScope getMemSyncScope();
  ::mlir::triton::MemSemanticAttr getMemSyncSemanticAttr() {
    auto attr = ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().mem_sync_semantic);
    return attr;
  }

  ::mlir::triton::MemSemantic getMemSyncSemantic();
};

} // namespace detail
template <typename RangeT>
class AtomicWriteOpGenericAdaptor : public detail::AtomicWriteOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AtomicWriteOpGenericAdaptorBase;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_value = 1;
  static constexpr int odsIndex_mask = 2;
public:
  AtomicWriteOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AtomicWriteOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AtomicWriteOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AtomicWriteOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AtomicWriteOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  AtomicWriteOpGenericAdaptor(RangeT values, const AtomicWriteOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AtomicWriteOp, typename = std::enable_if_t<std::is_same_v<LateInst, AtomicWriteOp>>>
  AtomicWriteOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getPtr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getValue() {
    return (*getODSOperands(1).begin());
  }

  ValueT getMask() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AtomicWriteOpAdaptor : public AtomicWriteOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AtomicWriteOpGenericAdaptor::AtomicWriteOpGenericAdaptor;
  AtomicWriteOpAdaptor(AtomicWriteOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AtomicWriteOp : public ::mlir::Op<AtomicWriteOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicWriteOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AtomicWriteOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ptr = 0;
  static constexpr int odsIndex_value = 1;
  static constexpr int odsIndex_mask = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("mem_sync_scope"), ::llvm::StringRef("mem_sync_semantic")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMemSyncScopeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMemSyncScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemSyncSemanticAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemSyncSemanticAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.atomic_write");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getPtr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getMask() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::Type>{} : ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*operands.begin());
  }

  ::mlir::OpOperand &getPtrMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getValueMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getMaskMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::triton::MemSyncScopeAttr getMemSyncScopeAttr() {
    return ::llvm::cast<::mlir::triton::MemSyncScopeAttr>(getProperties().mem_sync_scope);
  }

  ::mlir::triton::MemSyncScope getMemSyncScope();
  ::mlir::triton::MemSemanticAttr getMemSyncSemanticAttr() {
    return ::llvm::cast<::mlir::triton::MemSemanticAttr>(getProperties().mem_sync_semantic);
  }

  ::mlir::triton::MemSemantic getMemSyncSemantic();
  void setMemSyncScopeAttr(::mlir::triton::MemSyncScopeAttr attr) {
    getProperties().mem_sync_scope = attr;
  }

  void setMemSyncScope(::mlir::triton::MemSyncScope attrValue);
  void setMemSyncSemanticAttr(::mlir::triton::MemSemanticAttr attr) {
    getProperties().mem_sync_semantic = attr;
  }

  void setMemSyncSemantic(::mlir::triton::MemSemantic attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic);
  static AtomicWriteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic);
  static AtomicWriteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic);
  static AtomicWriteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic);
  static AtomicWriteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScopeAttr mem_sync_scope, ::mlir::triton::MemSemanticAttr mem_sync_semantic);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic);
  static AtomicWriteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic);
  static AtomicWriteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic);
  static AtomicWriteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic);
  static AtomicWriteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value value, /*optional*/::mlir::Value mask, ::mlir::triton::MemSyncScope mem_sync_scope, ::mlir::triton::MemSemantic mem_sync_semantic);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AtomicWriteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AtomicWriteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AtomicWriteOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AtomicWriteOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::AtomicWriteOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::BlockBarrierOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class BlockBarrierOpGenericAdaptorBase {
public:
  struct Properties {
    using world_sizeTy = ::mlir::IntegerAttr;
    world_sizeTy world_size;

    auto getWorldSize() const {
      auto &propStorage = this->world_size;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setWorldSize(const ::mlir::IntegerAttr &propValue) {
      this->world_size = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.world_size == this->world_size &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  BlockBarrierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.block_barrier", odsAttrs.getContext());
  }

  BlockBarrierOpGenericAdaptorBase(BlockBarrierOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getWorldSizeAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().world_size);
    return attr;
  }

  uint32_t getWorldSize();
};

} // namespace detail
template <typename RangeT>
class BlockBarrierOpGenericAdaptor : public detail::BlockBarrierOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BlockBarrierOpGenericAdaptorBase;
  static constexpr int odsIndex_signal_buffers = 0;
  static constexpr int odsIndex_device_rank = 1;
  static constexpr int odsIndex_signal_value = 2;
public:
  BlockBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  BlockBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : BlockBarrierOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  BlockBarrierOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : BlockBarrierOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  BlockBarrierOpGenericAdaptor(RangeT values, const BlockBarrierOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = BlockBarrierOp, typename = std::enable_if_t<std::is_same_v<LateInst, BlockBarrierOp>>>
  BlockBarrierOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSignalBuffers() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDeviceRank() {
    return (*getODSOperands(1).begin());
  }

  ValueT getSignalValue() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BlockBarrierOpAdaptor : public BlockBarrierOpGenericAdaptor<::mlir::ValueRange> {
public:
  using BlockBarrierOpGenericAdaptor::BlockBarrierOpGenericAdaptor;
  BlockBarrierOpAdaptor(BlockBarrierOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class BlockBarrierOp : public ::mlir::Op<BlockBarrierOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockBarrierOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BlockBarrierOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_signal_buffers = 0;
  static constexpr int odsIndex_device_rank = 1;
  static constexpr int odsIndex_signal_value = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("world_size")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getWorldSizeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getWorldSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.block_barrier");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getSignalBuffers() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getDeviceRank() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getSignalValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getSignalBuffersMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDeviceRankMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getSignalValueMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getWorldSizeAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().world_size);
  }

  uint32_t getWorldSize();
  void setWorldSizeAttr(::mlir::IntegerAttr attr) {
    getProperties().world_size = attr;
  }

  void setWorldSize(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, ::mlir::IntegerAttr world_size);
  static BlockBarrierOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, ::mlir::IntegerAttr world_size);
  static BlockBarrierOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, ::mlir::IntegerAttr world_size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, ::mlir::IntegerAttr world_size);
  static BlockBarrierOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, ::mlir::IntegerAttr world_size);
  static BlockBarrierOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, ::mlir::IntegerAttr world_size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, uint32_t world_size);
  static BlockBarrierOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, uint32_t world_size);
  static BlockBarrierOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, uint32_t world_size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, uint32_t world_size);
  static BlockBarrierOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, uint32_t world_size);
  static BlockBarrierOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value signal_buffers, ::mlir::Value device_rank, ::mlir::Value signal_value, uint32_t world_size);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static BlockBarrierOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static BlockBarrierOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static BlockBarrierOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static BlockBarrierOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::BlockBarrierOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::ExtractOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExtractOpGenericAdaptorBase {
public:
  struct Properties {
    using src_layoutTy = ::mlir::DenseI64ArrayAttr;
    src_layoutTy src_layout;

    auto getSrcLayout() const {
      auto &propStorage = this->src_layout;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setSrcLayout(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->src_layout = propValue;
    }
    using src_shapeTy = ::mlir::DenseI64ArrayAttr;
    src_shapeTy src_shape;

    auto getSrcShape() const {
      auto &propStorage = this->src_shape;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setSrcShape(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->src_shape = propValue;
    }
    using static_offsetsTy = ::mlir::DenseI64ArrayAttr;
    static_offsetsTy static_offsets;

    auto getStaticOffsets() const {
      auto &propStorage = this->static_offsets;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setStaticOffsets(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->static_offsets = propValue;
    }
    using static_sizesTy = ::mlir::DenseI64ArrayAttr;
    static_sizesTy static_sizes;

    auto getStaticSizes() const {
      auto &propStorage = this->static_sizes;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setStaticSizes(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->static_sizes = propValue;
    }
    using static_stridesTy = ::mlir::DenseI64ArrayAttr;
    static_stridesTy static_strides;

    auto getStaticStrides() const {
      auto &propStorage = this->static_strides;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setStaticStrides(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->static_strides = propValue;
    }
    using operandSegmentSizesTy = std::array<int32_t, 4>;
    operandSegmentSizesTy operandSegmentSizes;
    ::llvm::ArrayRef<int32_t> getOperandSegmentSizes() const {
      auto &propStorage = this->operandSegmentSizes;
      return propStorage;
    }
    void setOperandSegmentSizes(::llvm::ArrayRef<int32_t> propValue) {
      auto &propStorage = this->operandSegmentSizes;
      ::llvm::copy(propValue, propStorage.begin());
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.src_layout == this->src_layout &&
        rhs.src_shape == this->src_shape &&
        rhs.static_offsets == this->static_offsets &&
        rhs.static_sizes == this->static_sizes &&
        rhs.static_strides == this->static_strides &&
        rhs.operandSegmentSizes == this->operandSegmentSizes &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExtractOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.extract", odsAttrs.getContext());
  }

  ExtractOpGenericAdaptorBase(ExtractOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_offsets);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_sizes);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_strides);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getStaticStrides();
  ::mlir::DenseI64ArrayAttr getSrcShapeAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().src_shape);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getSrcShape();
  ::mlir::DenseI64ArrayAttr getSrcLayoutAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().src_layout);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getSrcLayout();
};

} // namespace detail
template <typename RangeT>
class ExtractOpGenericAdaptor : public detail::ExtractOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExtractOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_offsets = 1;
  static constexpr int odsIndex_sizes = 2;
  static constexpr int odsIndex_strides = 3;
public:
  ExtractOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExtractOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ExtractOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExtractOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs) : ExtractOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExtractOpGenericAdaptor(RangeT values, const ExtractOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExtractOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExtractOp>>>
  ExtractOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOffsets() {
    return getODSOperands(1);
  }

  RangeT getSizes() {
    return getODSOperands(2);
  }

  RangeT getStrides() {
    return getODSOperands(3);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExtractOpAdaptor : public ExtractOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExtractOpGenericAdaptor::ExtractOpGenericAdaptor;
  ExtractOpAdaptor(ExtractOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExtractOp : public ::mlir::Op<ExtractOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::OffsetSizeAndStrideOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExtractOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_offsets = 1;
  static constexpr int odsIndex_sizes = 2;
  static constexpr int odsIndex_strides = 3;
  using Properties = FoldAdaptor::Properties;
  using ::mlir::OpAsmOpInterface::Trait<ExtractOp>::getAsmBlockArgumentNames;
  using ::mlir::OpAsmOpInterface::Trait<ExtractOp>::getAsmBlockNames;
  using ::mlir::OpAsmOpInterface::Trait<ExtractOp>::getDefaultDialect;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("src_layout"), ::llvm::StringRef("src_shape"), ::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides"), ::llvm::StringRef("operandSegmentSizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getSrcLayoutAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getSrcLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSrcShapeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSrcShapeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticOffsetsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticOffsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStaticStridesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStaticStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
   return (*this)->getName().getAttributeNames().back();
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
   return name.getAttributeNames().back();
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.extract");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getOffsets() {
    return getODSOperands(1);
  }

  ::mlir::Operation::operand_range getSizes() {
    return getODSOperands(2);
  }

  ::mlir::Operation::operand_range getStrides() {
    return getODSOperands(3);
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getOffsetsMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  ::mlir::MutableOperandRange getStridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_offsets);
  }

  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_sizes);
  }

  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_strides);
  }

  ::llvm::ArrayRef<int64_t> getStaticStrides();
  ::mlir::DenseI64ArrayAttr getSrcShapeAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().src_shape);
  }

  ::llvm::ArrayRef<int64_t> getSrcShape();
  ::mlir::DenseI64ArrayAttr getSrcLayoutAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().src_layout);
  }

  ::llvm::ArrayRef<int64_t> getSrcLayout();
  void setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().static_offsets = attr;
  }

  void setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().static_sizes = attr;
  }

  void setStaticSizes(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().static_strides = attr;
  }

  void setStaticStrides(::llvm::ArrayRef<int64_t> attrValue);
  void setSrcShapeAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().src_shape = attr;
  }

  void setSrcShape(::llvm::ArrayRef<int64_t> attrValue);
  void setSrcLayoutAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().src_layout = attr;
  }

  void setSrcLayout(::llvm::ArrayRef<int64_t> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, RankedTensorType result_type, Value src, ValueRange offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> src_shape, ArrayRef<int64_t> src_layout);
  static ExtractOp create(::mlir::OpBuilder &builder, ::mlir::Location location, RankedTensorType result_type, Value src, ValueRange offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> src_shape, ArrayRef<int64_t> src_layout);
  static ExtractOp create(::mlir::ImplicitLocOpBuilder &builder, RankedTensorType result_type, Value src, ValueRange offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> src_shape, ArrayRef<int64_t> src_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, RankedTensorType result_type, Value src, ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> src_shape, ArrayRef<int64_t> src_layout);
  static ExtractOp create(::mlir::OpBuilder &builder, ::mlir::Location location, RankedTensorType result_type, Value src, ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> src_shape, ArrayRef<int64_t> src_layout);
  static ExtractOp create(::mlir::ImplicitLocOpBuilder &builder, RankedTensorType result_type, Value src, ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> src_shape, ArrayRef<int64_t> src_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr src_shape, ::mlir::DenseI64ArrayAttr src_layout);
  static ExtractOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr src_shape, ::mlir::DenseI64ArrayAttr src_layout);
  static ExtractOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr src_shape, ::mlir::DenseI64ArrayAttr src_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr src_shape, ::mlir::DenseI64ArrayAttr src_layout);
  static ExtractOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr src_shape, ::mlir::DenseI64ArrayAttr src_layout);
  static ExtractOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr src_shape, ::mlir::DenseI64ArrayAttr src_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> src_shape, ::llvm::ArrayRef<int64_t> src_layout);
  static ExtractOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> src_shape, ::llvm::ArrayRef<int64_t> src_layout);
  static ExtractOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> src_shape, ::llvm::ArrayRef<int64_t> src_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> src_shape, ::llvm::ArrayRef<int64_t> src_layout);
  static ExtractOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> src_shape, ::llvm::ArrayRef<int64_t> src_layout);
  static ExtractOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> src_shape, ::llvm::ArrayRef<int64_t> src_layout);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExtractOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExtractOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExtractOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExtractOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  SmallVector<Value> getOffsetsAsValues(::mlir::ImplicitLocOpBuilder &b) {
    return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
      getMixedOffsets());
  }

  SmallVector<Value> getSizesAsValues(::mlir::ImplicitLocOpBuilder &b) {
    return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
      getMixedSizes());
  }

  SmallVector<Value> getStridesAsValues(::mlir::ImplicitLocOpBuilder &b) {
    return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
      getMixedStrides());
  }

  /// Return the expected rank of each of the `static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks() {
    unsigned rank = getSrcShape().size();
    return {rank, rank, rank};
  }
  /// Return the number of leading operands before the `offsets`, `sizes` and
  /// and `strides` operands.
  static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::ExtractOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::GetPeerPtrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class GetPeerPtrOpGenericAdaptorBase {
public:
  struct Properties {
    using argument_indexTy = ::mlir::IntegerAttr;
    argument_indexTy argument_index;

    auto getArgumentIndex() const {
      auto &propStorage = this->argument_index;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setArgumentIndex(const ::mlir::IntegerAttr &propValue) {
      this->argument_index = propValue;
    }
    using world_sizeTy = ::mlir::IntegerAttr;
    world_sizeTy world_size;

    auto getWorldSize() const {
      auto &propStorage = this->world_size;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setWorldSize(const ::mlir::IntegerAttr &propValue) {
      this->world_size = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.argument_index == this->argument_index &&
        rhs.world_size == this->world_size &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  GetPeerPtrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.get_peer_ptr", odsAttrs.getContext());
  }

  GetPeerPtrOpGenericAdaptorBase(GetPeerPtrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getArgumentIndexAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().argument_index);
    return attr;
  }

  uint32_t getArgumentIndex();
  ::mlir::IntegerAttr getWorldSizeAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().world_size);
    return attr;
  }

  uint32_t getWorldSize();
};

} // namespace detail
template <typename RangeT>
class GetPeerPtrOpGenericAdaptor : public detail::GetPeerPtrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GetPeerPtrOpGenericAdaptorBase;
  static constexpr int odsIndex_address = 0;
  static constexpr int odsIndex_peer_id = 1;
  static constexpr int odsIndex_metadata = 2;
public:
  GetPeerPtrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GetPeerPtrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GetPeerPtrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  GetPeerPtrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : GetPeerPtrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  GetPeerPtrOpGenericAdaptor(RangeT values, const GetPeerPtrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = GetPeerPtrOp, typename = std::enable_if_t<std::is_same_v<LateInst, GetPeerPtrOp>>>
  GetPeerPtrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAddress() {
    return (*getODSOperands(0).begin());
  }

  ValueT getPeerId() {
    return (*getODSOperands(1).begin());
  }

  ValueT getMetadata() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GetPeerPtrOpAdaptor : public GetPeerPtrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GetPeerPtrOpGenericAdaptor::GetPeerPtrOpGenericAdaptor;
  GetPeerPtrOpAdaptor(GetPeerPtrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class GetPeerPtrOp : public ::mlir::Op<GetPeerPtrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetPeerPtrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GetPeerPtrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_address = 0;
  static constexpr int odsIndex_peer_id = 1;
  static constexpr int odsIndex_metadata = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("argument_index"), ::llvm::StringRef("world_size")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getArgumentIndexAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getArgumentIndexAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getWorldSizeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getWorldSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.get_peer_ptr");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getAddress() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPeerId() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getMetadata() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getAddressMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPeerIdMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getMetadataMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getArgumentIndexAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().argument_index);
  }

  uint32_t getArgumentIndex();
  ::mlir::IntegerAttr getWorldSizeAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().world_size);
  }

  uint32_t getWorldSize();
  void setArgumentIndexAttr(::mlir::IntegerAttr attr) {
    getProperties().argument_index = attr;
  }

  void setArgumentIndex(uint32_t attrValue);
  void setWorldSizeAttr(::mlir::IntegerAttr attr) {
    getProperties().world_size = attr;
  }

  void setWorldSize(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, ::mlir::IntegerAttr argument_index, ::mlir::IntegerAttr world_size);
  static GetPeerPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, ::mlir::IntegerAttr argument_index, ::mlir::IntegerAttr world_size);
  static GetPeerPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, ::mlir::IntegerAttr argument_index, ::mlir::IntegerAttr world_size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, ::mlir::IntegerAttr argument_index, ::mlir::IntegerAttr world_size);
  static GetPeerPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, ::mlir::IntegerAttr argument_index, ::mlir::IntegerAttr world_size);
  static GetPeerPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, ::mlir::IntegerAttr argument_index, ::mlir::IntegerAttr world_size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, uint32_t argument_index, uint32_t world_size);
  static GetPeerPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, uint32_t argument_index, uint32_t world_size);
  static GetPeerPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, uint32_t argument_index, uint32_t world_size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, uint32_t argument_index, uint32_t world_size);
  static GetPeerPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, uint32_t argument_index, uint32_t world_size);
  static GetPeerPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value peer_id, ::mlir::Value metadata, uint32_t argument_index, uint32_t world_size);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetPeerPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetPeerPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetPeerPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetPeerPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::GetPeerPtrOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::GetRankOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class GetRankOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  GetRankOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.get_rank", odsAttrs.getContext());
  }

  GetRankOpGenericAdaptorBase(GetRankOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class GetRankOpGenericAdaptor : public detail::GetRankOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GetRankOpGenericAdaptorBase;
  static constexpr int odsIndex_metadata = 0;
public:
  GetRankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GetRankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GetRankOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  GetRankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : GetRankOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  GetRankOpGenericAdaptor(RangeT values, const GetRankOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = GetRankOp, typename = std::enable_if_t<std::is_same_v<LateInst, GetRankOp>>>
  GetRankOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMetadata() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GetRankOpAdaptor : public GetRankOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GetRankOpGenericAdaptor::GetRankOpGenericAdaptor;
  GetRankOpAdaptor(GetRankOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class GetRankOp : public ::mlir::Op<GetRankOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetRankOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GetRankOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_metadata = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.get_rank");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getMetadata() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getMetadataMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value metadata);
  static GetRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value metadata);
  static GetRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value metadata);
  static GetRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value metadata);
  static GetRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value metadata);
  static GetRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value metadata);
  static GetRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value metadata);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::GetRankOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::GetTidOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class GetTidOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  GetTidOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.get_tid", odsAttrs.getContext());
  }

  GetTidOpGenericAdaptorBase(GetTidOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class GetTidOpGenericAdaptor : public detail::GetTidOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GetTidOpGenericAdaptorBase;
public:
  GetTidOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GetTidOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GetTidOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  GetTidOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : GetTidOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  GetTidOpGenericAdaptor(RangeT values, const GetTidOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = GetTidOp, typename = std::enable_if_t<std::is_same_v<LateInst, GetTidOp>>>
  GetTidOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GetTidOpAdaptor : public GetTidOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GetTidOpGenericAdaptor::GetTidOpGenericAdaptor;
  GetTidOpAdaptor(GetTidOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class GetTidOp : public ::mlir::Op<GetTidOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetTidOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GetTidOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.get_tid");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result);
  static GetTidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result);
  static GetTidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static GetTidOp create(::mlir::OpBuilder &builder, ::mlir::Location location);
  static GetTidOp create(::mlir::ImplicitLocOpBuilder &builder);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static GetTidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static GetTidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetTidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetTidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetTidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static GetTidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetTidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetTidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetTidOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static GetTidOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::GetTidOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::InsertOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class InsertOpGenericAdaptorBase {
public:
  struct Properties {
    using dst_layoutTy = ::mlir::DenseI64ArrayAttr;
    dst_layoutTy dst_layout;

    auto getDstLayout() const {
      auto &propStorage = this->dst_layout;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setDstLayout(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->dst_layout = propValue;
    }
    using dst_shapeTy = ::mlir::DenseI64ArrayAttr;
    dst_shapeTy dst_shape;

    auto getDstShape() const {
      auto &propStorage = this->dst_shape;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setDstShape(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->dst_shape = propValue;
    }
    using static_offsetsTy = ::mlir::DenseI64ArrayAttr;
    static_offsetsTy static_offsets;

    auto getStaticOffsets() const {
      auto &propStorage = this->static_offsets;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setStaticOffsets(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->static_offsets = propValue;
    }
    using static_sizesTy = ::mlir::DenseI64ArrayAttr;
    static_sizesTy static_sizes;

    auto getStaticSizes() const {
      auto &propStorage = this->static_sizes;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setStaticSizes(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->static_sizes = propValue;
    }
    using static_stridesTy = ::mlir::DenseI64ArrayAttr;
    static_stridesTy static_strides;

    auto getStaticStrides() const {
      auto &propStorage = this->static_strides;
      return ::llvm::cast<::mlir::DenseI64ArrayAttr>(propStorage);
    }
    void setStaticStrides(const ::mlir::DenseI64ArrayAttr &propValue) {
      this->static_strides = propValue;
    }
    using operandSegmentSizesTy = std::array<int32_t, 5>;
    operandSegmentSizesTy operandSegmentSizes;
    ::llvm::ArrayRef<int32_t> getOperandSegmentSizes() const {
      auto &propStorage = this->operandSegmentSizes;
      return propStorage;
    }
    void setOperandSegmentSizes(::llvm::ArrayRef<int32_t> propValue) {
      auto &propStorage = this->operandSegmentSizes;
      ::llvm::copy(propValue, propStorage.begin());
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.dst_layout == this->dst_layout &&
        rhs.dst_shape == this->dst_shape &&
        rhs.static_offsets == this->static_offsets &&
        rhs.static_sizes == this->static_sizes &&
        rhs.static_strides == this->static_strides &&
        rhs.operandSegmentSizes == this->operandSegmentSizes &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  InsertOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.insert", odsAttrs.getContext());
  }

  InsertOpGenericAdaptorBase(InsertOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_offsets);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_sizes);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_strides);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getStaticStrides();
  ::mlir::DenseI64ArrayAttr getDstShapeAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().dst_shape);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getDstShape();
  ::mlir::DenseI64ArrayAttr getDstLayoutAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().dst_layout);
    return attr;
  }

  ::llvm::ArrayRef<int64_t> getDstLayout();
};

} // namespace detail
template <typename RangeT>
class InsertOpGenericAdaptor : public detail::InsertOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InsertOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_dst = 1;
  static constexpr int odsIndex_offsets = 2;
  static constexpr int odsIndex_sizes = 3;
  static constexpr int odsIndex_strides = 4;
public:
  InsertOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  InsertOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : InsertOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  InsertOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs) : InsertOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  InsertOpGenericAdaptor(RangeT values, const InsertOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = InsertOp, typename = std::enable_if_t<std::is_same_v<LateInst, InsertOp>>>
  InsertOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDst() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOffsets() {
    return getODSOperands(2);
  }

  RangeT getSizes() {
    return getODSOperands(3);
  }

  RangeT getStrides() {
    return getODSOperands(4);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InsertOpAdaptor : public InsertOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InsertOpGenericAdaptor::InsertOpGenericAdaptor;
  InsertOpAdaptor(InsertOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class InsertOp : public ::mlir::Op<InsertOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OffsetSizeAndStrideOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InsertOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InsertOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  static constexpr int odsIndex_dst = 1;
  static constexpr int odsIndex_offsets = 2;
  static constexpr int odsIndex_sizes = 3;
  static constexpr int odsIndex_strides = 4;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dst_layout"), ::llvm::StringRef("dst_shape"), ::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides"), ::llvm::StringRef("operandSegmentSizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDstLayoutAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDstLayoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getDstShapeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getDstShapeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticOffsetsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticOffsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticSizesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStaticStridesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStaticStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
   return (*this)->getName().getAttributeNames().back();
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
   return name.getAttributeNames().back();
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.insert");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getDst() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(1).begin());
  }

  ::mlir::Operation::operand_range getOffsets() {
    return getODSOperands(2);
  }

  ::mlir::Operation::operand_range getSizes() {
    return getODSOperands(3);
  }

  ::mlir::Operation::operand_range getStrides() {
    return getODSOperands(4);
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDstMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getOffsetsMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  ::mlir::MutableOperandRange getStridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_offsets);
  }

  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_sizes);
  }

  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().static_strides);
  }

  ::llvm::ArrayRef<int64_t> getStaticStrides();
  ::mlir::DenseI64ArrayAttr getDstShapeAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().dst_shape);
  }

  ::llvm::ArrayRef<int64_t> getDstShape();
  ::mlir::DenseI64ArrayAttr getDstLayoutAttr() {
    return ::llvm::cast<::mlir::DenseI64ArrayAttr>(getProperties().dst_layout);
  }

  ::llvm::ArrayRef<int64_t> getDstLayout();
  void setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().static_offsets = attr;
  }

  void setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().static_sizes = attr;
  }

  void setStaticSizes(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().static_strides = attr;
  }

  void setStaticStrides(::llvm::ArrayRef<int64_t> attrValue);
  void setDstShapeAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().dst_shape = attr;
  }

  void setDstShape(::llvm::ArrayRef<int64_t> attrValue);
  void setDstLayoutAttr(::mlir::DenseI64ArrayAttr attr) {
    getProperties().dst_layout = attr;
  }

  void setDstLayout(::llvm::ArrayRef<int64_t> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, Value dst, ValueRange offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> dst_shape, ArrayRef<int64_t> dst_layout);
  static InsertOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value src, Value dst, ValueRange offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> dst_shape, ArrayRef<int64_t> dst_layout);
  static InsertOp create(::mlir::ImplicitLocOpBuilder &builder, Value src, Value dst, ValueRange offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> dst_shape, ArrayRef<int64_t> dst_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, Value dst, ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> dst_shape, ArrayRef<int64_t> dst_layout);
  static InsertOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value src, Value dst, ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> dst_shape, ArrayRef<int64_t> dst_layout);
  static InsertOp create(::mlir::ImplicitLocOpBuilder &builder, Value src, Value dst, ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<int64_t> dst_shape, ArrayRef<int64_t> dst_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr dst_shape, ::mlir::DenseI64ArrayAttr dst_layout);
  static InsertOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr dst_shape, ::mlir::DenseI64ArrayAttr dst_layout);
  static InsertOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr dst_shape, ::mlir::DenseI64ArrayAttr dst_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr dst_shape, ::mlir::DenseI64ArrayAttr dst_layout);
  static InsertOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr dst_shape, ::mlir::DenseI64ArrayAttr dst_layout);
  static InsertOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides, ::mlir::DenseI64ArrayAttr dst_shape, ::mlir::DenseI64ArrayAttr dst_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> dst_shape, ::llvm::ArrayRef<int64_t> dst_layout);
  static InsertOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> dst_shape, ::llvm::ArrayRef<int64_t> dst_layout);
  static InsertOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> dst_shape, ::llvm::ArrayRef<int64_t> dst_layout);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> dst_shape, ::llvm::ArrayRef<int64_t> dst_layout);
  static InsertOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> dst_shape, ::llvm::ArrayRef<int64_t> dst_layout);
  static InsertOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value dst, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides, ::llvm::ArrayRef<int64_t> dst_shape, ::llvm::ArrayRef<int64_t> dst_layout);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static InsertOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static InsertOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static InsertOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static InsertOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
  SmallVector<Value> getOffsetsAsValues(::mlir::ImplicitLocOpBuilder &b) {
    return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
      getMixedOffsets());
  }

  SmallVector<Value> getSizesAsValues(::mlir::ImplicitLocOpBuilder &b) {
    return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
      getMixedSizes());
  }

  SmallVector<Value> getStridesAsValues(::mlir::ImplicitLocOpBuilder &b) {
    return ::mlir::getValueOrCreateConstantIndexOp(b, b.getLoc(),
      getMixedStrides());
  }

  /// Return the expected rank of each of the `static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks() {
    unsigned rank = getDstShape().size();
    return {rank, rank, rank};
  }
  /// Return the number of leading operands before the `offsets`, `sizes` and
  /// and `strides` operands.
  static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::InsertOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::MemrefToPtrOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MemrefToPtrOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  MemrefToPtrOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.memref_to_ptr", odsAttrs.getContext());
  }

  MemrefToPtrOpGenericAdaptorBase(MemrefToPtrOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class MemrefToPtrOpGenericAdaptor : public detail::MemrefToPtrOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MemrefToPtrOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
public:
  MemrefToPtrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MemrefToPtrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MemrefToPtrOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  MemrefToPtrOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : MemrefToPtrOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  MemrefToPtrOpGenericAdaptor(RangeT values, const MemrefToPtrOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MemrefToPtrOp, typename = std::enable_if_t<std::is_same_v<LateInst, MemrefToPtrOp>>>
  MemrefToPtrOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MemrefToPtrOpAdaptor : public MemrefToPtrOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MemrefToPtrOpGenericAdaptor::MemrefToPtrOpGenericAdaptor;
  MemrefToPtrOpAdaptor(MemrefToPtrOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MemrefToPtrOp : public ::mlir::Op<MemrefToPtrOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::triton::PointerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MemrefToPtrOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MemrefToPtrOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.memref_to_ptr");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src);
  static MemrefToPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src);
  static MemrefToPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static MemrefToPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static MemrefToPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MemrefToPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MemrefToPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static MemrefToPtrOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static MemrefToPtrOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::MemrefToPtrOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::PtrToMemrefOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class PtrToMemrefOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  PtrToMemrefOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.ptr_to_memref", odsAttrs.getContext());
  }

  PtrToMemrefOpGenericAdaptorBase(PtrToMemrefOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class PtrToMemrefOpGenericAdaptor : public detail::PtrToMemrefOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PtrToMemrefOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
public:
  PtrToMemrefOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  PtrToMemrefOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : PtrToMemrefOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  PtrToMemrefOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : PtrToMemrefOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  PtrToMemrefOpGenericAdaptor(RangeT values, const PtrToMemrefOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = PtrToMemrefOp, typename = std::enable_if_t<std::is_same_v<LateInst, PtrToMemrefOp>>>
  PtrToMemrefOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PtrToMemrefOpAdaptor : public PtrToMemrefOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PtrToMemrefOpGenericAdaptor::PtrToMemrefOpGenericAdaptor;
  PtrToMemrefOpAdaptor(PtrToMemrefOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class PtrToMemrefOp : public ::mlir::Op<PtrToMemrefOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PtrToMemrefOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PtrToMemrefOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.ptr_to_memref");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::PointerType> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::PointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src);
  static PtrToMemrefOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src);
  static PtrToMemrefOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static PtrToMemrefOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static PtrToMemrefOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PtrToMemrefOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static PtrToMemrefOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static PtrToMemrefOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static PtrToMemrefOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::PtrToMemrefOp)

namespace mlir::triton::xla {


//===----------------------------------------------------------------------===//
// ::mlir::triton::xla::SqueezeDimsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SqueezeDimsOpGenericAdaptorBase {
public:
  struct Properties {
    using axisTy = ::mlir::IntegerAttr;
    axisTy axis;

    auto getAxis() const {
      auto &propStorage = this->axis;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setAxis(const ::mlir::IntegerAttr &propValue) {
      this->axis = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.axis == this->axis &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  SqueezeDimsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("triton_xla.squeeze_dims", odsAttrs.getContext());
  }

  SqueezeDimsOpGenericAdaptorBase(SqueezeDimsOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getAxisAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().axis);
    return attr;
  }

  uint32_t getAxis();
};

} // namespace detail
template <typename RangeT>
class SqueezeDimsOpGenericAdaptor : public detail::SqueezeDimsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SqueezeDimsOpGenericAdaptorBase;
  static constexpr int odsIndex_src = 0;
public:
  SqueezeDimsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SqueezeDimsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SqueezeDimsOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  SqueezeDimsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : SqueezeDimsOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  SqueezeDimsOpGenericAdaptor(RangeT values, const SqueezeDimsOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SqueezeDimsOp, typename = std::enable_if_t<std::is_same_v<LateInst, SqueezeDimsOp>>>
  SqueezeDimsOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SqueezeDimsOpAdaptor : public SqueezeDimsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SqueezeDimsOpGenericAdaptor::SqueezeDimsOpGenericAdaptor;
  SqueezeDimsOpAdaptor(SqueezeDimsOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SqueezeDimsOp : public ::mlir::Op<SqueezeDimsOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultElementType> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SqueezeDimsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SqueezeDimsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_src = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("axis")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAxisAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAxisAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("triton_xla.squeeze_dims");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getSrc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getSrcMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getAxisAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().axis);
  }

  uint32_t getAxis();
  void setAxisAttr(::mlir::IntegerAttr attr) {
    getProperties().axis = attr;
  }

  void setAxis(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::IntegerAttr axis);
  static SqueezeDimsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src, ::mlir::IntegerAttr axis);
  static SqueezeDimsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src, ::mlir::IntegerAttr axis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::IntegerAttr axis);
  static SqueezeDimsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::IntegerAttr axis);
  static SqueezeDimsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::IntegerAttr axis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, uint32_t axis);
  static SqueezeDimsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value src, uint32_t axis);
  static SqueezeDimsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value src, uint32_t axis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, uint32_t axis);
  static SqueezeDimsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value src, uint32_t axis);
  static SqueezeDimsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value src, uint32_t axis);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SqueezeDimsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SqueezeDimsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static SqueezeDimsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static SqueezeDimsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::xla
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::xla::SqueezeDimsOp)


#endif // GET_OP_CLASSES

