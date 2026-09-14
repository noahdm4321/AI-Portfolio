/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVWSOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvws {

/// Get buffer from aref
class ArefBufferOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Create an asynchronous reference.
/// Create an asynchronous reference.
/// 
///     Takes as inputs a variadic number of buffers, and returns an ARef.
///     The inputs are expected to be array-like (i.e., Tensor, MemDesc, etc)
///     and the first axis of the shape should match between all inputs, representing
///     multi-buffering of the values.
class ArefCreateOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Enter ArefGet region where the buffer can be used to read data
/// Enter a "region" where you can freely read from the buffer)
///                       These ArefGet "regions" can span multiple iterations.
class ArefGetEnterOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Exit ArefGet region, where the buffer should no longer be used
/// Leave the region where you can freely read from the buffer).
///                       These ArefGet "regions" can span multiple iterations.
class ArefGetExitOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Enter ArefPut region where the buffer can be used to read data
/// Enter a "region" where you can freely write to the buffer)
///                       These ArefPut "regions" can span multiple iterations.
class ArefPutEnterOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Exit ArefPut region, where the buffer should no longer be used
/// Leave the region where you can freely write to the buffer).
///                       These ArefPut "regions" can span multiple iterations.
class ArefPutExitOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Consumer releases the token
/// The consumer will release the token and signal the producer
///     that the buffers are ready to be filled.
class ConsumerReleaseOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Consumer awaits buffer readiness
/// The consumer will wait for the buffer to be ready
///     to be consumed. If the buffers are not ready, the consumer will wait to be
///     signalled by the producer which finishes filling the buffers and
///     releases the token.
class ConsumerWaitOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Create a token to be used for synchronizations in communication channels
/// A token will be used by the producer and consumer to synchronize.
///     The producer will acquire and hold the token, until it has filled the buffers,
///     and signal the waiting consumer.
///     The consumer will hold the token until it has consumed the buffers,
///     and will signal the waiting producer trying to acquire the token.
class CreateTokenOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// gather multiple rows from a descriptor into shared memory
/// This op behaves exactly like the op with the same name in Triton Dialect, but the result of the load is stored into shared memory.
///     The execution is still synchronous.
class DescriptorGatherOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Load from descriptor and store into shared memory
/// This op behaves exactly like the op with the same name in Triton Dialect, but the result of the load is stored into shared memory.
///     The execution is still synchronous.
class DescriptorLoadOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Producer acquires a token to fill buffers
/// The producer will try to acquire the token prior to filling
///     the buffers. If the buffers are not ready to be filled, the producer will wait to be
///     signalled by the consumer which finishes consuming the buffers and
///     releases the token.
class ProducerAcquireOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Producer commits the buffer changes
/// The producer will release the token and signal the consumer
///     that the buffers are ready to be consumed.
class ProducerCommitOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Container Op for Warp Specialization
/// Higher level container for Warp Specialization Analysis.
/// 
///     Contains a variadic number warp groups, with
///     the number of warps in each group, plus a region to hold the
///     computation for that warp group.
/// 
///     The results of this op, if any, are those of the first region, as returned by
///     nvws.warp_group.yield op.
/// 
///     nvws.warp_group should be lowered to ttg.warp_specialize
///     before execution.
class WarpGroupOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// Terminator for a warp group region
/// Warp groups are expected to return values via referential modification
///     of their inputs. Thus, the warp_group.return op takes no values to
///     return from the warp group.
class WarpGroupReturnOp;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// yield from the first region of `nvws.warp_group`
/// This op is equivalent to ttg.warp_yield op for ttg.warp_specialize op.
/// 
///     TODO: Decide if we should move nvws.warp_group to TritonGPU, or continue to
///     have TritonGPU depend on NVWS. In the former case, this op can be removed.
///     The latter one involves a circular dependency between TritonGPU and NVWS.
class WarpGroupYieldOp;

} // namespace mlir::triton::nvws
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ArefBufferOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ArefBufferOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ArefBufferOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.aref.buffer", odsAttrs.getContext());
  }

  ArefBufferOpGenericAdaptorBase(ArefBufferOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ArefBufferOpGenericAdaptor : public detail::ArefBufferOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ArefBufferOpGenericAdaptorBase;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_token = 1;
  static constexpr int odsIndex_stage = 2;
public:
  ArefBufferOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ArefBufferOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ArefBufferOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ArefBufferOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ArefBufferOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ArefBufferOpGenericAdaptor(RangeT values, const ArefBufferOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ArefBufferOp, typename = std::enable_if_t<std::is_same_v<LateInst, ArefBufferOp>>>
  ArefBufferOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAref() {
    return (*getODSOperands(0).begin());
  }

  ValueT getToken() {
    return (*getODSOperands(1).begin());
  }

  ValueT getStage() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ArefBufferOpAdaptor : public ArefBufferOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ArefBufferOpGenericAdaptor::ArefBufferOpGenericAdaptor;
  ArefBufferOpAdaptor(ArefBufferOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ArefBufferOp : public ::mlir::Op<ArefBufferOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::triton::nvws::ArefStageInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ArefBufferOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ArefBufferOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_token = 1;
  static constexpr int odsIndex_stage = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.aref.buffer");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::nvws::ArefType> getAref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::nvws::ArefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getStage() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getArefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTokenMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getStageMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getBuffers() {
    return getODSResults(0);
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value aref, TypeRange bufferTypes, Value token);
  static ArefBufferOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value aref, TypeRange bufferTypes, Value token);
  static ArefBufferOp create(::mlir::ImplicitLocOpBuilder &builder, Value aref, TypeRange bufferTypes, Value token);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange buffers, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage);
  static ArefBufferOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange buffers, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage);
  static ArefBufferOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange buffers, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefBufferOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefBufferOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefBufferOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefBufferOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  void setStage(::mlir::Value stage);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ArefBufferOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ArefCreateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ArefCreateOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ArefCreateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.aref.create", odsAttrs.getContext());
  }

  ArefCreateOpGenericAdaptorBase(ArefCreateOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ArefCreateOpGenericAdaptor : public detail::ArefCreateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ArefCreateOpGenericAdaptorBase;
  static constexpr int odsIndex_buffers = 0;
public:
  ArefCreateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ArefCreateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ArefCreateOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ArefCreateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ArefCreateOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ArefCreateOpGenericAdaptor(RangeT values, const ArefCreateOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ArefCreateOp, typename = std::enable_if_t<std::is_same_v<LateInst, ArefCreateOp>>>
  ArefCreateOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getBuffers() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ArefCreateOpAdaptor : public ArefCreateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ArefCreateOpGenericAdaptor::ArefCreateOpGenericAdaptor;
  ArefCreateOpAdaptor(ArefCreateOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ArefCreateOp : public ::mlir::Op<ArefCreateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::triton::nvws::ArefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ArefCreateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ArefCreateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_buffers = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.aref.create");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getBuffers() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getBuffersMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::nvws::ArefType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::nvws::ArefType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange buffers);
  static ArefCreateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::ValueRange buffers);
  static ArefCreateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::ValueRange buffers);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefCreateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefCreateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefCreateOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefCreateOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ArefCreateOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ArefGetEnterOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ArefGetEnterOpGenericAdaptorBase {
public:
  struct Properties {
    using operandSegmentSizesTy = std::array<int32_t, 3>;
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
  ArefGetEnterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.aref.get.enter", odsAttrs.getContext());
  }

  ArefGetEnterOpGenericAdaptorBase(ArefGetEnterOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ArefGetEnterOpGenericAdaptor : public detail::ArefGetEnterOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ArefGetEnterOpGenericAdaptorBase;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_stage = 1;
  static constexpr int odsIndex_phase = 2;
public:
  ArefGetEnterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ArefGetEnterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ArefGetEnterOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ArefGetEnterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs) : ArefGetEnterOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ArefGetEnterOpGenericAdaptor(RangeT values, const ArefGetEnterOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ArefGetEnterOp, typename = std::enable_if_t<std::is_same_v<LateInst, ArefGetEnterOp>>>
  ArefGetEnterOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAref() {
    return (*getODSOperands(0).begin());
  }

  ValueT getStage() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getPhase() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ArefGetEnterOpAdaptor : public ArefGetEnterOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ArefGetEnterOpGenericAdaptor::ArefGetEnterOpGenericAdaptor;
  ArefGetEnterOpAdaptor(ArefGetEnterOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ArefGetEnterOp : public ::mlir::Op<ArefGetEnterOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::nvws::ArefStageInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ArefGetEnterOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ArefGetEnterOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_stage = 1;
  static constexpr int odsIndex_phase = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operandSegmentSizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
   return (*this)->getName().getAttributeNames().back();
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
   return name.getAttributeNames().back();
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.aref.get.enter");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::nvws::ArefType> getAref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::nvws::ArefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getStage() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPhase() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getArefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getStageMutable();
  ::mlir::MutableOperandRange getPhaseMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getBuffers() {
    return getODSResults(0);
  }

  ::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType>>(*getODSResults(1).begin());
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value aref, TypeRange bufferTypes, Type tokenType);
  static ArefGetEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value aref, TypeRange bufferTypes, Type tokenType);
  static ArefGetEnterOp create(::mlir::ImplicitLocOpBuilder &builder, Value aref, TypeRange bufferTypes, Type tokenType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange buffers, ::mlir::Type token, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static ArefGetEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange buffers, ::mlir::Type token, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static ArefGetEnterOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange buffers, ::mlir::Type token, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static ArefGetEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static ArefGetEnterOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefGetEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefGetEnterOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefGetEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefGetEnterOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  void setStage(::mlir::Value stage);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    return {};
  }

public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ArefGetEnterOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ArefGetExitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ArefGetExitOpGenericAdaptorBase {
public:
  struct Properties {
    using async_opsTy = ::mlir::ArrayAttr;
    async_opsTy async_ops;

    auto getAsyncOps() const {
      auto &propStorage = this->async_ops;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setAsyncOps(const ::mlir::ArrayAttr &propValue) {
      this->async_ops = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.async_ops == this->async_ops &&
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
  ArefGetExitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.aref.get.exit", odsAttrs.getContext());
  }

  ArefGetExitOpGenericAdaptorBase(ArefGetExitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getAsyncOpsAttr() {
    auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().async_ops);
    return attr;
  }

  ::mlir::ArrayAttr getAsyncOps();
};

} // namespace detail
template <typename RangeT>
class ArefGetExitOpGenericAdaptor : public detail::ArefGetExitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ArefGetExitOpGenericAdaptorBase;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_token = 1;
  static constexpr int odsIndex_stage = 2;
public:
  ArefGetExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ArefGetExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ArefGetExitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ArefGetExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ArefGetExitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ArefGetExitOpGenericAdaptor(RangeT values, const ArefGetExitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ArefGetExitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ArefGetExitOp>>>
  ArefGetExitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAref() {
    return (*getODSOperands(0).begin());
  }

  ValueT getToken() {
    return (*getODSOperands(1).begin());
  }

  ValueT getStage() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ArefGetExitOpAdaptor : public ArefGetExitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ArefGetExitOpGenericAdaptor::ArefGetExitOpGenericAdaptor;
  ArefGetExitOpAdaptor(ArefGetExitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ArefGetExitOp : public ::mlir::Op<ArefGetExitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::nvws::ArefStageInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ArefGetExitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ArefGetExitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_token = 1;
  static constexpr int odsIndex_stage = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async_ops")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncOpsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncOpsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.aref.get.exit");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::nvws::ArefType> getAref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::nvws::ArefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getStage() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getArefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTokenMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getStageMutable();
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
  ::mlir::ArrayAttr getAsyncOpsAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(getProperties().async_ops);
  }

  ::mlir::ArrayAttr getAsyncOps();
  void setAsyncOpsAttr(::mlir::ArrayAttr attr) {
    getProperties().async_ops = attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value aref, Value token, ArrayAttr async_ops);
  static ArefGetExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value aref, Value token, ArrayAttr async_ops);
  static ArefGetExitOp create(::mlir::ImplicitLocOpBuilder &builder, Value aref, Value token, ArrayAttr async_ops);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static ArefGetExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static ArefGetExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static ArefGetExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static ArefGetExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefGetExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefGetExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefGetExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefGetExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  void setStage(::mlir::Value stage);
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

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ArefGetExitOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ArefPutEnterOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ArefPutEnterOpGenericAdaptorBase {
public:
  struct Properties {
    using operandSegmentSizesTy = std::array<int32_t, 3>;
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
  ArefPutEnterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.aref.put.enter", odsAttrs.getContext());
  }

  ArefPutEnterOpGenericAdaptorBase(ArefPutEnterOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ArefPutEnterOpGenericAdaptor : public detail::ArefPutEnterOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ArefPutEnterOpGenericAdaptorBase;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_stage = 1;
  static constexpr int odsIndex_phase = 2;
public:
  ArefPutEnterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ArefPutEnterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ArefPutEnterOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ArefPutEnterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs) : ArefPutEnterOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ArefPutEnterOpGenericAdaptor(RangeT values, const ArefPutEnterOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ArefPutEnterOp, typename = std::enable_if_t<std::is_same_v<LateInst, ArefPutEnterOp>>>
  ArefPutEnterOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAref() {
    return (*getODSOperands(0).begin());
  }

  ValueT getStage() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  ValueT getPhase() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ArefPutEnterOpAdaptor : public ArefPutEnterOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ArefPutEnterOpGenericAdaptor::ArefPutEnterOpGenericAdaptor;
  ArefPutEnterOpAdaptor(ArefPutEnterOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ArefPutEnterOp : public ::mlir::Op<ArefPutEnterOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::nvws::ArefStageInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ArefPutEnterOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ArefPutEnterOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_stage = 1;
  static constexpr int odsIndex_phase = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operandSegmentSizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
   return (*this)->getName().getAttributeNames().back();
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
   return name.getAttributeNames().back();
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.aref.put.enter");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::nvws::ArefType> getAref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::nvws::ArefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getStage() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPhase() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getArefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getStageMutable();
  ::mlir::MutableOperandRange getPhaseMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getBuffers() {
    return getODSResults(0);
  }

  ::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType>>(*getODSResults(1).begin());
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value aref, TypeRange bufferTypes, Type tokenType);
  static ArefPutEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value aref, TypeRange bufferTypes, Type tokenType);
  static ArefPutEnterOp create(::mlir::ImplicitLocOpBuilder &builder, Value aref, TypeRange bufferTypes, Type tokenType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange buffers, ::mlir::Type token, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static ArefPutEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange buffers, ::mlir::Type token, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static ArefPutEnterOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange buffers, ::mlir::Type token, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static ArefPutEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static ArefPutEnterOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value aref, /*optional*/::mlir::Value stage, /*optional*/::mlir::Value phase);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefPutEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefPutEnterOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefPutEnterOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefPutEnterOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  void setStage(::mlir::Value stage);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    return {};
  }

public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ArefPutEnterOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ArefPutExitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ArefPutExitOpGenericAdaptorBase {
public:
  struct Properties {
    using async_opsTy = ::mlir::ArrayAttr;
    async_opsTy async_ops;

    auto getAsyncOps() const {
      auto &propStorage = this->async_ops;
      return ::llvm::cast<::mlir::ArrayAttr>(propStorage);
    }
    void setAsyncOps(const ::mlir::ArrayAttr &propValue) {
      this->async_ops = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.async_ops == this->async_ops &&
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
  ArefPutExitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.aref.put.exit", odsAttrs.getContext());
  }

  ArefPutExitOpGenericAdaptorBase(ArefPutExitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::ArrayAttr getAsyncOpsAttr() {
    auto attr = ::llvm::cast<::mlir::ArrayAttr>(getProperties().async_ops);
    return attr;
  }

  ::mlir::ArrayAttr getAsyncOps();
};

} // namespace detail
template <typename RangeT>
class ArefPutExitOpGenericAdaptor : public detail::ArefPutExitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ArefPutExitOpGenericAdaptorBase;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_token = 1;
  static constexpr int odsIndex_stage = 2;
public:
  ArefPutExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ArefPutExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ArefPutExitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ArefPutExitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ArefPutExitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ArefPutExitOpGenericAdaptor(RangeT values, const ArefPutExitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ArefPutExitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ArefPutExitOp>>>
  ArefPutExitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAref() {
    return (*getODSOperands(0).begin());
  }

  ValueT getToken() {
    return (*getODSOperands(1).begin());
  }

  ValueT getStage() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ArefPutExitOpAdaptor : public ArefPutExitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ArefPutExitOpGenericAdaptor::ArefPutExitOpGenericAdaptor;
  ArefPutExitOpAdaptor(ArefPutExitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ArefPutExitOp : public ::mlir::Op<ArefPutExitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::nvws::ArefStageInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ArefPutExitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ArefPutExitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_aref = 0;
  static constexpr int odsIndex_token = 1;
  static constexpr int odsIndex_stage = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async_ops")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAsyncOpsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAsyncOpsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.aref.put.exit");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::nvws::ArefType> getAref() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::nvws::ArefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::AsyncTokenType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getStage() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getArefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTokenMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getStageMutable();
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
  ::mlir::ArrayAttr getAsyncOpsAttr() {
    return ::llvm::cast<::mlir::ArrayAttr>(getProperties().async_ops);
  }

  ::mlir::ArrayAttr getAsyncOps();
  void setAsyncOpsAttr(::mlir::ArrayAttr attr) {
    getProperties().async_ops = attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value aref, Value token, ArrayAttr async_ops);
  static ArefPutExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Value aref, Value token, ArrayAttr async_ops);
  static ArefPutExitOp create(::mlir::ImplicitLocOpBuilder &builder, Value aref, Value token, ArrayAttr async_ops);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static ArefPutExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static ArefPutExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static ArefPutExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static ArefPutExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value aref, ::mlir::Value token, /*optional*/::mlir::Value stage, ::mlir::ArrayAttr async_ops);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefPutExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ArefPutExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefPutExitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ArefPutExitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  void setStage(::mlir::Value stage);
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

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ArefPutExitOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ConsumerReleaseOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConsumerReleaseOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConsumerReleaseOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.consumer_release", odsAttrs.getContext());
  }

  ConsumerReleaseOpGenericAdaptorBase(ConsumerReleaseOp op);

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
class ConsumerReleaseOpGenericAdaptor : public detail::ConsumerReleaseOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConsumerReleaseOpGenericAdaptorBase;
  static constexpr int odsIndex_token = 0;
  static constexpr int odsIndex_idx = 1;
public:
  ConsumerReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConsumerReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConsumerReleaseOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConsumerReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConsumerReleaseOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConsumerReleaseOpGenericAdaptor(RangeT values, const ConsumerReleaseOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConsumerReleaseOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConsumerReleaseOp>>>
  ConsumerReleaseOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  ValueT getIdx() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConsumerReleaseOpAdaptor : public ConsumerReleaseOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConsumerReleaseOpGenericAdaptor::ConsumerReleaseOpGenericAdaptor;
  ConsumerReleaseOpAdaptor(ConsumerReleaseOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConsumerReleaseOp : public ::mlir::Op<ConsumerReleaseOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConsumerReleaseOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConsumerReleaseOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_token = 0;
  static constexpr int odsIndex_idx = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.consumer_release");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getIdx() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getTokenMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getIdxMutable() {
    auto range = getODSOperandIndexAndLength(1);
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::Value idx);
  static ConsumerReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value token, ::mlir::Value idx);
  static ConsumerReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value token, ::mlir::Value idx);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx);
  static ConsumerReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx);
  static ConsumerReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConsumerReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConsumerReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConsumerReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConsumerReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ConsumerReleaseOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ConsumerWaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConsumerWaitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConsumerWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.consumer_wait", odsAttrs.getContext());
  }

  ConsumerWaitOpGenericAdaptorBase(ConsumerWaitOp op);

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
class ConsumerWaitOpGenericAdaptor : public detail::ConsumerWaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConsumerWaitOpGenericAdaptorBase;
  static constexpr int odsIndex_token = 0;
  static constexpr int odsIndex_idx = 1;
  static constexpr int odsIndex_phase = 2;
public:
  ConsumerWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConsumerWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConsumerWaitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConsumerWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConsumerWaitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConsumerWaitOpGenericAdaptor(RangeT values, const ConsumerWaitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConsumerWaitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConsumerWaitOp>>>
  ConsumerWaitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  ValueT getIdx() {
    return (*getODSOperands(1).begin());
  }

  ValueT getPhase() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConsumerWaitOpAdaptor : public ConsumerWaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConsumerWaitOpGenericAdaptor::ConsumerWaitOpGenericAdaptor;
  ConsumerWaitOpAdaptor(ConsumerWaitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConsumerWaitOp : public ::mlir::Op<ConsumerWaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConsumerWaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConsumerWaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_token = 0;
  static constexpr int odsIndex_idx = 1;
  static constexpr int odsIndex_phase = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.consumer_wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getIdx() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPhase() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getTokenMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getIdxMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPhaseMutable() {
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static ConsumerWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static ConsumerWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static ConsumerWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static ConsumerWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConsumerWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConsumerWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConsumerWaitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConsumerWaitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ConsumerWaitOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::CreateTokenOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CreateTokenOpGenericAdaptorBase {
public:
  struct Properties {
    using loadTypeTy = ::mlir::triton::nvws::TokenLoadTypeAttr;
    loadTypeTy loadType;

    auto getLoadType() const {
      auto &propStorage = this->loadType;
      return ::llvm::cast<::mlir::triton::nvws::TokenLoadTypeAttr>(propStorage);
    }
    void setLoadType(const ::mlir::triton::nvws::TokenLoadTypeAttr &propValue) {
      this->loadType = propValue;
    }
    using numBuffersTy = ::mlir::IntegerAttr;
    numBuffersTy numBuffers;

    auto getNumBuffers() const {
      auto &propStorage = this->numBuffers;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setNumBuffers(const ::mlir::IntegerAttr &propValue) {
      this->numBuffers = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.loadType == this->loadType &&
        rhs.numBuffers == this->numBuffers &&
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
  CreateTokenOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.create_token", odsAttrs.getContext());
  }

  CreateTokenOpGenericAdaptorBase(CreateTokenOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getNumBuffersAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().numBuffers);
    return attr;
  }

  uint32_t getNumBuffers();
  ::mlir::triton::nvws::TokenLoadTypeAttr getLoadTypeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::nvws::TokenLoadTypeAttr>(getProperties().loadType);
    return attr;
  }

  ::mlir::triton::nvws::TokenLoadType getLoadType();
};

} // namespace detail
template <typename RangeT>
class CreateTokenOpGenericAdaptor : public detail::CreateTokenOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CreateTokenOpGenericAdaptorBase;
public:
  CreateTokenOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CreateTokenOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CreateTokenOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CreateTokenOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CreateTokenOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CreateTokenOpGenericAdaptor(RangeT values, const CreateTokenOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CreateTokenOp, typename = std::enable_if_t<std::is_same_v<LateInst, CreateTokenOp>>>
  CreateTokenOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class CreateTokenOpAdaptor : public CreateTokenOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CreateTokenOpGenericAdaptor::CreateTokenOpGenericAdaptor;
  CreateTokenOpAdaptor(CreateTokenOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CreateTokenOp : public ::mlir::Op<CreateTokenOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateTokenOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CreateTokenOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("loadType"), ::llvm::StringRef("numBuffers")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getLoadTypeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getLoadTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNumBuffersAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNumBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.create_token");
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

  ::mlir::TypedValue<::mlir::TensorType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
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
  ::mlir::IntegerAttr getNumBuffersAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().numBuffers);
  }

  uint32_t getNumBuffers();
  ::mlir::triton::nvws::TokenLoadTypeAttr getLoadTypeAttr() {
    return ::llvm::cast<::mlir::triton::nvws::TokenLoadTypeAttr>(getProperties().loadType);
  }

  ::mlir::triton::nvws::TokenLoadType getLoadType();
  void setNumBuffersAttr(::mlir::IntegerAttr attr) {
    getProperties().numBuffers = attr;
  }

  void setNumBuffers(uint32_t attrValue);
  void setLoadTypeAttr(::mlir::triton::nvws::TokenLoadTypeAttr attr) {
    getProperties().loadType = attr;
  }

  void setLoadType(::mlir::triton::nvws::TokenLoadType attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint32_t numBuffers, triton::nvws::TokenLoadType loadType);
  static CreateTokenOp create(::mlir::OpBuilder &builder, ::mlir::Location location, uint32_t numBuffers, triton::nvws::TokenLoadType loadType);
  static CreateTokenOp create(::mlir::ImplicitLocOpBuilder &builder, uint32_t numBuffers, triton::nvws::TokenLoadType loadType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr numBuffers, ::mlir::triton::nvws::TokenLoadTypeAttr loadType);
  static CreateTokenOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::IntegerAttr numBuffers, ::mlir::triton::nvws::TokenLoadTypeAttr loadType);
  static CreateTokenOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::IntegerAttr numBuffers, ::mlir::triton::nvws::TokenLoadTypeAttr loadType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numBuffers, ::mlir::triton::nvws::TokenLoadTypeAttr loadType);
  static CreateTokenOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numBuffers, ::mlir::triton::nvws::TokenLoadTypeAttr loadType);
  static CreateTokenOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numBuffers, ::mlir::triton::nvws::TokenLoadTypeAttr loadType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t numBuffers, ::mlir::triton::nvws::TokenLoadType loadType);
  static CreateTokenOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, uint32_t numBuffers, ::mlir::triton::nvws::TokenLoadType loadType);
  static CreateTokenOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, uint32_t numBuffers, ::mlir::triton::nvws::TokenLoadType loadType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t numBuffers, ::mlir::triton::nvws::TokenLoadType loadType);
  static CreateTokenOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t numBuffers, ::mlir::triton::nvws::TokenLoadType loadType);
  static CreateTokenOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t numBuffers, ::mlir::triton::nvws::TokenLoadType loadType);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CreateTokenOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CreateTokenOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateTokenOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateTokenOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::CreateTokenOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::DescriptorGatherOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class DescriptorGatherOpGenericAdaptorBase {
public:
  struct Properties {
    using txCountTy = ::mlir::IntegerAttr;
    txCountTy txCount;

    auto getTxCount() const {
      auto &propStorage = this->txCount;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setTxCount(const ::mlir::IntegerAttr &propValue) {
      this->txCount = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.txCount == this->txCount &&
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
  DescriptorGatherOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.descriptor_gather", odsAttrs.getContext());
  }

  DescriptorGatherOpGenericAdaptorBase(DescriptorGatherOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getTxCountAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().txCount);
    return attr;
  }

  uint32_t getTxCount();
};

} // namespace detail
template <typename RangeT>
class DescriptorGatherOpGenericAdaptor : public detail::DescriptorGatherOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DescriptorGatherOpGenericAdaptorBase;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_x_offsets = 1;
  static constexpr int odsIndex_y_offset = 2;
  static constexpr int odsIndex_result = 3;
public:
  DescriptorGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DescriptorGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : DescriptorGatherOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  DescriptorGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : DescriptorGatherOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  DescriptorGatherOpGenericAdaptor(RangeT values, const DescriptorGatherOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = DescriptorGatherOp, typename = std::enable_if_t<std::is_same_v<LateInst, DescriptorGatherOp>>>
  DescriptorGatherOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDesc() {
    return (*getODSOperands(0).begin());
  }

  ValueT getXOffsets() {
    return (*getODSOperands(1).begin());
  }

  ValueT getYOffset() {
    return (*getODSOperands(2).begin());
  }

  ValueT getResult() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DescriptorGatherOpAdaptor : public DescriptorGatherOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DescriptorGatherOpGenericAdaptor::DescriptorGatherOpGenericAdaptor;
  DescriptorGatherOpAdaptor(DescriptorGatherOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class DescriptorGatherOp : public ::mlir::Op<DescriptorGatherOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::DescriptorOpInterface::Trait, ::mlir::triton::nvws::DescriptorLoadOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DescriptorGatherOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DescriptorGatherOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_x_offsets = 1;
  static constexpr int odsIndex_y_offset = 2;
  static constexpr int odsIndex_result = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("txCount")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getTxCountAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTxCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.descriptor_gather");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::TensorDescType> getDesc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::TensorDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::RankedTensorType> getXOffsets() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getYOffset() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getDescMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getXOffsetsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getYOffsetMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getResultMutable() {
    auto range = getODSOperandIndexAndLength(3);
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
  ::mlir::IntegerAttr getTxCountAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().txCount);
  }

  uint32_t getTxCount();
  void setTxCountAttr(::mlir::IntegerAttr attr) {
    getProperties().txCount = attr;
  }

  void setTxCount(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::IntegerAttr txCount, ::mlir::Value result);
  static DescriptorGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::IntegerAttr txCount, ::mlir::Value result);
  static DescriptorGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::IntegerAttr txCount, ::mlir::Value result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::IntegerAttr txCount, ::mlir::Value result);
  static DescriptorGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::IntegerAttr txCount, ::mlir::Value result);
  static DescriptorGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, ::mlir::IntegerAttr txCount, ::mlir::Value result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, uint32_t txCount, ::mlir::Value result);
  static DescriptorGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, uint32_t txCount, ::mlir::Value result);
  static DescriptorGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, uint32_t txCount, ::mlir::Value result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, uint32_t txCount, ::mlir::Value result);
  static DescriptorGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, uint32_t txCount, ::mlir::Value result);
  static DescriptorGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::Value x_offsets, ::mlir::Value y_offset, uint32_t txCount, ::mlir::Value result);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DescriptorGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DescriptorGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DescriptorGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DescriptorGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
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

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::DescriptorGatherOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::DescriptorLoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class DescriptorLoadOpGenericAdaptorBase {
public:
  struct Properties {
    using cacheTy = ::mlir::triton::CacheModifierAttr;
    cacheTy cache;

    auto getCache() const {
      auto &propStorage = this->cache;
      return ::llvm::dyn_cast_or_null<::mlir::triton::CacheModifierAttr>(propStorage);
    }
    void setCache(const ::mlir::triton::CacheModifierAttr &propValue) {
      this->cache = propValue;
    }
    using evictTy = ::mlir::triton::EvictionPolicyAttr;
    evictTy evict;

    auto getEvict() const {
      auto &propStorage = this->evict;
      return ::llvm::dyn_cast_or_null<::mlir::triton::EvictionPolicyAttr>(propStorage);
    }
    void setEvict(const ::mlir::triton::EvictionPolicyAttr &propValue) {
      this->evict = propValue;
    }
    using txCountTy = ::mlir::IntegerAttr;
    txCountTy txCount;

    auto getTxCount() const {
      auto &propStorage = this->txCount;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setTxCount(const ::mlir::IntegerAttr &propValue) {
      this->txCount = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.cache == this->cache &&
        rhs.evict == this->evict &&
        rhs.txCount == this->txCount &&
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
  DescriptorLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.descriptor_load", odsAttrs.getContext());
  }

  DescriptorLoadOpGenericAdaptorBase(DescriptorLoadOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getTxCountAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().txCount);
    return attr;
  }

  uint32_t getTxCount();
  ::mlir::triton::CacheModifierAttr getCacheAttr();
  ::mlir::triton::CacheModifier getCache();
  ::mlir::triton::EvictionPolicyAttr getEvictAttr();
  ::mlir::triton::EvictionPolicy getEvict();
};

} // namespace detail
template <typename RangeT>
class DescriptorLoadOpGenericAdaptor : public detail::DescriptorLoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DescriptorLoadOpGenericAdaptorBase;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_indices = 1;
  static constexpr int odsIndex_result = 2;
public:
  DescriptorLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DescriptorLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : DescriptorLoadOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  DescriptorLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : DescriptorLoadOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  DescriptorLoadOpGenericAdaptor(RangeT values, const DescriptorLoadOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = DescriptorLoadOp, typename = std::enable_if_t<std::is_same_v<LateInst, DescriptorLoadOp>>>
  DescriptorLoadOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDesc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  ValueT getResult() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DescriptorLoadOpAdaptor : public DescriptorLoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DescriptorLoadOpGenericAdaptor::DescriptorLoadOpGenericAdaptor;
  DescriptorLoadOpAdaptor(DescriptorLoadOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class DescriptorLoadOp : public ::mlir::Op<DescriptorLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::triton::DescriptorOpInterface::Trait, ::mlir::triton::nvws::DescriptorLoadOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DescriptorLoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DescriptorLoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_desc = 0;
  static constexpr int odsIndex_indices = 1;
  static constexpr int odsIndex_result = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cache"), ::llvm::StringRef("evict"), ::llvm::StringRef("txCount")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCacheAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCacheAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getEvictAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getEvictAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getTxCountAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getTxCountAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.descriptor_load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::TensorDescType> getDesc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::TensorDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::Operation::operand_range getIndices() {
    return getODSOperands(1);
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getDescMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::OpOperand &getResultMutable() {
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
  ::mlir::IntegerAttr getTxCountAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().txCount);
  }

  uint32_t getTxCount();
  ::mlir::triton::CacheModifierAttr getCacheAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::triton::CacheModifierAttr>(getProperties().cache);
  }

  ::mlir::triton::CacheModifier getCache();
  ::mlir::triton::EvictionPolicyAttr getEvictAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::triton::EvictionPolicyAttr>(getProperties().evict);
  }

  ::mlir::triton::EvictionPolicy getEvict();
  void setTxCountAttr(::mlir::IntegerAttr attr) {
    getProperties().txCount = attr;
  }

  void setTxCount(uint32_t attrValue);
  void setCacheAttr(::mlir::triton::CacheModifierAttr attr) {
    getProperties().cache = attr;
  }

  void setCache(::mlir::triton::CacheModifier attrValue);
  void setEvictAttr(::mlir::triton::EvictionPolicyAttr attr) {
    getProperties().evict = attr;
  }

  void setEvict(::mlir::triton::EvictionPolicy attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::ValueRange indices, ::mlir::IntegerAttr txCount, ::mlir::Value result, ::mlir::triton::CacheModifierAttr cache = nullptr, ::mlir::triton::EvictionPolicyAttr evict = nullptr);
  static DescriptorLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::ValueRange indices, ::mlir::IntegerAttr txCount, ::mlir::Value result, ::mlir::triton::CacheModifierAttr cache = nullptr, ::mlir::triton::EvictionPolicyAttr evict = nullptr);
  static DescriptorLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::ValueRange indices, ::mlir::IntegerAttr txCount, ::mlir::Value result, ::mlir::triton::CacheModifierAttr cache = nullptr, ::mlir::triton::EvictionPolicyAttr evict = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange indices, ::mlir::IntegerAttr txCount, ::mlir::Value result, ::mlir::triton::CacheModifierAttr cache = nullptr, ::mlir::triton::EvictionPolicyAttr evict = nullptr);
  static DescriptorLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange indices, ::mlir::IntegerAttr txCount, ::mlir::Value result, ::mlir::triton::CacheModifierAttr cache = nullptr, ::mlir::triton::EvictionPolicyAttr evict = nullptr);
  static DescriptorLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange indices, ::mlir::IntegerAttr txCount, ::mlir::Value result, ::mlir::triton::CacheModifierAttr cache = nullptr, ::mlir::triton::EvictionPolicyAttr evict = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value desc, ::mlir::ValueRange indices, uint32_t txCount, ::mlir::Value result, ::mlir::triton::CacheModifier cache = ::mlir::triton::CacheModifier::NONE, ::mlir::triton::EvictionPolicy evict = ::mlir::triton::EvictionPolicy::NORMAL);
  static DescriptorLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value desc, ::mlir::ValueRange indices, uint32_t txCount, ::mlir::Value result, ::mlir::triton::CacheModifier cache = ::mlir::triton::CacheModifier::NONE, ::mlir::triton::EvictionPolicy evict = ::mlir::triton::EvictionPolicy::NORMAL);
  static DescriptorLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value desc, ::mlir::ValueRange indices, uint32_t txCount, ::mlir::Value result, ::mlir::triton::CacheModifier cache = ::mlir::triton::CacheModifier::NONE, ::mlir::triton::EvictionPolicy evict = ::mlir::triton::EvictionPolicy::NORMAL);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange indices, uint32_t txCount, ::mlir::Value result, ::mlir::triton::CacheModifier cache = ::mlir::triton::CacheModifier::NONE, ::mlir::triton::EvictionPolicy evict = ::mlir::triton::EvictionPolicy::NORMAL);
  static DescriptorLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange indices, uint32_t txCount, ::mlir::Value result, ::mlir::triton::CacheModifier cache = ::mlir::triton::CacheModifier::NONE, ::mlir::triton::EvictionPolicy evict = ::mlir::triton::EvictionPolicy::NORMAL);
  static DescriptorLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value desc, ::mlir::ValueRange indices, uint32_t txCount, ::mlir::Value result, ::mlir::triton::CacheModifier cache = ::mlir::triton::CacheModifier::NONE, ::mlir::triton::EvictionPolicy evict = ::mlir::triton::EvictionPolicy::NORMAL);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DescriptorLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DescriptorLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DescriptorLoadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DescriptorLoadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void populateDefaultProperties(::mlir::OperationName opName, Properties &properties);
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

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::DescriptorLoadOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ProducerAcquireOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ProducerAcquireOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ProducerAcquireOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.producer_acquire", odsAttrs.getContext());
  }

  ProducerAcquireOpGenericAdaptorBase(ProducerAcquireOp op);

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
class ProducerAcquireOpGenericAdaptor : public detail::ProducerAcquireOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ProducerAcquireOpGenericAdaptorBase;
  static constexpr int odsIndex_token = 0;
  static constexpr int odsIndex_idx = 1;
  static constexpr int odsIndex_phase = 2;
public:
  ProducerAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ProducerAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ProducerAcquireOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ProducerAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ProducerAcquireOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ProducerAcquireOpGenericAdaptor(RangeT values, const ProducerAcquireOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ProducerAcquireOp, typename = std::enable_if_t<std::is_same_v<LateInst, ProducerAcquireOp>>>
  ProducerAcquireOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  ValueT getIdx() {
    return (*getODSOperands(1).begin());
  }

  ValueT getPhase() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ProducerAcquireOpAdaptor : public ProducerAcquireOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ProducerAcquireOpGenericAdaptor::ProducerAcquireOpGenericAdaptor;
  ProducerAcquireOpAdaptor(ProducerAcquireOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ProducerAcquireOp : public ::mlir::Op<ProducerAcquireOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ProducerAcquireOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ProducerAcquireOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_token = 0;
  static constexpr int odsIndex_idx = 1;
  static constexpr int odsIndex_phase = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.producer_acquire");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getIdx() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPhase() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getTokenMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getIdxMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getPhaseMutable() {
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static ProducerAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static ProducerAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static ProducerAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static ProducerAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx, ::mlir::Value phase);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ProducerAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ProducerAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ProducerAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ProducerAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ProducerAcquireOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::ProducerCommitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ProducerCommitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ProducerCommitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.producer_commit", odsAttrs.getContext());
  }

  ProducerCommitOpGenericAdaptorBase(ProducerCommitOp op);

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
class ProducerCommitOpGenericAdaptor : public detail::ProducerCommitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ProducerCommitOpGenericAdaptorBase;
  static constexpr int odsIndex_token = 0;
  static constexpr int odsIndex_idx = 1;
public:
  ProducerCommitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ProducerCommitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ProducerCommitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ProducerCommitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ProducerCommitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ProducerCommitOpGenericAdaptor(RangeT values, const ProducerCommitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ProducerCommitOp, typename = std::enable_if_t<std::is_same_v<LateInst, ProducerCommitOp>>>
  ProducerCommitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getToken() {
    return (*getODSOperands(0).begin());
  }

  ValueT getIdx() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ProducerCommitOpAdaptor : public ProducerCommitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ProducerCommitOpGenericAdaptor::ProducerCommitOpGenericAdaptor;
  ProducerCommitOpAdaptor(ProducerCommitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ProducerCommitOp : public ::mlir::Op<ProducerCommitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ProducerCommitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ProducerCommitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_token = 0;
  static constexpr int odsIndex_idx = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.producer_commit");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::TensorType> getToken() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getIdx() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getTokenMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getIdxMutable() {
    auto range = getODSOperandIndexAndLength(1);
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::Value idx);
  static ProducerCommitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value token, ::mlir::Value idx);
  static ProducerCommitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value token, ::mlir::Value idx);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx);
  static ProducerCommitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx);
  static ProducerCommitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value idx);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ProducerCommitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ProducerCommitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ProducerCommitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ProducerCommitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ProducerCommitOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::WarpGroupOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class WarpGroupOpGenericAdaptorBase {
public:
  struct Properties {
    using numWarpsTy = ::mlir::DenseI32ArrayAttr;
    numWarpsTy numWarps;

    auto getNumWarps() const {
      auto &propStorage = this->numWarps;
      return ::llvm::cast<::mlir::DenseI32ArrayAttr>(propStorage);
    }
    void setNumWarps(const ::mlir::DenseI32ArrayAttr &propValue) {
      this->numWarps = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.numWarps == this->numWarps &&
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
  WarpGroupOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.warp_group", odsAttrs.getContext());
  }

  WarpGroupOpGenericAdaptorBase(WarpGroupOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::DenseI32ArrayAttr getNumWarpsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().numWarps);
    return attr;
  }

  ::llvm::ArrayRef<int32_t> getNumWarps();
  ::mlir::RegionRange getPartitionRegions() {
    return odsRegions.drop_front(0);
  }

  ::mlir::RegionRange getRegions() {
    return odsRegions;
  }

};

} // namespace detail
template <typename RangeT>
class WarpGroupOpGenericAdaptor : public detail::WarpGroupOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WarpGroupOpGenericAdaptorBase;
public:
  WarpGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  WarpGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : WarpGroupOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  WarpGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : WarpGroupOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  WarpGroupOpGenericAdaptor(RangeT values, const WarpGroupOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = WarpGroupOp, typename = std::enable_if_t<std::is_same_v<LateInst, WarpGroupOp>>>
  WarpGroupOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class WarpGroupOpAdaptor : public WarpGroupOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WarpGroupOpGenericAdaptor::WarpGroupOpGenericAdaptor;
  WarpGroupOpAdaptor(WarpGroupOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class WarpGroupOp : public ::mlir::Op<WarpGroupOp, ::mlir::OpTrait::VariadicRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::RecursivelySpeculatableImplTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WarpGroupOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WarpGroupOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("numWarps")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNumWarpsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNumWarpsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.warp_group");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::result_range getResults() {
    return getODSResults(0);
  }

  ::mlir::MutableArrayRef<::mlir::Region> getPartitionRegions() {
    return (*this)->getRegions().drop_front(0);
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
  ::mlir::DenseI32ArrayAttr getNumWarpsAttr() {
    return ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().numWarps);
  }

  ::llvm::ArrayRef<int32_t> getNumWarps();
  void setNumWarpsAttr(::mlir::DenseI32ArrayAttr attr) {
    getProperties().numWarps = attr;
  }

  void setNumWarps(::llvm::ArrayRef<int32_t> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::DenseI32ArrayAttr numWarps, unsigned partitionRegionsCount);
  static WarpGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange results, ::mlir::DenseI32ArrayAttr numWarps, unsigned partitionRegionsCount);
  static WarpGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange results, ::mlir::DenseI32ArrayAttr numWarps, unsigned partitionRegionsCount);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::ArrayRef<int32_t> numWarps, unsigned partitionRegionsCount);
  static WarpGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange results, ::llvm::ArrayRef<int32_t> numWarps, unsigned partitionRegionsCount);
  static WarpGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange results, ::llvm::ArrayRef<int32_t> numWarps, unsigned partitionRegionsCount);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  static WarpGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  static WarpGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes, unsigned numRegions);
  static WarpGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes, unsigned numRegions);
  static WarpGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes, unsigned numRegions);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::llvm::LogicalResult verify();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
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

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::WarpGroupOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::WarpGroupReturnOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class WarpGroupReturnOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  WarpGroupReturnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.warp_group.return", odsAttrs.getContext());
  }

  WarpGroupReturnOpGenericAdaptorBase(WarpGroupReturnOp op);

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
class WarpGroupReturnOpGenericAdaptor : public detail::WarpGroupReturnOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WarpGroupReturnOpGenericAdaptorBase;
public:
  WarpGroupReturnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  WarpGroupReturnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : WarpGroupReturnOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  WarpGroupReturnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : WarpGroupReturnOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  WarpGroupReturnOpGenericAdaptor(RangeT values, const WarpGroupReturnOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = WarpGroupReturnOp, typename = std::enable_if_t<std::is_same_v<LateInst, WarpGroupReturnOp>>>
  WarpGroupReturnOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class WarpGroupReturnOpAdaptor : public WarpGroupReturnOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WarpGroupReturnOpGenericAdaptor::WarpGroupReturnOpGenericAdaptor;
  WarpGroupReturnOpAdaptor(WarpGroupReturnOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class WarpGroupReturnOp : public ::mlir::Op<WarpGroupReturnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<WarpGroupOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WarpGroupReturnOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WarpGroupReturnOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.warp_group.return");
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static WarpGroupReturnOp create(::mlir::OpBuilder &builder, ::mlir::Location location);
  static WarpGroupReturnOp create(::mlir::ImplicitLocOpBuilder &builder);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static WarpGroupReturnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static WarpGroupReturnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WarpGroupReturnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WarpGroupReturnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WarpGroupReturnOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WarpGroupReturnOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::WarpGroupReturnOp)

namespace mlir::triton::nvws {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvws::WarpGroupYieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class WarpGroupYieldOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  WarpGroupYieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvws.warp_group.yield", odsAttrs.getContext());
  }

  WarpGroupYieldOpGenericAdaptorBase(WarpGroupYieldOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class WarpGroupYieldOpGenericAdaptor : public detail::WarpGroupYieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WarpGroupYieldOpGenericAdaptorBase;
  static constexpr int odsIndex_values = 0;
public:
  WarpGroupYieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  WarpGroupYieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : WarpGroupYieldOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  WarpGroupYieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : WarpGroupYieldOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  WarpGroupYieldOpGenericAdaptor(RangeT values, const WarpGroupYieldOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = WarpGroupYieldOp, typename = std::enable_if_t<std::is_same_v<LateInst, WarpGroupYieldOp>>>
  WarpGroupYieldOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getValues() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WarpGroupYieldOpAdaptor : public WarpGroupYieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WarpGroupYieldOpGenericAdaptor::WarpGroupYieldOpGenericAdaptor;
  WarpGroupYieldOpAdaptor(WarpGroupYieldOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class WarpGroupYieldOp : public ::mlir::Op<WarpGroupYieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<WarpGroupOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator, ::mlir::RegionBranchTerminatorOpInterface::Trait, ::mlir::OpTrait::ReturnLike> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WarpGroupYieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WarpGroupYieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_values = 0;
  using Properties = FoldAdaptor::Properties;
  using ::mlir::RegionBranchTerminatorOpInterface::Trait<WarpGroupYieldOp>::getSuccessorRegions;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvws.warp_group.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::Operation::operand_range getValues() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getValuesMutable();
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange values);
  static WarpGroupYieldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange values);
  static WarpGroupYieldOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange values);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WarpGroupYieldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WarpGroupYieldOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WarpGroupYieldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WarpGroupYieldOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::mlir::RegionSuccessor point);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::WarpGroupYieldOp)


#endif // GET_OP_CLASSES

