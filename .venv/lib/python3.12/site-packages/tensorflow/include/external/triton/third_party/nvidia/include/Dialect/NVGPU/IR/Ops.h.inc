/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVGPUOps.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvgpu {

class ClusterCTAIdOp;

} // namespace mlir::triton::nvgpu
namespace mlir::triton::nvgpu {

class LoadAcquireOp;

} // namespace mlir::triton::nvgpu
namespace mlir::triton::nvgpu {

/// Op to represent base address of tensor memory in a kernel.
///     This is used to simplify lowering from TritonGPU to LLVM.
class TensorMemoryBaseAddress;

} // namespace mlir::triton::nvgpu
namespace mlir::triton::nvgpu {

class WGMMAOp;

} // namespace mlir::triton::nvgpu
namespace mlir::triton::nvgpu {

class WGMMAWaitGroupOp;

} // namespace mlir::triton::nvgpu
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::ClusterCTAIdOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ClusterCTAIdOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ClusterCTAIdOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvg.cluster_id", odsAttrs.getContext());
  }

  ClusterCTAIdOpGenericAdaptorBase(ClusterCTAIdOp op);

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
class ClusterCTAIdOpGenericAdaptor : public detail::ClusterCTAIdOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ClusterCTAIdOpGenericAdaptorBase;
public:
  ClusterCTAIdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ClusterCTAIdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ClusterCTAIdOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ClusterCTAIdOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ClusterCTAIdOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ClusterCTAIdOpGenericAdaptor(RangeT values, const ClusterCTAIdOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ClusterCTAIdOp, typename = std::enable_if_t<std::is_same_v<LateInst, ClusterCTAIdOp>>>
  ClusterCTAIdOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class ClusterCTAIdOpAdaptor : public ClusterCTAIdOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ClusterCTAIdOpGenericAdaptor::ClusterCTAIdOpGenericAdaptor;
  ClusterCTAIdOpAdaptor(ClusterCTAIdOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ClusterCTAIdOp : public ::mlir::Op<ClusterCTAIdOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ClusterCTAIdOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ClusterCTAIdOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvg.cluster_id");
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
  static ClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result);
  static ClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static ClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location);
  static ClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static ClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static ClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ClusterCTAIdOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ClusterCTAIdOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::nvgpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::ClusterCTAIdOp)

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::LoadAcquireOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoadAcquireOpGenericAdaptorBase {
public:
  struct Properties {
    using scopeTy = ::mlir::triton::nvgpu::MemSyncScopeAttr;
    scopeTy scope;

    auto getScope() const {
      auto &propStorage = this->scope;
      return ::llvm::cast<::mlir::triton::nvgpu::MemSyncScopeAttr>(propStorage);
    }
    void setScope(const ::mlir::triton::nvgpu::MemSyncScopeAttr &propValue) {
      this->scope = propValue;
    }
    using semTy = ::mlir::triton::nvgpu::MemSemanticAttr;
    semTy sem;

    auto getSem() const {
      auto &propStorage = this->sem;
      return ::llvm::cast<::mlir::triton::nvgpu::MemSemanticAttr>(propStorage);
    }
    void setSem(const ::mlir::triton::nvgpu::MemSemanticAttr &propValue) {
      this->sem = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.scope == this->scope &&
        rhs.sem == this->sem &&
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
  LoadAcquireOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvg.ld_acquire", odsAttrs.getContext());
  }

  LoadAcquireOpGenericAdaptorBase(LoadAcquireOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::triton::nvgpu::MemSemanticAttr getSemAttr() {
    auto attr = ::llvm::cast<::mlir::triton::nvgpu::MemSemanticAttr>(getProperties().sem);
    return attr;
  }

  ::mlir::triton::nvgpu::MemSemantic getSem();
  ::mlir::triton::nvgpu::MemSyncScopeAttr getScopeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::nvgpu::MemSyncScopeAttr>(getProperties().scope);
    return attr;
  }

  ::mlir::triton::nvgpu::MemSyncScope getScope();
};

} // namespace detail
template <typename RangeT>
class LoadAcquireOpGenericAdaptor : public detail::LoadAcquireOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoadAcquireOpGenericAdaptorBase;
  static constexpr int odsIndex_addr = 0;
  static constexpr int odsIndex_mask = 1;
public:
  LoadAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoadAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : LoadAcquireOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoadAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoadAcquireOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoadAcquireOpGenericAdaptor(RangeT values, const LoadAcquireOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoadAcquireOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoadAcquireOp>>>
  LoadAcquireOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getAddr() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMask() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoadAcquireOpAdaptor : public LoadAcquireOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoadAcquireOpGenericAdaptor::LoadAcquireOpGenericAdaptor;
  LoadAcquireOpAdaptor(LoadAcquireOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoadAcquireOp : public ::mlir::Op<LoadAcquireOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadAcquireOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoadAcquireOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_addr = 0;
  static constexpr int odsIndex_mask = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("scope"), ::llvm::StringRef("sem")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getScopeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getScopeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSemAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSemAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvg.ld_acquire");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> getAddr() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getMask() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getAddrMutable() {
    auto range = getODSOperandIndexAndLength(0);
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
  ::mlir::triton::nvgpu::MemSemanticAttr getSemAttr() {
    return ::llvm::cast<::mlir::triton::nvgpu::MemSemanticAttr>(getProperties().sem);
  }

  ::mlir::triton::nvgpu::MemSemantic getSem();
  ::mlir::triton::nvgpu::MemSyncScopeAttr getScopeAttr() {
    return ::llvm::cast<::mlir::triton::nvgpu::MemSyncScopeAttr>(getProperties().scope);
  }

  ::mlir::triton::nvgpu::MemSyncScope getScope();
  void setSemAttr(::mlir::triton::nvgpu::MemSemanticAttr attr) {
    getProperties().sem = attr;
  }

  void setSem(::mlir::triton::nvgpu::MemSemantic attrValue);
  void setScopeAttr(::mlir::triton::nvgpu::MemSyncScopeAttr attr) {
    getProperties().scope = attr;
  }

  void setScope(::mlir::triton::nvgpu::MemSyncScope attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope);
  static LoadAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope);
  static LoadAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope);
  static LoadAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope);
  static LoadAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope);
  static LoadAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope);
  static LoadAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope);
  static LoadAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope);
  static LoadAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoadAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoadAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoadAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoadAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
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

} // namespace mlir::triton::nvgpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::LoadAcquireOp)

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::TensorMemoryBaseAddress declarations
//===----------------------------------------------------------------------===//

namespace detail {

class TensorMemoryBaseAddressGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  TensorMemoryBaseAddressGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvg.tensor_memory_base", odsAttrs.getContext());
  }

  TensorMemoryBaseAddressGenericAdaptorBase(TensorMemoryBaseAddress op);

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
class TensorMemoryBaseAddressGenericAdaptor : public detail::TensorMemoryBaseAddressGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TensorMemoryBaseAddressGenericAdaptorBase;
public:
  TensorMemoryBaseAddressGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TensorMemoryBaseAddressGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TensorMemoryBaseAddressGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  TensorMemoryBaseAddressGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : TensorMemoryBaseAddressGenericAdaptor(values, attrs, Properties{}, {}) {}

  TensorMemoryBaseAddressGenericAdaptor(RangeT values, const TensorMemoryBaseAddressGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = TensorMemoryBaseAddress, typename = std::enable_if_t<std::is_same_v<LateInst, TensorMemoryBaseAddress>>>
  TensorMemoryBaseAddressGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class TensorMemoryBaseAddressAdaptor : public TensorMemoryBaseAddressGenericAdaptor<::mlir::ValueRange> {
public:
  using TensorMemoryBaseAddressGenericAdaptor::TensorMemoryBaseAddressGenericAdaptor;
  TensorMemoryBaseAddressAdaptor(TensorMemoryBaseAddress op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class TensorMemoryBaseAddress : public ::mlir::Op<TensorMemoryBaseAddress, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::LLVM::LLVMPointerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TensorMemoryBaseAddressAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TensorMemoryBaseAddressGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvg.tensor_memory_base");
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

  ::mlir::TypedValue<::mlir::LLVM::LLVMPointerType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::LLVM::LLVMPointerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result);
  static TensorMemoryBaseAddress create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result);
  static TensorMemoryBaseAddress create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static TensorMemoryBaseAddress create(::mlir::OpBuilder &builder, ::mlir::Location location);
  static TensorMemoryBaseAddress create(::mlir::ImplicitLocOpBuilder &builder);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static TensorMemoryBaseAddress create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static TensorMemoryBaseAddress create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TensorMemoryBaseAddress create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TensorMemoryBaseAddress create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TensorMemoryBaseAddress create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static TensorMemoryBaseAddress create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TensorMemoryBaseAddress create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TensorMemoryBaseAddress create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TensorMemoryBaseAddress create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static TensorMemoryBaseAddress create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::nvgpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::TensorMemoryBaseAddress)

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::WGMMAOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class WGMMAOpGenericAdaptorBase {
public:
  struct Properties {
    using eltTypeATy = ::mlir::triton::nvgpu::WGMMAEltTypeAttr;
    eltTypeATy eltTypeA;

    auto getEltTypeA() const {
      auto &propStorage = this->eltTypeA;
      return ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(propStorage);
    }
    void setEltTypeA(const ::mlir::triton::nvgpu::WGMMAEltTypeAttr &propValue) {
      this->eltTypeA = propValue;
    }
    using eltTypeBTy = ::mlir::triton::nvgpu::WGMMAEltTypeAttr;
    eltTypeBTy eltTypeB;

    auto getEltTypeB() const {
      auto &propStorage = this->eltTypeB;
      return ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(propStorage);
    }
    void setEltTypeB(const ::mlir::triton::nvgpu::WGMMAEltTypeAttr &propValue) {
      this->eltTypeB = propValue;
    }
    using eltTypeCTy = ::mlir::triton::nvgpu::WGMMAEltTypeAttr;
    eltTypeCTy eltTypeC;

    auto getEltTypeC() const {
      auto &propStorage = this->eltTypeC;
      return ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(propStorage);
    }
    void setEltTypeC(const ::mlir::triton::nvgpu::WGMMAEltTypeAttr &propValue) {
      this->eltTypeC = propValue;
    }
    using kTy = ::mlir::IntegerAttr;
    kTy k;

    auto getK() const {
      auto &propStorage = this->k;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setK(const ::mlir::IntegerAttr &propValue) {
      this->k = propValue;
    }
    using layoutATy = ::mlir::triton::nvgpu::WGMMALayoutAttr;
    layoutATy layoutA;

    auto getLayoutA() const {
      auto &propStorage = this->layoutA;
      return ::llvm::cast<::mlir::triton::nvgpu::WGMMALayoutAttr>(propStorage);
    }
    void setLayoutA(const ::mlir::triton::nvgpu::WGMMALayoutAttr &propValue) {
      this->layoutA = propValue;
    }
    using layoutBTy = ::mlir::triton::nvgpu::WGMMALayoutAttr;
    layoutBTy layoutB;

    auto getLayoutB() const {
      auto &propStorage = this->layoutB;
      return ::llvm::cast<::mlir::triton::nvgpu::WGMMALayoutAttr>(propStorage);
    }
    void setLayoutB(const ::mlir::triton::nvgpu::WGMMALayoutAttr &propValue) {
      this->layoutB = propValue;
    }
    using mTy = ::mlir::IntegerAttr;
    mTy m;

    auto getM() const {
      auto &propStorage = this->m;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setM(const ::mlir::IntegerAttr &propValue) {
      this->m = propValue;
    }
    using nTy = ::mlir::IntegerAttr;
    nTy n;

    auto getN() const {
      auto &propStorage = this->n;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setN(const ::mlir::IntegerAttr &propValue) {
      this->n = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.eltTypeA == this->eltTypeA &&
        rhs.eltTypeB == this->eltTypeB &&
        rhs.eltTypeC == this->eltTypeC &&
        rhs.k == this->k &&
        rhs.layoutA == this->layoutA &&
        rhs.layoutB == this->layoutB &&
        rhs.m == this->m &&
        rhs.n == this->n &&
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
  WGMMAOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvg.wgmma", odsAttrs.getContext());
  }

  WGMMAOpGenericAdaptorBase(WGMMAOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getMAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().m);
    return attr;
  }

  uint32_t getM();
  ::mlir::IntegerAttr getNAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().n);
    return attr;
  }

  uint32_t getN();
  ::mlir::IntegerAttr getKAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().k);
    return attr;
  }

  uint32_t getK();
  ::mlir::triton::nvgpu::WGMMAEltTypeAttr getEltTypeCAttr() {
    auto attr = ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(getProperties().eltTypeC);
    return attr;
  }

  ::mlir::triton::nvgpu::WGMMAEltType getEltTypeC();
  ::mlir::triton::nvgpu::WGMMAEltTypeAttr getEltTypeAAttr() {
    auto attr = ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(getProperties().eltTypeA);
    return attr;
  }

  ::mlir::triton::nvgpu::WGMMAEltType getEltTypeA();
  ::mlir::triton::nvgpu::WGMMAEltTypeAttr getEltTypeBAttr() {
    auto attr = ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(getProperties().eltTypeB);
    return attr;
  }

  ::mlir::triton::nvgpu::WGMMAEltType getEltTypeB();
  ::mlir::triton::nvgpu::WGMMALayoutAttr getLayoutAAttr() {
    auto attr = ::llvm::cast<::mlir::triton::nvgpu::WGMMALayoutAttr>(getProperties().layoutA);
    return attr;
  }

  ::mlir::triton::nvgpu::WGMMALayout getLayoutA();
  ::mlir::triton::nvgpu::WGMMALayoutAttr getLayoutBAttr() {
    auto attr = ::llvm::cast<::mlir::triton::nvgpu::WGMMALayoutAttr>(getProperties().layoutB);
    return attr;
  }

  ::mlir::triton::nvgpu::WGMMALayout getLayoutB();
};

} // namespace detail
template <typename RangeT>
class WGMMAOpGenericAdaptor : public detail::WGMMAOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WGMMAOpGenericAdaptorBase;
  static constexpr int odsIndex_opA = 0;
  static constexpr int odsIndex_opB = 1;
  static constexpr int odsIndex_useC = 2;
  static constexpr int odsIndex_opC = 3;
public:
  WGMMAOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  WGMMAOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : WGMMAOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  WGMMAOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : WGMMAOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  WGMMAOpGenericAdaptor(RangeT values, const WGMMAOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = WGMMAOp, typename = std::enable_if_t<std::is_same_v<LateInst, WGMMAOp>>>
  WGMMAOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getOpA() {
    return (*getODSOperands(0).begin());
  }

  ValueT getOpB() {
    return (*getODSOperands(1).begin());
  }

  ValueT getUseC() {
    return (*getODSOperands(2).begin());
  }

  ValueT getOpC() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WGMMAOpAdaptor : public WGMMAOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WGMMAOpGenericAdaptor::WGMMAOpGenericAdaptor;
  WGMMAOpAdaptor(WGMMAOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class WGMMAOp : public ::mlir::Op<WGMMAOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WGMMAOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WGMMAOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_opA = 0;
  static constexpr int odsIndex_opB = 1;
  static constexpr int odsIndex_useC = 2;
  static constexpr int odsIndex_opC = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("eltTypeA"), ::llvm::StringRef("eltTypeB"), ::llvm::StringRef("eltTypeC"), ::llvm::StringRef("k"), ::llvm::StringRef("layoutA"), ::llvm::StringRef("layoutB"), ::llvm::StringRef("m"), ::llvm::StringRef("n")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getEltTypeAAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getEltTypeAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getEltTypeBAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getEltTypeBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getEltTypeCAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getEltTypeCAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getKAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getKAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getLayoutAAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getLayoutAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getLayoutBAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getLayoutBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getMAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getMAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getNAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getNAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvg.wgmma");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getOpA() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getOpB() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getUseC() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getOpC() {
    auto operands = getODSOperands(3);
    return operands.empty() ? ::mlir::TypedValue<::mlir::Type>{} : ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*operands.begin());
  }

  ::mlir::OpOperand &getOpAMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getOpBMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getUseCMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getOpCMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getRes() {
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
  ::mlir::IntegerAttr getMAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().m);
  }

  uint32_t getM();
  ::mlir::IntegerAttr getNAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().n);
  }

  uint32_t getN();
  ::mlir::IntegerAttr getKAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().k);
  }

  uint32_t getK();
  ::mlir::triton::nvgpu::WGMMAEltTypeAttr getEltTypeCAttr() {
    return ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(getProperties().eltTypeC);
  }

  ::mlir::triton::nvgpu::WGMMAEltType getEltTypeC();
  ::mlir::triton::nvgpu::WGMMAEltTypeAttr getEltTypeAAttr() {
    return ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(getProperties().eltTypeA);
  }

  ::mlir::triton::nvgpu::WGMMAEltType getEltTypeA();
  ::mlir::triton::nvgpu::WGMMAEltTypeAttr getEltTypeBAttr() {
    return ::llvm::cast<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(getProperties().eltTypeB);
  }

  ::mlir::triton::nvgpu::WGMMAEltType getEltTypeB();
  ::mlir::triton::nvgpu::WGMMALayoutAttr getLayoutAAttr() {
    return ::llvm::cast<::mlir::triton::nvgpu::WGMMALayoutAttr>(getProperties().layoutA);
  }

  ::mlir::triton::nvgpu::WGMMALayout getLayoutA();
  ::mlir::triton::nvgpu::WGMMALayoutAttr getLayoutBAttr() {
    return ::llvm::cast<::mlir::triton::nvgpu::WGMMALayoutAttr>(getProperties().layoutB);
  }

  ::mlir::triton::nvgpu::WGMMALayout getLayoutB();
  void setMAttr(::mlir::IntegerAttr attr) {
    getProperties().m = attr;
  }

  void setM(uint32_t attrValue);
  void setNAttr(::mlir::IntegerAttr attr) {
    getProperties().n = attr;
  }

  void setN(uint32_t attrValue);
  void setKAttr(::mlir::IntegerAttr attr) {
    getProperties().k = attr;
  }

  void setK(uint32_t attrValue);
  void setEltTypeCAttr(::mlir::triton::nvgpu::WGMMAEltTypeAttr attr) {
    getProperties().eltTypeC = attr;
  }

  void setEltTypeC(::mlir::triton::nvgpu::WGMMAEltType attrValue);
  void setEltTypeAAttr(::mlir::triton::nvgpu::WGMMAEltTypeAttr attr) {
    getProperties().eltTypeA = attr;
  }

  void setEltTypeA(::mlir::triton::nvgpu::WGMMAEltType attrValue);
  void setEltTypeBAttr(::mlir::triton::nvgpu::WGMMAEltTypeAttr attr) {
    getProperties().eltTypeB = attr;
  }

  void setEltTypeB(::mlir::triton::nvgpu::WGMMAEltType attrValue);
  void setLayoutAAttr(::mlir::triton::nvgpu::WGMMALayoutAttr attr) {
    getProperties().layoutA = attr;
  }

  void setLayoutA(::mlir::triton::nvgpu::WGMMALayout attrValue);
  void setLayoutBAttr(::mlir::triton::nvgpu::WGMMALayoutAttr attr) {
    getProperties().layoutB = attr;
  }

  void setLayoutB(::mlir::triton::nvgpu::WGMMALayout attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB);
  static WGMMAOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB);
  static WGMMAOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB);
  static WGMMAOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB);
  static WGMMAOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB);
  static WGMMAOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB);
  static WGMMAOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB);
  static WGMMAOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB);
  static WGMMAOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WGMMAOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WGMMAOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WGMMAOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WGMMAOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 8 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::triton::nvgpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::WGMMAOp)

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::WGMMAWaitGroupOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class WGMMAWaitGroupOpGenericAdaptorBase {
public:
  struct Properties {
    using pendingsTy = ::mlir::IntegerAttr;
    pendingsTy pendings;

    auto getPendings() const {
      auto &propStorage = this->pendings;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setPendings(const ::mlir::IntegerAttr &propValue) {
      this->pendings = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.pendings == this->pendings &&
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
  WGMMAWaitGroupOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("nvg.wgmma_wait_group", odsAttrs.getContext());
  }

  WGMMAWaitGroupOpGenericAdaptorBase(WGMMAWaitGroupOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getPendingsAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().pendings);
    return attr;
  }

  uint32_t getPendings();
};

} // namespace detail
template <typename RangeT>
class WGMMAWaitGroupOpGenericAdaptor : public detail::WGMMAWaitGroupOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WGMMAWaitGroupOpGenericAdaptorBase;
  static constexpr int odsIndex_input = 0;
public:
  WGMMAWaitGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  WGMMAWaitGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : WGMMAWaitGroupOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  WGMMAWaitGroupOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : WGMMAWaitGroupOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  WGMMAWaitGroupOpGenericAdaptor(RangeT values, const WGMMAWaitGroupOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = WGMMAWaitGroupOp, typename = std::enable_if_t<std::is_same_v<LateInst, WGMMAWaitGroupOp>>>
  WGMMAWaitGroupOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WGMMAWaitGroupOpAdaptor : public WGMMAWaitGroupOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WGMMAWaitGroupOpGenericAdaptor::WGMMAWaitGroupOpGenericAdaptor;
  WGMMAWaitGroupOpAdaptor(WGMMAWaitGroupOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class WGMMAWaitGroupOp : public ::mlir::Op<WGMMAWaitGroupOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WGMMAWaitGroupOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WGMMAWaitGroupOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_input = 0;
  using Properties = FoldAdaptor::Properties;
  using ::mlir::InferTypeOpInterface::Trait<WGMMAWaitGroupOp>::refineReturnTypes;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("pendings")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getPendingsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getPendingsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvg.wgmma_wait_group");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
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

  ::mlir::TypedValue<::mlir::Type> getOutput() {
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
  ::mlir::IntegerAttr getPendingsAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().pendings);
  }

  uint32_t getPendings();
  void setPendingsAttr(::mlir::IntegerAttr attr) {
    getProperties().pendings = attr;
  }

  void setPendings(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint32_t pendings);
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type output, ::mlir::Value input, uint32_t pendings);
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type output, ::mlir::Value input, uint32_t pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t pendings);
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t pendings);
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t pendings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t pendings);
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t pendings);
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t pendings);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WGMMAWaitGroupOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static WGMMAWaitGroupOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
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

} // namespace mlir::triton::nvgpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::WGMMAWaitGroupOp)


#endif // GET_OP_CLASSES

