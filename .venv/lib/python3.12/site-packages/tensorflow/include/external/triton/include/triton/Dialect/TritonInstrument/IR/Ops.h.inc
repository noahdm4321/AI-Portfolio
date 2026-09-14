/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonInstrumentOps.td                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::instrument {

/// assert the condition within the current thread
/// Assert that the condition is true given all the values are available in the current thread.
///     If the condition is false, the message is printed, and the program is aborted.
///     If check_any is true, any of the values in the condition must be true. Otherwise, all the
///     values in the condition must be true.
class ExperimentalAssertInThreadOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// define an array of buffer descriptors
/// Create a tensor of buffer descriptors packing 32-bit pointer offsets and
///     32-bit lengths into 64-bit elements.
class ExperimentalBufferDescriptorsOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Acquire a lock.
/// Enter a critical section by acquiring a lock with single thread.
class ExperimentalLockAcquireOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Release a lock.
/// Leave a critical section by releasing a lock with single thread.
class ExperimentalLockReleaseOp;

} // namespace mlir::triton::instrument
namespace mlir::triton::instrument {

/// Convert a memdesc into its base pointer as i32
/// Extract the base pointer from the given memdesc and return it as a 32-bit
///     integer. This can be used to compare the memdesc against tensors of barrier
///     pointers maintained by the concurrency sanitizer.
class ExperimentalMemDescToI32Op;

} // namespace mlir::triton::instrument
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalAssertInThreadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalAssertInThreadOpGenericAdaptorBase {
public:
  struct Properties {
    using check_anyTy = ::mlir::BoolAttr;
    check_anyTy check_any;

    auto getCheckAny() const {
      auto &propStorage = this->check_any;
      return ::llvm::cast<::mlir::BoolAttr>(propStorage);
    }
    void setCheckAny(const ::mlir::BoolAttr &propValue) {
      this->check_any = propValue;
    }
    using messageTy = ::mlir::StringAttr;
    messageTy message;

    auto getMessage() const {
      auto &propStorage = this->message;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setMessage(const ::mlir::StringAttr &propValue) {
      this->message = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.check_any == this->check_any &&
        rhs.message == this->message &&
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
  ExperimentalAssertInThreadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_assert_in_thread", odsAttrs.getContext());
  }

  ExperimentalAssertInThreadOpGenericAdaptorBase(ExperimentalAssertInThreadOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getMessageAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().message);
    return attr;
  }

  ::llvm::StringRef getMessage();
  ::mlir::BoolAttr getCheckAnyAttr() {
    auto attr = ::llvm::cast<::mlir::BoolAttr>(getProperties().check_any);
    return attr;
  }

  bool getCheckAny();
};

} // namespace detail
template <typename RangeT>
class ExperimentalAssertInThreadOpGenericAdaptor : public detail::ExperimentalAssertInThreadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalAssertInThreadOpGenericAdaptorBase;
  static constexpr int odsIndex_condition = 0;
public:
  ExperimentalAssertInThreadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalAssertInThreadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ExperimentalAssertInThreadOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalAssertInThreadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalAssertInThreadOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalAssertInThreadOpGenericAdaptor(RangeT values, const ExperimentalAssertInThreadOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalAssertInThreadOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalAssertInThreadOp>>>
  ExperimentalAssertInThreadOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCondition() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalAssertInThreadOpAdaptor : public ExperimentalAssertInThreadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalAssertInThreadOpGenericAdaptor::ExperimentalAssertInThreadOpGenericAdaptor;
  ExperimentalAssertInThreadOpAdaptor(ExperimentalAssertInThreadOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalAssertInThreadOp : public ::mlir::Op<ExperimentalAssertInThreadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalAssertInThreadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalAssertInThreadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_condition = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("check_any"), ::llvm::StringRef("message")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCheckAnyAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCheckAnyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMessageAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMessageAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_assert_in_thread");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getCondition() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getConditionMutable() {
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getMessageAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().message);
  }

  ::llvm::StringRef getMessage();
  ::mlir::BoolAttr getCheckAnyAttr() {
    return ::llvm::cast<::mlir::BoolAttr>(getProperties().check_any);
  }

  bool getCheckAny();
  void setMessageAttr(::mlir::StringAttr attr) {
    getProperties().message = attr;
  }

  void setMessage(::llvm::StringRef attrValue);
  void setCheckAnyAttr(::mlir::BoolAttr attr) {
    getProperties().check_any = attr;
  }

  void setCheckAny(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any);
  static ExperimentalAssertInThreadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any);
  static ExperimentalAssertInThreadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any);
  static ExperimentalAssertInThreadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any);
  static ExperimentalAssertInThreadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::llvm::StringRef message, bool check_any);
  static ExperimentalAssertInThreadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value condition, ::llvm::StringRef message, bool check_any);
  static ExperimentalAssertInThreadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value condition, ::llvm::StringRef message, bool check_any);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message, bool check_any);
  static ExperimentalAssertInThreadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message, bool check_any);
  static ExperimentalAssertInThreadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message, bool check_any);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalAssertInThreadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalAssertInThreadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalAssertInThreadOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalAssertInThreadOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
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

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalAssertInThreadOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalBufferDescriptorsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalBufferDescriptorsOpGenericAdaptorBase {
public:
  struct Properties {
    using lengthsTy = ::mlir::DenseI32ArrayAttr;
    lengthsTy lengths;

    auto getLengths() const {
      auto &propStorage = this->lengths;
      return ::llvm::cast<::mlir::DenseI32ArrayAttr>(propStorage);
    }
    void setLengths(const ::mlir::DenseI32ArrayAttr &propValue) {
      this->lengths = propValue;
    }
    using memTypeTy = ::mlir::triton::instrument::MemTypeAttr;
    memTypeTy memType;

    auto getMemType() const {
      auto &propStorage = this->memType;
      return ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(propStorage);
    }
    void setMemType(const ::mlir::triton::instrument::MemTypeAttr &propValue) {
      this->memType = propValue;
    }
    using offsetsTy = ::mlir::DenseI32ArrayAttr;
    offsetsTy offsets;

    auto getOffsets() const {
      auto &propStorage = this->offsets;
      return ::llvm::cast<::mlir::DenseI32ArrayAttr>(propStorage);
    }
    void setOffsets(const ::mlir::DenseI32ArrayAttr &propValue) {
      this->offsets = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.lengths == this->lengths &&
        rhs.memType == this->memType &&
        rhs.offsets == this->offsets &&
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
  ExperimentalBufferDescriptorsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_buffer_descriptors", odsAttrs.getContext());
  }

  ExperimentalBufferDescriptorsOpGenericAdaptorBase(ExperimentalBufferDescriptorsOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::DenseI32ArrayAttr getOffsetsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().offsets);
    return attr;
  }

  ::llvm::ArrayRef<int32_t> getOffsets();
  ::mlir::DenseI32ArrayAttr getLengthsAttr() {
    auto attr = ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().lengths);
    return attr;
  }

  ::llvm::ArrayRef<int32_t> getLengths();
  ::mlir::triton::instrument::MemTypeAttr getMemTypeAttr() {
    auto attr = ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(getProperties().memType);
    return attr;
  }

  ::mlir::triton::instrument::MemType getMemType();
};

} // namespace detail
template <typename RangeT>
class ExperimentalBufferDescriptorsOpGenericAdaptor : public detail::ExperimentalBufferDescriptorsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalBufferDescriptorsOpGenericAdaptorBase;
public:
  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ExperimentalBufferDescriptorsOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalBufferDescriptorsOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, const ExperimentalBufferDescriptorsOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalBufferDescriptorsOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalBufferDescriptorsOp>>>
  ExperimentalBufferDescriptorsOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class ExperimentalBufferDescriptorsOpAdaptor : public ExperimentalBufferDescriptorsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalBufferDescriptorsOpGenericAdaptor::ExperimentalBufferDescriptorsOpGenericAdaptor;
  ExperimentalBufferDescriptorsOpAdaptor(ExperimentalBufferDescriptorsOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalBufferDescriptorsOp : public ::mlir::Op<ExperimentalBufferDescriptorsOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalBufferDescriptorsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalBufferDescriptorsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("lengths"), ::llvm::StringRef("memType"), ::llvm::StringRef("offsets")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getLengthsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getLengthsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemTypeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOffsetsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOffsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_buffer_descriptors");
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
  ::mlir::DenseI32ArrayAttr getOffsetsAttr() {
    return ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().offsets);
  }

  ::llvm::ArrayRef<int32_t> getOffsets();
  ::mlir::DenseI32ArrayAttr getLengthsAttr() {
    return ::llvm::cast<::mlir::DenseI32ArrayAttr>(getProperties().lengths);
  }

  ::llvm::ArrayRef<int32_t> getLengths();
  ::mlir::triton::instrument::MemTypeAttr getMemTypeAttr() {
    return ::llvm::cast<::mlir::triton::instrument::MemTypeAttr>(getProperties().memType);
  }

  ::mlir::triton::instrument::MemType getMemType();
  void setOffsetsAttr(::mlir::DenseI32ArrayAttr attr) {
    getProperties().offsets = attr;
  }

  void setOffsets(::llvm::ArrayRef<int32_t> attrValue);
  void setLengthsAttr(::mlir::DenseI32ArrayAttr attr) {
    getProperties().lengths = attr;
  }

  void setLengths(::llvm::ArrayRef<int32_t> attrValue);
  void setMemTypeAttr(::mlir::triton::instrument::MemTypeAttr attr) {
    getProperties().memType = attr;
  }

  void setMemType(::mlir::triton::instrument::MemType attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalBufferDescriptorsOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalBufferDescriptorsOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
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

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalBufferDescriptorsOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalLockAcquireOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalLockAcquireOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalLockAcquireOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_lock_acquire", odsAttrs.getContext());
  }

  ExperimentalLockAcquireOpGenericAdaptorBase(ExperimentalLockAcquireOp op);

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
class ExperimentalLockAcquireOpGenericAdaptor : public detail::ExperimentalLockAcquireOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalLockAcquireOpGenericAdaptorBase;
  static constexpr int odsIndex_lock = 0;
  static constexpr int odsIndex_pred = 1;
public:
  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ExperimentalLockAcquireOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalLockAcquireOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, const ExperimentalLockAcquireOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalLockAcquireOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalLockAcquireOp>>>
  ExperimentalLockAcquireOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLock() {
    return (*getODSOperands(0).begin());
  }

  ValueT getPred() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalLockAcquireOpAdaptor : public ExperimentalLockAcquireOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalLockAcquireOpGenericAdaptor::ExperimentalLockAcquireOpGenericAdaptor;
  ExperimentalLockAcquireOpAdaptor(ExperimentalLockAcquireOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalLockAcquireOp : public ::mlir::Op<ExperimentalLockAcquireOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalLockAcquireOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalLockAcquireOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lock = 0;
  static constexpr int odsIndex_pred = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_lock_acquire");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getLock() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getLockMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getPredMutable();
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLockAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLockAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLockAcquireOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLockAcquireOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalLockAcquireOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalLockReleaseOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalLockReleaseOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalLockReleaseOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_lock_release", odsAttrs.getContext());
  }

  ExperimentalLockReleaseOpGenericAdaptorBase(ExperimentalLockReleaseOp op);

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
class ExperimentalLockReleaseOpGenericAdaptor : public detail::ExperimentalLockReleaseOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalLockReleaseOpGenericAdaptorBase;
  static constexpr int odsIndex_lock = 0;
  static constexpr int odsIndex_pred = 1;
public:
  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ExperimentalLockReleaseOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalLockReleaseOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, const ExperimentalLockReleaseOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalLockReleaseOp, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalLockReleaseOp>>>
  ExperimentalLockReleaseOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLock() {
    return (*getODSOperands(0).begin());
  }

  ValueT getPred() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalLockReleaseOpAdaptor : public ExperimentalLockReleaseOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalLockReleaseOpGenericAdaptor::ExperimentalLockReleaseOpGenericAdaptor;
  ExperimentalLockReleaseOpAdaptor(ExperimentalLockReleaseOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalLockReleaseOp : public ::mlir::Op<ExperimentalLockReleaseOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalLockReleaseOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalLockReleaseOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lock = 0;
  static constexpr int odsIndex_pred = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_lock_release");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::Type> getLock() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getPred() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ::mlir::TypedValue<::mlir::IntegerType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*operands.begin());
  }

  ::mlir::OpOperand &getLockMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getPredMutable();
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static ExperimentalLockReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLockReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalLockReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLockReleaseOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalLockReleaseOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalLockReleaseOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalMemDescToI32Op declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ExperimentalMemDescToI32OpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ExperimentalMemDescToI32OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("tti.experimental_memdesc_to_i32", odsAttrs.getContext());
  }

  ExperimentalMemDescToI32OpGenericAdaptorBase(ExperimentalMemDescToI32Op op);

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
class ExperimentalMemDescToI32OpGenericAdaptor : public detail::ExperimentalMemDescToI32OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExperimentalMemDescToI32OpGenericAdaptorBase;
  static constexpr int odsIndex_memdesc = 0;
public:
  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ExperimentalMemDescToI32OpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ExperimentalMemDescToI32OpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, const ExperimentalMemDescToI32OpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ExperimentalMemDescToI32Op, typename = std::enable_if_t<std::is_same_v<LateInst, ExperimentalMemDescToI32Op>>>
  ExperimentalMemDescToI32OpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemdesc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExperimentalMemDescToI32OpAdaptor : public ExperimentalMemDescToI32OpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExperimentalMemDescToI32OpGenericAdaptor::ExperimentalMemDescToI32OpGenericAdaptor;
  ExperimentalMemDescToI32OpAdaptor(ExperimentalMemDescToI32Op op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ExperimentalMemDescToI32Op : public ::mlir::Op<ExperimentalMemDescToI32Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExperimentalMemDescToI32OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExperimentalMemDescToI32OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_memdesc = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tti.experimental_memdesc_to_i32");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getMemdesc() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::triton::gpu::MemDescType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getMemdescMutable() {
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, Value memdesc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value memdesc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc);
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ExperimentalMemDescToI32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::triton::instrument
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalMemDescToI32Op)


#endif // GET_OP_CLASSES

