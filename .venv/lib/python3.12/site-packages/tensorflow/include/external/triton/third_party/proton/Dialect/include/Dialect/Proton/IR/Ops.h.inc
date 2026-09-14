/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ProtonOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::proton {

/// Record an event
/// This operation annotates a region of IR where events are recorded.
///     Events can be classified as hardware or software events.
///     Hardware events are provided by the hardware performance counters obtained in later passes that convert Triton to target-specific IR.
///     Software events are provided by the user or the compiler.
/// 
///     Example:
/// 
///     ```mlir
///     proton.record start "name0"
///     ...
///     proton.record end "name0"
///     ```
/// 
///     Scope names cannot be reused within the same function.
class RecordOp;

} // namespace mlir::triton::proton
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::triton::proton {


//===----------------------------------------------------------------------===//
// ::mlir::triton::proton::RecordOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RecordOpGenericAdaptorBase {
public:
  struct Properties {
    using isStartTy = ::mlir::UnitAttr;
    isStartTy isStart;

    auto getIsStart() const {
      auto &propStorage = this->isStart;
      return ::llvm::dyn_cast_or_null<::mlir::UnitAttr>(propStorage);
    }
    void setIsStart(const ::mlir::UnitAttr &propValue) {
      this->isStart = propValue;
    }
    using nameTy = ::mlir::StringAttr;
    nameTy name;

    auto getName() const {
      auto &propStorage = this->name;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setName(const ::mlir::StringAttr &propValue) {
      this->name = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.isStart == this->isStart &&
        rhs.name == this->name &&
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
  RecordOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("proton.record", odsAttrs.getContext());
  }

  RecordOpGenericAdaptorBase(RecordOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::UnitAttr getIsStartAttr();
  bool getIsStart();
  ::mlir::StringAttr getNameAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().name);
    return attr;
  }

  ::llvm::StringRef getName();
};

} // namespace detail
template <typename RangeT>
class RecordOpGenericAdaptor : public detail::RecordOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RecordOpGenericAdaptorBase;
public:
  RecordOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RecordOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : RecordOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RecordOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RecordOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  RecordOpGenericAdaptor(RangeT values, const RecordOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = RecordOp, typename = std::enable_if_t<std::is_same_v<LateInst, RecordOp>>>
  RecordOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class RecordOpAdaptor : public RecordOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RecordOpGenericAdaptor::RecordOpGenericAdaptor;
  RecordOpAdaptor(RecordOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RecordOp : public ::mlir::Op<RecordOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RecordOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RecordOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isStart"), ::llvm::StringRef("name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsStartAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsStartAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("proton.record");
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
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::UnitAttr getIsStartAttr() {
    return ::llvm::dyn_cast_or_null<::mlir::UnitAttr>(getProperties().isStart);
  }

  bool getIsStart();
  ::mlir::StringAttr getNameAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().name);
  }

  ::llvm::StringRef getName();
  void setIsStartAttr(::mlir::UnitAttr attr) {
    getProperties().isStart = attr;
  }

  void setIsStart(bool attrValue);
  void setNameAttr(::mlir::StringAttr attr) {
    getProperties().name = attr;
  }

  void setName(::llvm::StringRef attrValue);
  ::mlir::Attribute removeIsStartAttr() {
      auto attr = getProperties().isStart;
      getProperties().isStart = {};
      return attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name);
  static RecordOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name);
  static RecordOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name);
  static RecordOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name);
  static RecordOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool isStart, ::llvm::StringRef name);
  static RecordOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/bool isStart, ::llvm::StringRef name);
  static RecordOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/bool isStart, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool isStart, ::llvm::StringRef name);
  static RecordOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, /*optional*/bool isStart, ::llvm::StringRef name);
  static RecordOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, /*optional*/bool isStart, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RecordOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RecordOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RecordOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RecordOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
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

} // namespace mlir::triton::proton
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::proton::RecordOp)


#endif // GET_OP_CLASSES

