/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonInstrumentOps.td                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::triton::instrument::ExperimentalAssertInThreadOp,
::mlir::triton::instrument::ExperimentalBufferDescriptorsOp,
::mlir::triton::instrument::ExperimentalLockAcquireOp,
::mlir::triton::instrument::ExperimentalLockReleaseOp,
::mlir::triton::instrument::ExperimentalMemDescToI32Op
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonInstrumentOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isSignlessInteger(1))) || (((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer or tensor of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonInstrumentOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return (((::llvm::isa<::mlir::Float8E4M3FNType>(elementType))) || ((::llvm::isa<::mlir::Float8E4M3FNUZType>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2Type>(elementType))) || ((::llvm::isa<::mlir::Float8E5M2FNUZType>(elementType))) || ((elementType.isF16())) || ((::llvm::isa<::mlir::BFloat16Type>(elementType))) || ((elementType.isF32())) || ((elementType.isF64()))) || (((elementType.isSignlessInteger(1))) || ((elementType.isSignlessInteger(4))) || ((elementType.isSignlessInteger(8))) || ((elementType.isSignlessInteger(16))) || ((elementType.isSignlessInteger(32))) || ((elementType.isSignlessInteger(64)))) || (((::mlir::isa<::mlir::triton::PointerType>(elementType))) && ([](::mlir::Type pointeeType) { return (true); }(::mlir::cast<::mlir::triton::PointerType>(elementType).getPointeeType()))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of floating-point or integer or ptr values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonInstrumentOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::mlir::isa<::mlir::triton::PointerType>(type))) && ([](::mlir::Type pointeeType) { return (true); }(::mlir::cast<::mlir::triton::PointerType>(type).getPointeeType()))) || (((::llvm::isa<::mlir::RankedTensorType>(type))) && ([](::mlir::Type elementType) { return ((::mlir::isa<::mlir::triton::PointerType>(elementType))) && ([](::mlir::Type pointeeType) { return (true); }(::mlir::cast<::mlir::triton::PointerType>(elementType).getPointeeType())); }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ptr or ranked tensor of ptr values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonInstrumentOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonInstrumentOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::triton::gpu::MemDescType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memory descriptor type (`::mlir::triton::gpu::MemDescType`) in Triton IR type system, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_TritonInstrumentOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonInstrumentOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonInstrumentOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonInstrumentOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonInstrumentOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::BoolAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: bool attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonInstrumentOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonInstrumentOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonInstrumentOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DenseI32ArrayAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: i32 dense array attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonInstrumentOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonInstrumentOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonInstrumentOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::instrument::MemTypeAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_TritonInstrumentOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_TritonInstrumentOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalAssertInThreadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ExperimentalAssertInThreadOpGenericAdaptorBase::ExperimentalAssertInThreadOpGenericAdaptorBase(ExperimentalAssertInThreadOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef ExperimentalAssertInThreadOpGenericAdaptorBase::getMessage() {
  auto attr = getMessageAttr();
  return attr.getValue();
}

bool ExperimentalAssertInThreadOpGenericAdaptorBase::getCheckAny() {
  auto attr = getCheckAnyAttr();
  return attr.getValue();
}


} // namespace detail
ExperimentalAssertInThreadOpAdaptor::ExperimentalAssertInThreadOpAdaptor(ExperimentalAssertInThreadOp op) : ExperimentalAssertInThreadOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ExperimentalAssertInThreadOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_check_any = getProperties().check_any; (void)tblgen_check_any;
  if (!tblgen_check_any) return emitError(loc, "'tti.experimental_assert_in_thread' op requires attribute 'check_any'");
  auto tblgen_message = getProperties().message; (void)tblgen_message;
  if (!tblgen_message) return emitError(loc, "'tti.experimental_assert_in_thread' op requires attribute 'message'");

  if (tblgen_message && !((::llvm::isa<::mlir::StringAttr>(tblgen_message))))
    return emitError(loc, "'tti.experimental_assert_in_thread' op attribute 'message' failed to satisfy constraint: string attribute");

  if (tblgen_check_any && !((::llvm::isa<::mlir::BoolAttr>(tblgen_check_any))))
    return emitError(loc, "'tti.experimental_assert_in_thread' op attribute 'check_any' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ExperimentalAssertInThreadOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.check_any;
       auto attr = dict.get("check_any");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `check_any` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.message;
       auto attr = dict.get("message");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `message` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ExperimentalAssertInThreadOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.check_any;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("check_any",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.message;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("message",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ExperimentalAssertInThreadOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.check_any.getAsOpaquePointer()), 
    llvm::hash_value(prop.message.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ExperimentalAssertInThreadOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "check_any")
      return prop.check_any;

    if (name == "message")
      return prop.message;
  return std::nullopt;
}

void ExperimentalAssertInThreadOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "check_any") {
       prop.check_any = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.check_any)>>(value);
       return;
    }

    if (name == "message") {
       prop.message = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.message)>>(value);
       return;
    }
}

void ExperimentalAssertInThreadOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.check_any) attrs.append("check_any", prop.check_any);

    if (prop.message) attrs.append("message", prop.message);
}

::llvm::LogicalResult ExperimentalAssertInThreadOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getCheckAnyAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps2(attr, "check_any", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMessageAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps1(attr, "message", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ExperimentalAssertInThreadOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.check_any)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.message)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExperimentalAssertInThreadOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.check_any);
  writer.writeAttribute(prop.message);
}

::llvm::StringRef ExperimentalAssertInThreadOp::getMessage() {
  auto attr = getMessageAttr();
  return attr.getValue();
}

bool ExperimentalAssertInThreadOp::getCheckAny() {
  auto attr = getCheckAnyAttr();
  return attr.getValue();
}

void ExperimentalAssertInThreadOp::setMessage(::llvm::StringRef attrValue) {
  getProperties().message = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void ExperimentalAssertInThreadOp::setCheckAny(bool attrValue) {
  getProperties().check_any = ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue);
}

void ExperimentalAssertInThreadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any) {
  odsState.addOperands(condition);
  odsState.getOrAddProperties<Properties>().message = message;
  odsState.getOrAddProperties<Properties>().check_any = check_any;
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(condition)>(condition), std::forward<decltype(message)>(message), std::forward<decltype(check_any)>(check_any));
  auto __res__ = ::llvm::dyn_cast<ExperimentalAssertInThreadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any) {
  return create(builder, builder.getLoc(), std::forward<decltype(condition)>(condition), std::forward<decltype(message)>(message), std::forward<decltype(check_any)>(check_any));
}

void ExperimentalAssertInThreadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any) {
  odsState.addOperands(condition);
  odsState.getOrAddProperties<Properties>().message = message;
  odsState.getOrAddProperties<Properties>().check_any = check_any;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(condition)>(condition), std::forward<decltype(message)>(message), std::forward<decltype(check_any)>(check_any));
  auto __res__ = ::llvm::dyn_cast<ExperimentalAssertInThreadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::StringAttr message, ::mlir::BoolAttr check_any) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(condition)>(condition), std::forward<decltype(message)>(message), std::forward<decltype(check_any)>(check_any));
}

void ExperimentalAssertInThreadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::llvm::StringRef message, bool check_any) {
  odsState.addOperands(condition);
  odsState.getOrAddProperties<Properties>().message = odsBuilder.getStringAttr(message);
  odsState.getOrAddProperties<Properties>().check_any = odsBuilder.getBoolAttr(check_any);
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value condition, ::llvm::StringRef message, bool check_any) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(condition)>(condition), std::forward<decltype(message)>(message), std::forward<decltype(check_any)>(check_any));
  auto __res__ = ::llvm::dyn_cast<ExperimentalAssertInThreadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value condition, ::llvm::StringRef message, bool check_any) {
  return create(builder, builder.getLoc(), std::forward<decltype(condition)>(condition), std::forward<decltype(message)>(message), std::forward<decltype(check_any)>(check_any));
}

void ExperimentalAssertInThreadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message, bool check_any) {
  odsState.addOperands(condition);
  odsState.getOrAddProperties<Properties>().message = odsBuilder.getStringAttr(message);
  odsState.getOrAddProperties<Properties>().check_any = odsBuilder.getBoolAttr(check_any);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message, bool check_any) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(condition)>(condition), std::forward<decltype(message)>(message), std::forward<decltype(check_any)>(check_any));
  auto __res__ = ::llvm::dyn_cast<ExperimentalAssertInThreadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::llvm::StringRef message, bool check_any) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(condition)>(condition), std::forward<decltype(message)>(message), std::forward<decltype(check_any)>(check_any));
}

void ExperimentalAssertInThreadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ExperimentalAssertInThreadOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalAssertInThreadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ExperimentalAssertInThreadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalAssertInThreadOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalAssertInThreadOp ExperimentalAssertInThreadOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ExperimentalAssertInThreadOp::verifyInvariantsImpl() {
  auto tblgen_check_any = getProperties().check_any; (void)tblgen_check_any;
  if (!tblgen_check_any) return emitOpError("requires attribute 'check_any'");
  auto tblgen_message = getProperties().message; (void)tblgen_message;
  if (!tblgen_message) return emitOpError("requires attribute 'message'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps1(*this, tblgen_message, "message")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps2(*this, tblgen_check_any, "check_any")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonInstrumentOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ExperimentalAssertInThreadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExperimentalAssertInThreadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand conditionRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> conditionOperands(&conditionRawOperand, 1);  ::llvm::SMLoc conditionOperandsLoc;
  (void)conditionOperandsLoc;
  ::mlir::StringAttr messageAttr;
  ::mlir::Type conditionRawType{};
  ::llvm::ArrayRef<::mlir::Type> conditionTypes(&conditionRawType, 1);

  conditionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(conditionRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(messageAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (messageAttr) result.getOrAddProperties<ExperimentalAssertInThreadOp::Properties>().message = messageAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    conditionRawType = type;
  }
  if (parser.resolveOperands(conditionOperands, conditionTypes, conditionOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExperimentalAssertInThreadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCondition();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMessageAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("message");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getCondition().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExperimentalAssertInThreadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Write::get(), 0, false, ::mlir::triton::GlobalMemory::get());
}


} // namespace mlir::triton::instrument
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalAssertInThreadOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalBufferDescriptorsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ExperimentalBufferDescriptorsOpGenericAdaptorBase::ExperimentalBufferDescriptorsOpGenericAdaptorBase(ExperimentalBufferDescriptorsOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::ArrayRef<int32_t> ExperimentalBufferDescriptorsOpGenericAdaptorBase::getOffsets() {
  auto attr = getOffsetsAttr();
  return attr;
}

::llvm::ArrayRef<int32_t> ExperimentalBufferDescriptorsOpGenericAdaptorBase::getLengths() {
  auto attr = getLengthsAttr();
  return attr;
}

::mlir::triton::instrument::MemType ExperimentalBufferDescriptorsOpGenericAdaptorBase::getMemType() {
  auto attr = getMemTypeAttr();
  return attr.getValue();
}


} // namespace detail
ExperimentalBufferDescriptorsOpAdaptor::ExperimentalBufferDescriptorsOpAdaptor(ExperimentalBufferDescriptorsOp op) : ExperimentalBufferDescriptorsOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ExperimentalBufferDescriptorsOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_lengths = getProperties().lengths; (void)tblgen_lengths;
  if (!tblgen_lengths) return emitError(loc, "'tti.experimental_buffer_descriptors' op requires attribute 'lengths'");
  auto tblgen_memType = getProperties().memType; (void)tblgen_memType;
  if (!tblgen_memType) return emitError(loc, "'tti.experimental_buffer_descriptors' op requires attribute 'memType'");
  auto tblgen_offsets = getProperties().offsets; (void)tblgen_offsets;
  if (!tblgen_offsets) return emitError(loc, "'tti.experimental_buffer_descriptors' op requires attribute 'offsets'");

  if (tblgen_offsets && !((::llvm::isa<::mlir::DenseI32ArrayAttr>(tblgen_offsets))))
    return emitError(loc, "'tti.experimental_buffer_descriptors' op attribute 'offsets' failed to satisfy constraint: i32 dense array attribute");

  if (tblgen_lengths && !((::llvm::isa<::mlir::DenseI32ArrayAttr>(tblgen_lengths))))
    return emitError(loc, "'tti.experimental_buffer_descriptors' op attribute 'lengths' failed to satisfy constraint: i32 dense array attribute");

  if (tblgen_memType && !((::llvm::isa<::mlir::triton::instrument::MemTypeAttr>(tblgen_memType))))
    return emitError(loc, "'tti.experimental_buffer_descriptors' op attribute 'memType' failed to satisfy constraint: allowed 32-bit signless integer cases: 0, 1");
  return ::mlir::success();
}

::llvm::LogicalResult ExperimentalBufferDescriptorsOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.lengths;
       auto attr = dict.get("lengths");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `lengths` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.memType;
       auto attr = dict.get("memType");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `memType` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.offsets;
       auto attr = dict.get("offsets");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `offsets` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ExperimentalBufferDescriptorsOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.lengths;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("lengths",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.memType;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("memType",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.offsets;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("offsets",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ExperimentalBufferDescriptorsOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.lengths.getAsOpaquePointer()), 
    llvm::hash_value(prop.memType.getAsOpaquePointer()), 
    llvm::hash_value(prop.offsets.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ExperimentalBufferDescriptorsOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "lengths")
      return prop.lengths;

    if (name == "memType")
      return prop.memType;

    if (name == "offsets")
      return prop.offsets;
  return std::nullopt;
}

void ExperimentalBufferDescriptorsOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "lengths") {
       prop.lengths = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.lengths)>>(value);
       return;
    }

    if (name == "memType") {
       prop.memType = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.memType)>>(value);
       return;
    }

    if (name == "offsets") {
       prop.offsets = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.offsets)>>(value);
       return;
    }
}

void ExperimentalBufferDescriptorsOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.lengths) attrs.append("lengths", prop.lengths);

    if (prop.memType) attrs.append("memType", prop.memType);

    if (prop.offsets) attrs.append("offsets", prop.offsets);
}

::llvm::LogicalResult ExperimentalBufferDescriptorsOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getLengthsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps3(attr, "lengths", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMemTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps4(attr, "memType", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOffsetsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps3(attr, "offsets", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ExperimentalBufferDescriptorsOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.lengths)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.memType)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.offsets)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExperimentalBufferDescriptorsOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.lengths);
  writer.writeAttribute(prop.memType);
  writer.writeAttribute(prop.offsets);
}

::llvm::ArrayRef<int32_t> ExperimentalBufferDescriptorsOp::getOffsets() {
  auto attr = getOffsetsAttr();
  return attr;
}

::llvm::ArrayRef<int32_t> ExperimentalBufferDescriptorsOp::getLengths() {
  auto attr = getLengthsAttr();
  return attr;
}

::mlir::triton::instrument::MemType ExperimentalBufferDescriptorsOp::getMemType() {
  auto attr = getMemTypeAttr();
  return attr.getValue();
}

void ExperimentalBufferDescriptorsOp::setOffsets(::llvm::ArrayRef<int32_t> attrValue) {
  getProperties().offsets = ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue);
}

void ExperimentalBufferDescriptorsOp::setLengths(::llvm::ArrayRef<int32_t> attrValue) {
  getProperties().lengths = ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue);
}

void ExperimentalBufferDescriptorsOp::setMemType(::mlir::triton::instrument::MemType attrValue) {
  getProperties().memType = ::mlir::triton::instrument::MemTypeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void ExperimentalBufferDescriptorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType) {
  odsState.getOrAddProperties<Properties>().offsets = offsets;
  odsState.getOrAddProperties<Properties>().lengths = lengths;
  odsState.getOrAddProperties<Properties>().memType = memType;
  odsState.addTypes(result);
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(offsets)>(offsets), std::forward<decltype(lengths)>(lengths), std::forward<decltype(memType)>(memType));
  auto __res__ = ::llvm::dyn_cast<ExperimentalBufferDescriptorsOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(offsets)>(offsets), std::forward<decltype(lengths)>(lengths), std::forward<decltype(memType)>(memType));
}

void ExperimentalBufferDescriptorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType) {
  odsState.getOrAddProperties<Properties>().offsets = offsets;
  odsState.getOrAddProperties<Properties>().lengths = lengths;
  odsState.getOrAddProperties<Properties>().memType = memType;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(offsets)>(offsets), std::forward<decltype(lengths)>(lengths), std::forward<decltype(memType)>(memType));
  auto __res__ = ::llvm::dyn_cast<ExperimentalBufferDescriptorsOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::DenseI32ArrayAttr offsets, ::mlir::DenseI32ArrayAttr lengths, ::mlir::triton::instrument::MemTypeAttr memType) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(offsets)>(offsets), std::forward<decltype(lengths)>(lengths), std::forward<decltype(memType)>(memType));
}

void ExperimentalBufferDescriptorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType) {
  odsState.getOrAddProperties<Properties>().offsets = odsBuilder.getDenseI32ArrayAttr(offsets);
  odsState.getOrAddProperties<Properties>().lengths = odsBuilder.getDenseI32ArrayAttr(lengths);
  odsState.getOrAddProperties<Properties>().memType = ::mlir::triton::instrument::MemTypeAttr::get(odsBuilder.getContext(), memType);
  odsState.addTypes(result);
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(offsets)>(offsets), std::forward<decltype(lengths)>(lengths), std::forward<decltype(memType)>(memType));
  auto __res__ = ::llvm::dyn_cast<ExperimentalBufferDescriptorsOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(offsets)>(offsets), std::forward<decltype(lengths)>(lengths), std::forward<decltype(memType)>(memType));
}

void ExperimentalBufferDescriptorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType) {
  odsState.getOrAddProperties<Properties>().offsets = odsBuilder.getDenseI32ArrayAttr(offsets);
  odsState.getOrAddProperties<Properties>().lengths = odsBuilder.getDenseI32ArrayAttr(lengths);
  odsState.getOrAddProperties<Properties>().memType = ::mlir::triton::instrument::MemTypeAttr::get(odsBuilder.getContext(), memType);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(offsets)>(offsets), std::forward<decltype(lengths)>(lengths), std::forward<decltype(memType)>(memType));
  auto __res__ = ::llvm::dyn_cast<ExperimentalBufferDescriptorsOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::ArrayRef<int32_t> offsets, ::llvm::ArrayRef<int32_t> lengths, ::mlir::triton::instrument::MemType memType) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(offsets)>(offsets), std::forward<decltype(lengths)>(lengths), std::forward<decltype(memType)>(memType));
}

void ExperimentalBufferDescriptorsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<ExperimentalBufferDescriptorsOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalBufferDescriptorsOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ExperimentalBufferDescriptorsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalBufferDescriptorsOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalBufferDescriptorsOp ExperimentalBufferDescriptorsOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ExperimentalBufferDescriptorsOp::verifyInvariantsImpl() {
  auto tblgen_lengths = getProperties().lengths; (void)tblgen_lengths;
  if (!tblgen_lengths) return emitOpError("requires attribute 'lengths'");
  auto tblgen_memType = getProperties().memType; (void)tblgen_memType;
  if (!tblgen_memType) return emitOpError("requires attribute 'memType'");
  auto tblgen_offsets = getProperties().offsets; (void)tblgen_offsets;
  if (!tblgen_offsets) return emitOpError("requires attribute 'offsets'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps3(*this, tblgen_offsets, "offsets")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps3(*this, tblgen_lengths, "lengths")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TritonInstrumentOps4(*this, tblgen_memType, "memType")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonInstrumentOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ExperimentalBufferDescriptorsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExperimentalBufferDescriptorsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::DenseI32ArrayAttr offsetsAttr;
  ::mlir::DenseI32ArrayAttr lengthsAttr;
  ::mlir::triton::instrument::MemTypeAttr memTypeAttr;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);

  if (parser.parseCustomAttributeWithFallback(offsetsAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (offsetsAttr) result.getOrAddProperties<ExperimentalBufferDescriptorsOp::Properties>().offsets = offsetsAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(lengthsAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (lengthsAttr) result.getOrAddProperties<ExperimentalBufferDescriptorsOp::Properties>().lengths = lengthsAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"shared_mem","tensor_mem"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "memType", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'memType' [shared_mem, tensor_mem]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::instrument::symbolizeMemType(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "memType attribute specification: \"" << attrStr << '"';;

      memTypeAttr = ::mlir::triton::instrument::MemTypeAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<ExperimentalBufferDescriptorsOp::Properties>().memType = memTypeAttr;
    }
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  result.addTypes(resultTypes);
  return ::mlir::success();
}

void ExperimentalBufferDescriptorsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getOffsetsAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getLengthsAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';

  {
    auto caseValue = getMemType();
    auto caseValueStr = stringifyMemType(caseValue);
    _odsPrinter << caseValueStr;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("offsets");
  elidedAttrs.push_back("lengths");
  elidedAttrs.push_back("memType");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::RankedTensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExperimentalBufferDescriptorsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::instrument
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalBufferDescriptorsOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalLockAcquireOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ExperimentalLockAcquireOpGenericAdaptorBase::ExperimentalLockAcquireOpGenericAdaptorBase(ExperimentalLockAcquireOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ExperimentalLockAcquireOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}


} // namespace detail
ExperimentalLockAcquireOpAdaptor::ExperimentalLockAcquireOpAdaptor(ExperimentalLockAcquireOp op) : ExperimentalLockAcquireOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ExperimentalLockAcquireOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExperimentalLockAcquireOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange ExperimentalLockAcquireOp::getPredMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult ExperimentalLockAcquireOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute ExperimentalLockAcquireOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ExperimentalLockAcquireOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> ExperimentalLockAcquireOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void ExperimentalLockAcquireOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void ExperimentalLockAcquireOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult ExperimentalLockAcquireOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void ExperimentalLockAcquireOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  odsState.addOperands(lock);
  if (pred)
    odsState.addOperands(pred);
}

ExperimentalLockAcquireOp ExperimentalLockAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(lock)>(lock), std::forward<decltype(pred)>(pred));
  auto __res__ = ::llvm::dyn_cast<ExperimentalLockAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalLockAcquireOp ExperimentalLockAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  return create(builder, builder.getLoc(), std::forward<decltype(lock)>(lock), std::forward<decltype(pred)>(pred));
}

void ExperimentalLockAcquireOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  odsState.addOperands(lock);
  if (pred)
    odsState.addOperands(pred);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ExperimentalLockAcquireOp ExperimentalLockAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(lock)>(lock), std::forward<decltype(pred)>(pred));
  auto __res__ = ::llvm::dyn_cast<ExperimentalLockAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalLockAcquireOp ExperimentalLockAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(lock)>(lock), std::forward<decltype(pred)>(pred));
}

void ExperimentalLockAcquireOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ExperimentalLockAcquireOp ExperimentalLockAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalLockAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalLockAcquireOp ExperimentalLockAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ExperimentalLockAcquireOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ExperimentalLockAcquireOp ExperimentalLockAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalLockAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalLockAcquireOp ExperimentalLockAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ExperimentalLockAcquireOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonInstrumentOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonInstrumentOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ExperimentalLockAcquireOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExperimentalLockAcquireOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lockRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lockOperands(&lockRawOperand, 1);  ::llvm::SMLoc lockOperandsLoc;
  (void)lockOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> predOperands;
  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::Type lockRawType{};
  ::llvm::ArrayRef<::mlir::Type> lockTypes(&lockRawType, 1);

  lockOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lockRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    predOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      predOperands.push_back(operand);
    }
  }
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lockRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(lockOperands, lockTypes, lockOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType0, predOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExperimentalLockAcquireOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLock();
  if (getPred()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getPred())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getLock().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExperimentalLockAcquireOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Write::get(), 0, false, ::mlir::triton::GlobalMemory::get());
}


} // namespace mlir::triton::instrument
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalLockAcquireOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalLockReleaseOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ExperimentalLockReleaseOpGenericAdaptorBase::ExperimentalLockReleaseOpGenericAdaptorBase(ExperimentalLockReleaseOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> ExperimentalLockReleaseOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}


} // namespace detail
ExperimentalLockReleaseOpAdaptor::ExperimentalLockReleaseOpAdaptor(ExperimentalLockReleaseOp op) : ExperimentalLockReleaseOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ExperimentalLockReleaseOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExperimentalLockReleaseOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange ExperimentalLockReleaseOp::getPredMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult ExperimentalLockReleaseOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute ExperimentalLockReleaseOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ExperimentalLockReleaseOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> ExperimentalLockReleaseOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void ExperimentalLockReleaseOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void ExperimentalLockReleaseOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult ExperimentalLockReleaseOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void ExperimentalLockReleaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  odsState.addOperands(lock);
  if (pred)
    odsState.addOperands(pred);
}

ExperimentalLockReleaseOp ExperimentalLockReleaseOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(lock)>(lock), std::forward<decltype(pred)>(pred));
  auto __res__ = ::llvm::dyn_cast<ExperimentalLockReleaseOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalLockReleaseOp ExperimentalLockReleaseOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  return create(builder, builder.getLoc(), std::forward<decltype(lock)>(lock), std::forward<decltype(pred)>(pred));
}

void ExperimentalLockReleaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  odsState.addOperands(lock);
  if (pred)
    odsState.addOperands(pred);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ExperimentalLockReleaseOp ExperimentalLockReleaseOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(lock)>(lock), std::forward<decltype(pred)>(pred));
  auto __res__ = ::llvm::dyn_cast<ExperimentalLockReleaseOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalLockReleaseOp ExperimentalLockReleaseOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lock, /*optional*/::mlir::Value pred) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(lock)>(lock), std::forward<decltype(pred)>(pred));
}

void ExperimentalLockReleaseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ExperimentalLockReleaseOp ExperimentalLockReleaseOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalLockReleaseOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalLockReleaseOp ExperimentalLockReleaseOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ExperimentalLockReleaseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ExperimentalLockReleaseOp ExperimentalLockReleaseOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalLockReleaseOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalLockReleaseOp ExperimentalLockReleaseOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ExperimentalLockReleaseOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonInstrumentOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonInstrumentOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ExperimentalLockReleaseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExperimentalLockReleaseOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lockRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lockOperands(&lockRawOperand, 1);  ::llvm::SMLoc lockOperandsLoc;
  (void)lockOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> predOperands;
  ::llvm::SMLoc predOperandsLoc;
  (void)predOperandsLoc;
  ::mlir::Type lockRawType{};
  ::llvm::ArrayRef<::mlir::Type> lockTypes(&lockRawType, 1);

  lockOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lockRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    predOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      predOperands.push_back(operand);
    }
  }
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lockRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(lockOperands, lockTypes, lockOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(predOperands, odsBuildableType0, predOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExperimentalLockReleaseOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLock();
  if (getPred()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getPred())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getLock().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExperimentalLockReleaseOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Write::get(), 0, false, ::mlir::triton::GlobalMemory::get());
}


} // namespace mlir::triton::instrument
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalLockReleaseOp)

namespace mlir::triton::instrument {


//===----------------------------------------------------------------------===//
// ::mlir::triton::instrument::ExperimentalMemDescToI32Op definitions
//===----------------------------------------------------------------------===//

namespace detail {

ExperimentalMemDescToI32OpGenericAdaptorBase::ExperimentalMemDescToI32OpGenericAdaptorBase(ExperimentalMemDescToI32Op op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
ExperimentalMemDescToI32OpAdaptor::ExperimentalMemDescToI32OpAdaptor(ExperimentalMemDescToI32Op op) : ExperimentalMemDescToI32OpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ExperimentalMemDescToI32OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult ExperimentalMemDescToI32Op::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute ExperimentalMemDescToI32Op::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ExperimentalMemDescToI32Op::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> ExperimentalMemDescToI32Op::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void ExperimentalMemDescToI32Op::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void ExperimentalMemDescToI32Op::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult ExperimentalMemDescToI32Op::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void ExperimentalMemDescToI32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memdesc) {
      build(odsBuilder, odsState, odsBuilder.getI32Type(), memdesc);
    
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, Value memdesc) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(memdesc)>(memdesc));
  auto __res__ = ::llvm::dyn_cast<ExperimentalMemDescToI32Op>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::ImplicitLocOpBuilder &builder, Value memdesc) {
  return create(builder, builder.getLoc(), std::forward<decltype(memdesc)>(memdesc));
}

void ExperimentalMemDescToI32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memdesc) {
  odsState.addOperands(memdesc);
  odsState.addTypes(result);
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value memdesc) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(memdesc)>(memdesc));
  auto __res__ = ::llvm::dyn_cast<ExperimentalMemDescToI32Op>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value memdesc) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(memdesc)>(memdesc));
}

void ExperimentalMemDescToI32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc) {
  odsState.addOperands(memdesc);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(memdesc)>(memdesc));
  auto __res__ = ::llvm::dyn_cast<ExperimentalMemDescToI32Op>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value memdesc) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(memdesc)>(memdesc));
}

void ExperimentalMemDescToI32Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalMemDescToI32Op>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ExperimentalMemDescToI32Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ExperimentalMemDescToI32Op>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ExperimentalMemDescToI32Op ExperimentalMemDescToI32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ExperimentalMemDescToI32Op::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonInstrumentOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TritonInstrumentOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ExperimentalMemDescToI32Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExperimentalMemDescToI32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand memdescRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memdescOperands(&memdescRawOperand, 1);  ::llvm::SMLoc memdescOperandsLoc;
  (void)memdescOperandsLoc;
  ::mlir::Type memdescRawType{};
  ::llvm::ArrayRef<::mlir::Type> memdescTypes(&memdescRawType, 1);

  memdescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memdescRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::triton::gpu::MemDescType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memdescRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(memdescOperands, memdescTypes, memdescOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExperimentalMemDescToI32Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMemdesc();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemdesc().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::triton::gpu::MemDescType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExperimentalMemDescToI32Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::instrument
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::instrument::ExperimentalMemDescToI32Op)


#endif // GET_OP_CLASSES

