/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ProtonOps.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::triton::proton::RecordOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ProtonOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::UnitAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: unit attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ProtonOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ProtonOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ProtonOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_ProtonOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_ProtonOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace mlir::triton::proton {


//===----------------------------------------------------------------------===//
// ::mlir::triton::proton::RecordOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

RecordOpGenericAdaptorBase::RecordOpGenericAdaptorBase(RecordOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::UnitAttr RecordOpGenericAdaptorBase::getIsStartAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::UnitAttr>(getProperties().isStart);
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool RecordOpGenericAdaptorBase::getIsStart() {
  auto attr = getIsStartAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::llvm::StringRef RecordOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}


} // namespace detail
RecordOpAdaptor::RecordOpAdaptor(RecordOp op) : RecordOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RecordOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_isStart = getProperties().isStart; (void)tblgen_isStart;
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'proton.record' op requires attribute 'name'");

  if (tblgen_isStart && !((::llvm::isa<::mlir::UnitAttr>(tblgen_isStart))))
    return emitError(loc, "'proton.record' op attribute 'isStart' failed to satisfy constraint: unit attribute");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'proton.record' op attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult RecordOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.isStart;
       auto attr = dict.get("isStart");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `isStart` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.name;
       auto attr = dict.get("name");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute RecordOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.isStart;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("isStart",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code RecordOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.isStart.getAsOpaquePointer()), 
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> RecordOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "isStart")
      return prop.isStart;

    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void RecordOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "isStart") {
       prop.isStart = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.isStart)>>(value);
       return;
    }

    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void RecordOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.isStart) attrs.append("isStart", prop.isStart);

    if (prop.name) attrs.append("name", prop.name);
}

::llvm::LogicalResult RecordOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getIsStartAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ProtonOps1(attr, "isStart", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_ProtonOps2(attr, "name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult RecordOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.isStart)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void RecordOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.isStart);
  writer.writeAttribute(prop.name);
}

bool RecordOp::getIsStart() {
  auto attr = getIsStartAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::llvm::StringRef RecordOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void RecordOp::setIsStart(bool attrValue) {
    auto &odsProp = getProperties().isStart;
    if (attrValue)
      odsProp = ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr);
    else
      odsProp = nullptr;
}

void RecordOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void RecordOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name) {
  if (isStart) {
    odsState.getOrAddProperties<Properties>().isStart = isStart;
  }
  odsState.getOrAddProperties<Properties>().name = name;
}

RecordOp RecordOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(isStart)>(isStart), std::forward<decltype(name)>(name));
  auto __res__ = ::llvm::dyn_cast<RecordOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RecordOp RecordOp::create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name) {
  return create(builder, builder.getLoc(), std::forward<decltype(isStart)>(isStart), std::forward<decltype(name)>(name));
}

void RecordOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name) {
  if (isStart) {
    odsState.getOrAddProperties<Properties>().isStart = isStart;
  }
  odsState.getOrAddProperties<Properties>().name = name;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

RecordOp RecordOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(isStart)>(isStart), std::forward<decltype(name)>(name));
  auto __res__ = ::llvm::dyn_cast<RecordOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RecordOp RecordOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr isStart, ::mlir::StringAttr name) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(isStart)>(isStart), std::forward<decltype(name)>(name));
}

void RecordOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool isStart, ::llvm::StringRef name) {
  if (isStart) {
    odsState.getOrAddProperties<Properties>().isStart = ((isStart) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
}

RecordOp RecordOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/bool isStart, ::llvm::StringRef name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(isStart)>(isStart), std::forward<decltype(name)>(name));
  auto __res__ = ::llvm::dyn_cast<RecordOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RecordOp RecordOp::create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/bool isStart, ::llvm::StringRef name) {
  return create(builder, builder.getLoc(), std::forward<decltype(isStart)>(isStart), std::forward<decltype(name)>(name));
}

void RecordOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool isStart, ::llvm::StringRef name) {
  if (isStart) {
    odsState.getOrAddProperties<Properties>().isStart = ((isStart) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

RecordOp RecordOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, /*optional*/bool isStart, ::llvm::StringRef name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(isStart)>(isStart), std::forward<decltype(name)>(name));
  auto __res__ = ::llvm::dyn_cast<RecordOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RecordOp RecordOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, /*optional*/bool isStart, ::llvm::StringRef name) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(isStart)>(isStart), std::forward<decltype(name)>(name));
}

void RecordOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<RecordOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

RecordOp RecordOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<RecordOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RecordOp RecordOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void RecordOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RecordOp RecordOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<RecordOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

RecordOp RecordOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult RecordOp::verifyInvariantsImpl() {
  auto tblgen_isStart = getProperties().isStart; (void)tblgen_isStart;
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ProtonOps1(*this, tblgen_isStart, "isStart")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ProtonOps2(*this, tblgen_name, "name")))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::LogicalResult RecordOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult RecordOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  if (::mlir::succeeded(parser.parseOptionalKeyword("start"))) {
    result.getOrAddProperties<RecordOp::Properties>().isStart = parser.getBuilder().getUnitAttr();  } else {
  if (parser.parseKeyword("end"))
    return ::mlir::failure();
  }

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<RecordOp::Properties>().name = nameAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void RecordOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((getIsStartAttr() && getIsStartAttr() != ((false) ? ::mlir::OpBuilder((*this)->getContext()).getUnitAttr() : nullptr))) {
    _odsPrinter << ' ' << "start";
  } else {
    _odsPrinter << ' ' << "end";
  }
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("isStart");
  elidedAttrs.push_back("name");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIsStartAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("isStart");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void RecordOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(), 0, false, ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(::mlir::MemoryEffects::Write::get(), 0, false, ::mlir::SideEffects::DefaultResource::get());
}


} // namespace mlir::triton::proton
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::proton::RecordOp)


#endif // GET_OP_CLASSES

