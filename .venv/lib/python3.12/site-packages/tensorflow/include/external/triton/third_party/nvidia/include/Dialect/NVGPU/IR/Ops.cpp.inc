/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVGPUOps.td                                                          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::triton::nvgpu::ClusterCTAIdOp,
::mlir::triton::nvgpu::LoadAcquireOp,
::mlir::triton::nvgpu::TensorMemoryBaseAddress,
::mlir::triton::nvgpu::WGMMAOp,
::mlir::triton::nvgpu::WGMMAWaitGroupOp
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_NVGPUOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_NVGPUOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::LLVM::LLVMPointerType>(type))) && ((::llvm::cast<::mlir::LLVM::LLVMPointerType>(type).getAddressSpace() == 1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer in address space 1, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_NVGPUOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_NVGPUOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::Float8E4M3FNType>(type))) || ((::llvm::isa<::mlir::Float8E4M3FNUZType>(type))) || ((::llvm::isa<::mlir::Float8E5M2Type>(type))) || ((::llvm::isa<::mlir::Float8E5M2FNUZType>(type))) || ((type.isF16())) || ((::llvm::isa<::mlir::BFloat16Type>(type))) || ((type.isF32())) || ((type.isF64()))) || (((type.isSignlessInteger(1))) || ((type.isSignlessInteger(8))) || ((type.isSignlessInteger(16))) || ((type.isSignlessInteger(32))) || ((type.isSignlessInteger(64)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be floating-point or integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_NVGPUOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::LLVM::LLVMPointerType>(type))) && ((::llvm::cast<::mlir::LLVM::LLVMPointerType>(type).getAddressSpace() == 6)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer in address space 6, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_NVGPUOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::LLVM::LLVMStructType>(type))) || ((type.isSignlessInteger(64))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be wgmma operand A/B type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_NVGPUOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::LLVM::LLVMStructType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM structure type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::nvgpu::MemSemanticAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3, 4";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_NVGPUOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::nvgpu::MemSyncScopeAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_NVGPUOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_NVGPUOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: wgmma operand type, either 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', or 'f32'";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_NVGPUOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps5(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::triton::nvgpu::WGMMALayoutAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: wgmma layout, either 'row' or 'col'";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_NVGPUOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_NVGPUOps5(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::ClusterCTAIdOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ClusterCTAIdOpGenericAdaptorBase::ClusterCTAIdOpGenericAdaptorBase(ClusterCTAIdOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
ClusterCTAIdOpAdaptor::ClusterCTAIdOpAdaptor(ClusterCTAIdOp op) : ClusterCTAIdOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ClusterCTAIdOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult ClusterCTAIdOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute ClusterCTAIdOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ClusterCTAIdOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> ClusterCTAIdOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void ClusterCTAIdOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void ClusterCTAIdOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult ClusterCTAIdOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void ClusterCTAIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result) {
  odsState.addTypes(result);
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result));
  auto __res__ = ::llvm::dyn_cast<ClusterCTAIdOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result));
}

void ClusterCTAIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ClusterCTAIdOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::OpBuilder &builder, ::mlir::Location location) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__);
  auto __res__ = ::llvm::dyn_cast<ClusterCTAIdOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::ImplicitLocOpBuilder &builder) {
  return create(builder, builder.getLoc());
}

void ClusterCTAIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes));
  auto __res__ = ::llvm::dyn_cast<ClusterCTAIdOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes));
}

void ClusterCTAIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ClusterCTAIdOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ClusterCTAIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ClusterCTAIdOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<ClusterCTAIdOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ClusterCTAIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ClusterCTAIdOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

void ClusterCTAIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties &>(properties));
  odsState.addAttributes(discardableAttributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ClusterCTAIdOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<ClusterCTAIdOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

ClusterCTAIdOp ClusterCTAIdOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ClusterCTAIdOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ClusterCTAIdOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult ClusterCTAIdOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIntegerType(32);
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ClusterCTAIdOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void ClusterCTAIdOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ClusterCTAIdOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::nvgpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::ClusterCTAIdOp)

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::LoadAcquireOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

LoadAcquireOpGenericAdaptorBase::LoadAcquireOpGenericAdaptorBase(LoadAcquireOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> LoadAcquireOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::triton::nvgpu::MemSemantic LoadAcquireOpGenericAdaptorBase::getSem() {
  auto attr = getSemAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::MemSyncScope LoadAcquireOpGenericAdaptorBase::getScope() {
  auto attr = getScopeAttr();
  return attr.getValue();
}


} // namespace detail
LoadAcquireOpAdaptor::LoadAcquireOpAdaptor(LoadAcquireOp op) : LoadAcquireOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoadAcquireOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_scope = getProperties().scope; (void)tblgen_scope;
  if (!tblgen_scope) return emitError(loc, "'nvg.ld_acquire' op requires attribute 'scope'");
  auto tblgen_sem = getProperties().sem; (void)tblgen_sem;
  if (!tblgen_sem) return emitError(loc, "'nvg.ld_acquire' op requires attribute 'sem'");

  if (tblgen_sem && !((::llvm::isa<::mlir::triton::nvgpu::MemSemanticAttr>(tblgen_sem))))
    return emitError(loc, "'nvg.ld_acquire' op attribute 'sem' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3, 4");

  if (tblgen_scope && !((::llvm::isa<::mlir::triton::nvgpu::MemSyncScopeAttr>(tblgen_scope))))
    return emitError(loc, "'nvg.ld_acquire' op attribute 'scope' failed to satisfy constraint: allowed 32-bit signless integer cases: 1, 2, 3");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoadAcquireOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::MutableOperandRange LoadAcquireOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult LoadAcquireOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.scope;
       auto attr = dict.get("scope");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `scope` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.sem;
       auto attr = dict.get("sem");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `sem` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute LoadAcquireOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.scope;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("scope",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sem;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sem",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code LoadAcquireOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.scope.getAsOpaquePointer()), 
    llvm::hash_value(prop.sem.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> LoadAcquireOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "scope")
      return prop.scope;

    if (name == "sem")
      return prop.sem;
  return std::nullopt;
}

void LoadAcquireOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "scope") {
       prop.scope = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.scope)>>(value);
       return;
    }

    if (name == "sem") {
       prop.sem = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.sem)>>(value);
       return;
    }
}

void LoadAcquireOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.scope) attrs.append("scope", prop.scope);

    if (prop.sem) attrs.append("sem", prop.sem);
}

::llvm::LogicalResult LoadAcquireOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getScopeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps2(attr, "scope", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSemAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps1(attr, "sem", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult LoadAcquireOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.scope)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.sem)))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadAcquireOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.scope);
  writer.writeAttribute(prop.sem);
}

::mlir::triton::nvgpu::MemSemantic LoadAcquireOp::getSem() {
  auto attr = getSemAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::MemSyncScope LoadAcquireOp::getScope() {
  auto attr = getScopeAttr();
  return attr.getValue();
}

void LoadAcquireOp::setSem(::mlir::triton::nvgpu::MemSemantic attrValue) {
  getProperties().sem = ::mlir::triton::nvgpu::MemSemanticAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void LoadAcquireOp::setScope(::mlir::triton::nvgpu::MemSyncScope attrValue) {
  getProperties().scope = ::mlir::triton::nvgpu::MemSyncScopeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void LoadAcquireOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope) {
  odsState.addOperands(addr);
  if (mask)
    odsState.addOperands(mask);
  odsState.getOrAddProperties<Properties>().sem = sem;
  odsState.getOrAddProperties<Properties>().scope = scope;
  odsState.addTypes(result);
}

LoadAcquireOp LoadAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(addr)>(addr), std::forward<decltype(mask)>(mask), std::forward<decltype(sem)>(sem), std::forward<decltype(scope)>(scope));
  auto __res__ = ::llvm::dyn_cast<LoadAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LoadAcquireOp LoadAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(addr)>(addr), std::forward<decltype(mask)>(mask), std::forward<decltype(sem)>(sem), std::forward<decltype(scope)>(scope));
}

void LoadAcquireOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope) {
  odsState.addOperands(addr);
  if (mask)
    odsState.addOperands(mask);
  odsState.getOrAddProperties<Properties>().sem = sem;
  odsState.getOrAddProperties<Properties>().scope = scope;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

LoadAcquireOp LoadAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(addr)>(addr), std::forward<decltype(mask)>(mask), std::forward<decltype(sem)>(sem), std::forward<decltype(scope)>(scope));
  auto __res__ = ::llvm::dyn_cast<LoadAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LoadAcquireOp LoadAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemanticAttr sem, ::mlir::triton::nvgpu::MemSyncScopeAttr scope) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(addr)>(addr), std::forward<decltype(mask)>(mask), std::forward<decltype(sem)>(sem), std::forward<decltype(scope)>(scope));
}

void LoadAcquireOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope) {
  odsState.addOperands(addr);
  if (mask)
    odsState.addOperands(mask);
  odsState.getOrAddProperties<Properties>().sem = ::mlir::triton::nvgpu::MemSemanticAttr::get(odsBuilder.getContext(), sem);
  odsState.getOrAddProperties<Properties>().scope = ::mlir::triton::nvgpu::MemSyncScopeAttr::get(odsBuilder.getContext(), scope);
  odsState.addTypes(result);
}

LoadAcquireOp LoadAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(addr)>(addr), std::forward<decltype(mask)>(mask), std::forward<decltype(sem)>(sem), std::forward<decltype(scope)>(scope));
  auto __res__ = ::llvm::dyn_cast<LoadAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LoadAcquireOp LoadAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(addr)>(addr), std::forward<decltype(mask)>(mask), std::forward<decltype(sem)>(sem), std::forward<decltype(scope)>(scope));
}

void LoadAcquireOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope) {
  odsState.addOperands(addr);
  if (mask)
    odsState.addOperands(mask);
  odsState.getOrAddProperties<Properties>().sem = ::mlir::triton::nvgpu::MemSemanticAttr::get(odsBuilder.getContext(), sem);
  odsState.getOrAddProperties<Properties>().scope = ::mlir::triton::nvgpu::MemSyncScopeAttr::get(odsBuilder.getContext(), scope);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

LoadAcquireOp LoadAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(addr)>(addr), std::forward<decltype(mask)>(mask), std::forward<decltype(sem)>(sem), std::forward<decltype(scope)>(scope));
  auto __res__ = ::llvm::dyn_cast<LoadAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LoadAcquireOp LoadAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::Value mask, ::mlir::triton::nvgpu::MemSemantic sem, ::mlir::triton::nvgpu::MemSyncScope scope) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(addr)>(addr), std::forward<decltype(mask)>(mask), std::forward<decltype(sem)>(sem), std::forward<decltype(scope)>(scope));
}

void LoadAcquireOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<LoadAcquireOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

LoadAcquireOp LoadAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<LoadAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LoadAcquireOp LoadAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void LoadAcquireOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LoadAcquireOp LoadAcquireOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<LoadAcquireOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

LoadAcquireOp LoadAcquireOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult LoadAcquireOp::verifyInvariantsImpl() {
  auto tblgen_scope = getProperties().scope; (void)tblgen_scope;
  if (!tblgen_scope) return emitOpError("requires attribute 'scope'");
  auto tblgen_sem = getProperties().sem; (void)tblgen_sem;
  if (!tblgen_sem) return emitOpError("requires attribute 'sem'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps1(*this, tblgen_sem, "sem")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps2(*this, tblgen_scope, "scope")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoadAcquireOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoadAcquireOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::triton::nvgpu::MemSemanticAttr semAttr;
  ::mlir::triton::nvgpu::MemSyncScopeAttr scopeAttr;
  ::mlir::OpAsmParser::UnresolvedOperand addrRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addrOperands(&addrRawOperand, 1);  ::llvm::SMLoc addrOperandsLoc;
  (void)addrOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> maskOperands;
  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> addrTypes;
  ::llvm::ArrayRef<::mlir::Type> resultTypes;

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"relaxed","acquire","release","acq_rel"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "sem", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'sem' [relaxed, acquire, release, acq_rel]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::nvgpu::symbolizeMemSemantic(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "sem attribute specification: \"" << attrStr << '"';;

      semAttr = ::mlir::triton::nvgpu::MemSemanticAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<LoadAcquireOp::Properties>().sem = semAttr;
    }
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"gpu","cta","sys"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "scope", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'scope' [gpu, cta, sys]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::triton::nvgpu::symbolizeMemSyncScope(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "scope attribute specification: \"" << attrStr << '"';;

      scopeAttr = ::mlir::triton::nvgpu::MemSyncScopeAttr::get(parser.getBuilder().getContext(), *attrOptional);
        result.getOrAddProperties<LoadAcquireOp::Properties>().scope = scopeAttr;
    }
  }
  if (parser.parseComma())
    return ::mlir::failure();

  addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addrRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    maskOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      maskOperands.push_back(operand);
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

  ::mlir::FunctionType addr__result_functionType;
  if (parser.parseType(addr__result_functionType))
    return ::mlir::failure();
  addrTypes = addr__result_functionType.getInputs();
  resultTypes = addr__result_functionType.getResults();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(addrOperands, addrTypes, addrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, odsBuildableType0, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadAcquireOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';

  {
    auto caseValue = getSem();
    auto caseValueStr = stringifyMemSemantic(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';

  {
    auto caseValue = getScope();
    auto caseValueStr = stringifyMemSyncScope(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getAddr();
  if (getMask()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getMask())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sem");
  elidedAttrs.push_back("scope");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getAddr().getType()), ::llvm::ArrayRef<::mlir::Type>(getResult().getType()));
}

void LoadAcquireOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(), 0, false, ::mlir::SideEffects::DefaultResource::get());
}


} // namespace mlir::triton::nvgpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::LoadAcquireOp)

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::TensorMemoryBaseAddress definitions
//===----------------------------------------------------------------------===//

namespace detail {

TensorMemoryBaseAddressGenericAdaptorBase::TensorMemoryBaseAddressGenericAdaptorBase(TensorMemoryBaseAddress op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
TensorMemoryBaseAddressAdaptor::TensorMemoryBaseAddressAdaptor(TensorMemoryBaseAddress op) : TensorMemoryBaseAddressGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult TensorMemoryBaseAddressAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult TensorMemoryBaseAddress::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute TensorMemoryBaseAddress::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code TensorMemoryBaseAddress::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> TensorMemoryBaseAddress::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void TensorMemoryBaseAddress::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void TensorMemoryBaseAddress::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
}

::llvm::LogicalResult TensorMemoryBaseAddress::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void TensorMemoryBaseAddress::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result) {
  odsState.addTypes(result);
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result));
  auto __res__ = ::llvm::dyn_cast<TensorMemoryBaseAddress>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result));
}

void TensorMemoryBaseAddress::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(TensorMemoryBaseAddress::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::OpBuilder &builder, ::mlir::Location location) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__);
  auto __res__ = ::llvm::dyn_cast<TensorMemoryBaseAddress>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::ImplicitLocOpBuilder &builder) {
  return create(builder, builder.getLoc());
}

void TensorMemoryBaseAddress::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes));
  auto __res__ = ::llvm::dyn_cast<TensorMemoryBaseAddress>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes));
}

void TensorMemoryBaseAddress::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<TensorMemoryBaseAddress>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void TensorMemoryBaseAddress::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(TensorMemoryBaseAddress::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<TensorMemoryBaseAddress>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void TensorMemoryBaseAddress::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<TensorMemoryBaseAddress>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

void TensorMemoryBaseAddress::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties &>(properties));
  odsState.addAttributes(discardableAttributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(TensorMemoryBaseAddress::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<TensorMemoryBaseAddress>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

TensorMemoryBaseAddress TensorMemoryBaseAddress::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult TensorMemoryBaseAddress::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult TensorMemoryBaseAddress::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult TensorMemoryBaseAddress::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getType<::mlir::LLVM::LLVMPointerType>(6);
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult TensorMemoryBaseAddress::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::LLVM::LLVMPointerType>(6);
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TensorMemoryBaseAddress::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void TensorMemoryBaseAddress::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::triton::nvgpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::TensorMemoryBaseAddress)

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::WGMMAOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

WGMMAOpGenericAdaptorBase::WGMMAOpGenericAdaptorBase(WGMMAOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

std::pair<unsigned, unsigned> WGMMAOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

uint32_t WGMMAOpGenericAdaptorBase::getM() {
  auto attr = getMAttr();
  return attr.getValue().getZExtValue();
}

uint32_t WGMMAOpGenericAdaptorBase::getN() {
  auto attr = getNAttr();
  return attr.getValue().getZExtValue();
}

uint32_t WGMMAOpGenericAdaptorBase::getK() {
  auto attr = getKAttr();
  return attr.getValue().getZExtValue();
}

::mlir::triton::nvgpu::WGMMAEltType WGMMAOpGenericAdaptorBase::getEltTypeC() {
  auto attr = getEltTypeCAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::WGMMAEltType WGMMAOpGenericAdaptorBase::getEltTypeA() {
  auto attr = getEltTypeAAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::WGMMAEltType WGMMAOpGenericAdaptorBase::getEltTypeB() {
  auto attr = getEltTypeBAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::WGMMALayout WGMMAOpGenericAdaptorBase::getLayoutA() {
  auto attr = getLayoutAAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::WGMMALayout WGMMAOpGenericAdaptorBase::getLayoutB() {
  auto attr = getLayoutBAttr();
  return attr.getValue();
}


} // namespace detail
WGMMAOpAdaptor::WGMMAOpAdaptor(WGMMAOp op) : WGMMAOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult WGMMAOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_eltTypeA = getProperties().eltTypeA; (void)tblgen_eltTypeA;
  if (!tblgen_eltTypeA) return emitError(loc, "'nvg.wgmma' op requires attribute 'eltTypeA'");
  auto tblgen_eltTypeB = getProperties().eltTypeB; (void)tblgen_eltTypeB;
  if (!tblgen_eltTypeB) return emitError(loc, "'nvg.wgmma' op requires attribute 'eltTypeB'");
  auto tblgen_eltTypeC = getProperties().eltTypeC; (void)tblgen_eltTypeC;
  if (!tblgen_eltTypeC) return emitError(loc, "'nvg.wgmma' op requires attribute 'eltTypeC'");
  auto tblgen_k = getProperties().k; (void)tblgen_k;
  if (!tblgen_k) return emitError(loc, "'nvg.wgmma' op requires attribute 'k'");
  auto tblgen_layoutA = getProperties().layoutA; (void)tblgen_layoutA;
  if (!tblgen_layoutA) return emitError(loc, "'nvg.wgmma' op requires attribute 'layoutA'");
  auto tblgen_layoutB = getProperties().layoutB; (void)tblgen_layoutB;
  if (!tblgen_layoutB) return emitError(loc, "'nvg.wgmma' op requires attribute 'layoutB'");
  auto tblgen_m = getProperties().m; (void)tblgen_m;
  if (!tblgen_m) return emitError(loc, "'nvg.wgmma' op requires attribute 'm'");
  auto tblgen_n = getProperties().n; (void)tblgen_n;
  if (!tblgen_n) return emitError(loc, "'nvg.wgmma' op requires attribute 'n'");

  if (tblgen_m && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_m))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_m).getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvg.wgmma' op attribute 'm' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_n && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_n))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_n).getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvg.wgmma' op attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_k && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_k))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_k).getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvg.wgmma' op attribute 'k' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_eltTypeC && !((::llvm::isa<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(tblgen_eltTypeC))))
    return emitError(loc, "'nvg.wgmma' op attribute 'eltTypeC' failed to satisfy constraint: wgmma operand type, either 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', or 'f32'");

  if (tblgen_eltTypeA && !((::llvm::isa<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(tblgen_eltTypeA))))
    return emitError(loc, "'nvg.wgmma' op attribute 'eltTypeA' failed to satisfy constraint: wgmma operand type, either 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', or 'f32'");

  if (tblgen_eltTypeB && !((::llvm::isa<::mlir::triton::nvgpu::WGMMAEltTypeAttr>(tblgen_eltTypeB))))
    return emitError(loc, "'nvg.wgmma' op attribute 'eltTypeB' failed to satisfy constraint: wgmma operand type, either 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', or 'f32'");

  if (tblgen_layoutA && !((::llvm::isa<::mlir::triton::nvgpu::WGMMALayoutAttr>(tblgen_layoutA))))
    return emitError(loc, "'nvg.wgmma' op attribute 'layoutA' failed to satisfy constraint: wgmma layout, either 'row' or 'col'");

  if (tblgen_layoutB && !((::llvm::isa<::mlir::triton::nvgpu::WGMMALayoutAttr>(tblgen_layoutB))))
    return emitError(loc, "'nvg.wgmma' op attribute 'layoutB' failed to satisfy constraint: wgmma layout, either 'row' or 'col'");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WGMMAOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange WGMMAOp::getOpCMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::LogicalResult WGMMAOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.eltTypeA;
       auto attr = dict.get("eltTypeA");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `eltTypeA` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.eltTypeB;
       auto attr = dict.get("eltTypeB");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `eltTypeB` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.eltTypeC;
       auto attr = dict.get("eltTypeC");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `eltTypeC` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.k;
       auto attr = dict.get("k");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `k` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.layoutA;
       auto attr = dict.get("layoutA");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `layoutA` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.layoutB;
       auto attr = dict.get("layoutB");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `layoutB` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.m;
       auto attr = dict.get("m");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `m` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.n;
       auto attr = dict.get("n");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `n` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute WGMMAOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.eltTypeA;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("eltTypeA",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.eltTypeB;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("eltTypeB",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.eltTypeC;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("eltTypeC",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.k;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("k",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.layoutA;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("layoutA",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.layoutB;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("layoutB",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.m;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("m",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.n;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("n",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code WGMMAOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.eltTypeA.getAsOpaquePointer()), 
    llvm::hash_value(prop.eltTypeB.getAsOpaquePointer()), 
    llvm::hash_value(prop.eltTypeC.getAsOpaquePointer()), 
    llvm::hash_value(prop.k.getAsOpaquePointer()), 
    llvm::hash_value(prop.layoutA.getAsOpaquePointer()), 
    llvm::hash_value(prop.layoutB.getAsOpaquePointer()), 
    llvm::hash_value(prop.m.getAsOpaquePointer()), 
    llvm::hash_value(prop.n.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> WGMMAOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "eltTypeA")
      return prop.eltTypeA;

    if (name == "eltTypeB")
      return prop.eltTypeB;

    if (name == "eltTypeC")
      return prop.eltTypeC;

    if (name == "k")
      return prop.k;

    if (name == "layoutA")
      return prop.layoutA;

    if (name == "layoutB")
      return prop.layoutB;

    if (name == "m")
      return prop.m;

    if (name == "n")
      return prop.n;
  return std::nullopt;
}

void WGMMAOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "eltTypeA") {
       prop.eltTypeA = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.eltTypeA)>>(value);
       return;
    }

    if (name == "eltTypeB") {
       prop.eltTypeB = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.eltTypeB)>>(value);
       return;
    }

    if (name == "eltTypeC") {
       prop.eltTypeC = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.eltTypeC)>>(value);
       return;
    }

    if (name == "k") {
       prop.k = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.k)>>(value);
       return;
    }

    if (name == "layoutA") {
       prop.layoutA = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.layoutA)>>(value);
       return;
    }

    if (name == "layoutB") {
       prop.layoutB = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.layoutB)>>(value);
       return;
    }

    if (name == "m") {
       prop.m = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.m)>>(value);
       return;
    }

    if (name == "n") {
       prop.n = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.n)>>(value);
       return;
    }
}

void WGMMAOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.eltTypeA) attrs.append("eltTypeA", prop.eltTypeA);

    if (prop.eltTypeB) attrs.append("eltTypeB", prop.eltTypeB);

    if (prop.eltTypeC) attrs.append("eltTypeC", prop.eltTypeC);

    if (prop.k) attrs.append("k", prop.k);

    if (prop.layoutA) attrs.append("layoutA", prop.layoutA);

    if (prop.layoutB) attrs.append("layoutB", prop.layoutB);

    if (prop.m) attrs.append("m", prop.m);

    if (prop.n) attrs.append("n", prop.n);
}

::llvm::LogicalResult WGMMAOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getEltTypeAAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps4(attr, "eltTypeA", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getEltTypeBAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps4(attr, "eltTypeB", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getEltTypeCAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps4(attr, "eltTypeC", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getKAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps3(attr, "k", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getLayoutAAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps5(attr, "layoutA", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getLayoutBAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps5(attr, "layoutB", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps3(attr, "m", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps3(attr, "n", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult WGMMAOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.eltTypeA)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.eltTypeB)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.eltTypeC)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.k)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.layoutA)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.layoutB)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.m)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.n)))
    return ::mlir::failure();
  return ::mlir::success();
}

void WGMMAOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.eltTypeA);
  writer.writeAttribute(prop.eltTypeB);
  writer.writeAttribute(prop.eltTypeC);
  writer.writeAttribute(prop.k);
  writer.writeAttribute(prop.layoutA);
  writer.writeAttribute(prop.layoutB);
  writer.writeAttribute(prop.m);
  writer.writeAttribute(prop.n);
}

uint32_t WGMMAOp::getM() {
  auto attr = getMAttr();
  return attr.getValue().getZExtValue();
}

uint32_t WGMMAOp::getN() {
  auto attr = getNAttr();
  return attr.getValue().getZExtValue();
}

uint32_t WGMMAOp::getK() {
  auto attr = getKAttr();
  return attr.getValue().getZExtValue();
}

::mlir::triton::nvgpu::WGMMAEltType WGMMAOp::getEltTypeC() {
  auto attr = getEltTypeCAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::WGMMAEltType WGMMAOp::getEltTypeA() {
  auto attr = getEltTypeAAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::WGMMAEltType WGMMAOp::getEltTypeB() {
  auto attr = getEltTypeBAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::WGMMALayout WGMMAOp::getLayoutA() {
  auto attr = getLayoutAAttr();
  return attr.getValue();
}

::mlir::triton::nvgpu::WGMMALayout WGMMAOp::getLayoutB() {
  auto attr = getLayoutBAttr();
  return attr.getValue();
}

void WGMMAOp::setM(uint32_t attrValue) {
  getProperties().m = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void WGMMAOp::setN(uint32_t attrValue) {
  getProperties().n = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void WGMMAOp::setK(uint32_t attrValue) {
  getProperties().k = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void WGMMAOp::setEltTypeC(::mlir::triton::nvgpu::WGMMAEltType attrValue) {
  getProperties().eltTypeC = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void WGMMAOp::setEltTypeA(::mlir::triton::nvgpu::WGMMAEltType attrValue) {
  getProperties().eltTypeA = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void WGMMAOp::setEltTypeB(::mlir::triton::nvgpu::WGMMAEltType attrValue) {
  getProperties().eltTypeB = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void WGMMAOp::setLayoutA(::mlir::triton::nvgpu::WGMMALayout attrValue) {
  getProperties().layoutA = ::mlir::triton::nvgpu::WGMMALayoutAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void WGMMAOp::setLayoutB(::mlir::triton::nvgpu::WGMMALayout attrValue) {
  getProperties().layoutB = ::mlir::triton::nvgpu::WGMMALayoutAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue);
}

void WGMMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB) {
  odsState.addOperands(opA);
  odsState.addOperands(opB);
  odsState.addOperands(useC);
  if (opC)
    odsState.addOperands(opC);
  odsState.getOrAddProperties<Properties>().m = m;
  odsState.getOrAddProperties<Properties>().n = n;
  odsState.getOrAddProperties<Properties>().k = k;
  odsState.getOrAddProperties<Properties>().eltTypeC = eltTypeC;
  odsState.getOrAddProperties<Properties>().eltTypeA = eltTypeA;
  odsState.getOrAddProperties<Properties>().eltTypeB = eltTypeB;
  odsState.getOrAddProperties<Properties>().layoutA = layoutA;
  odsState.getOrAddProperties<Properties>().layoutB = layoutB;
  odsState.addTypes(res);
}

WGMMAOp WGMMAOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(res)>(res), std::forward<decltype(opA)>(opA), std::forward<decltype(opB)>(opB), std::forward<decltype(useC)>(useC), std::forward<decltype(opC)>(opC), std::forward<decltype(m)>(m), std::forward<decltype(n)>(n), std::forward<decltype(k)>(k), std::forward<decltype(eltTypeC)>(eltTypeC), std::forward<decltype(eltTypeA)>(eltTypeA), std::forward<decltype(eltTypeB)>(eltTypeB), std::forward<decltype(layoutA)>(layoutA), std::forward<decltype(layoutB)>(layoutB));
  auto __res__ = ::llvm::dyn_cast<WGMMAOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAOp WGMMAOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB) {
  return create(builder, builder.getLoc(), std::forward<decltype(res)>(res), std::forward<decltype(opA)>(opA), std::forward<decltype(opB)>(opB), std::forward<decltype(useC)>(useC), std::forward<decltype(opC)>(opC), std::forward<decltype(m)>(m), std::forward<decltype(n)>(n), std::forward<decltype(k)>(k), std::forward<decltype(eltTypeC)>(eltTypeC), std::forward<decltype(eltTypeA)>(eltTypeA), std::forward<decltype(eltTypeB)>(eltTypeB), std::forward<decltype(layoutA)>(layoutA), std::forward<decltype(layoutB)>(layoutB));
}

void WGMMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB) {
  odsState.addOperands(opA);
  odsState.addOperands(opB);
  odsState.addOperands(useC);
  if (opC)
    odsState.addOperands(opC);
  odsState.getOrAddProperties<Properties>().m = m;
  odsState.getOrAddProperties<Properties>().n = n;
  odsState.getOrAddProperties<Properties>().k = k;
  odsState.getOrAddProperties<Properties>().eltTypeC = eltTypeC;
  odsState.getOrAddProperties<Properties>().eltTypeA = eltTypeA;
  odsState.getOrAddProperties<Properties>().eltTypeB = eltTypeB;
  odsState.getOrAddProperties<Properties>().layoutA = layoutA;
  odsState.getOrAddProperties<Properties>().layoutB = layoutB;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

WGMMAOp WGMMAOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(opA)>(opA), std::forward<decltype(opB)>(opB), std::forward<decltype(useC)>(useC), std::forward<decltype(opC)>(opC), std::forward<decltype(m)>(m), std::forward<decltype(n)>(n), std::forward<decltype(k)>(k), std::forward<decltype(eltTypeC)>(eltTypeC), std::forward<decltype(eltTypeA)>(eltTypeA), std::forward<decltype(eltTypeB)>(eltTypeB), std::forward<decltype(layoutA)>(layoutA), std::forward<decltype(layoutB)>(layoutB));
  auto __res__ = ::llvm::dyn_cast<WGMMAOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAOp WGMMAOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeC, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeA, ::mlir::triton::nvgpu::WGMMAEltTypeAttr eltTypeB, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutA, ::mlir::triton::nvgpu::WGMMALayoutAttr layoutB) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(opA)>(opA), std::forward<decltype(opB)>(opB), std::forward<decltype(useC)>(useC), std::forward<decltype(opC)>(opC), std::forward<decltype(m)>(m), std::forward<decltype(n)>(n), std::forward<decltype(k)>(k), std::forward<decltype(eltTypeC)>(eltTypeC), std::forward<decltype(eltTypeA)>(eltTypeA), std::forward<decltype(eltTypeB)>(eltTypeB), std::forward<decltype(layoutA)>(layoutA), std::forward<decltype(layoutB)>(layoutB));
}

void WGMMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB) {
  odsState.addOperands(opA);
  odsState.addOperands(opB);
  odsState.addOperands(useC);
  if (opC)
    odsState.addOperands(opC);
  odsState.getOrAddProperties<Properties>().m = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m);
  odsState.getOrAddProperties<Properties>().n = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n);
  odsState.getOrAddProperties<Properties>().k = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k);
  odsState.getOrAddProperties<Properties>().eltTypeC = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(odsBuilder.getContext(), eltTypeC);
  odsState.getOrAddProperties<Properties>().eltTypeA = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(odsBuilder.getContext(), eltTypeA);
  odsState.getOrAddProperties<Properties>().eltTypeB = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(odsBuilder.getContext(), eltTypeB);
  odsState.getOrAddProperties<Properties>().layoutA = ::mlir::triton::nvgpu::WGMMALayoutAttr::get(odsBuilder.getContext(), layoutA);
  odsState.getOrAddProperties<Properties>().layoutB = ::mlir::triton::nvgpu::WGMMALayoutAttr::get(odsBuilder.getContext(), layoutB);
  odsState.addTypes(res);
}

WGMMAOp WGMMAOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(res)>(res), std::forward<decltype(opA)>(opA), std::forward<decltype(opB)>(opB), std::forward<decltype(useC)>(useC), std::forward<decltype(opC)>(opC), std::forward<decltype(m)>(m), std::forward<decltype(n)>(n), std::forward<decltype(k)>(k), std::forward<decltype(eltTypeC)>(eltTypeC), std::forward<decltype(eltTypeA)>(eltTypeA), std::forward<decltype(eltTypeB)>(eltTypeB), std::forward<decltype(layoutA)>(layoutA), std::forward<decltype(layoutB)>(layoutB));
  auto __res__ = ::llvm::dyn_cast<WGMMAOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAOp WGMMAOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type res, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB) {
  return create(builder, builder.getLoc(), std::forward<decltype(res)>(res), std::forward<decltype(opA)>(opA), std::forward<decltype(opB)>(opB), std::forward<decltype(useC)>(useC), std::forward<decltype(opC)>(opC), std::forward<decltype(m)>(m), std::forward<decltype(n)>(n), std::forward<decltype(k)>(k), std::forward<decltype(eltTypeC)>(eltTypeC), std::forward<decltype(eltTypeA)>(eltTypeA), std::forward<decltype(eltTypeB)>(eltTypeB), std::forward<decltype(layoutA)>(layoutA), std::forward<decltype(layoutB)>(layoutB));
}

void WGMMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB) {
  odsState.addOperands(opA);
  odsState.addOperands(opB);
  odsState.addOperands(useC);
  if (opC)
    odsState.addOperands(opC);
  odsState.getOrAddProperties<Properties>().m = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m);
  odsState.getOrAddProperties<Properties>().n = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n);
  odsState.getOrAddProperties<Properties>().k = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k);
  odsState.getOrAddProperties<Properties>().eltTypeC = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(odsBuilder.getContext(), eltTypeC);
  odsState.getOrAddProperties<Properties>().eltTypeA = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(odsBuilder.getContext(), eltTypeA);
  odsState.getOrAddProperties<Properties>().eltTypeB = ::mlir::triton::nvgpu::WGMMAEltTypeAttr::get(odsBuilder.getContext(), eltTypeB);
  odsState.getOrAddProperties<Properties>().layoutA = ::mlir::triton::nvgpu::WGMMALayoutAttr::get(odsBuilder.getContext(), layoutA);
  odsState.getOrAddProperties<Properties>().layoutB = ::mlir::triton::nvgpu::WGMMALayoutAttr::get(odsBuilder.getContext(), layoutB);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

WGMMAOp WGMMAOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(opA)>(opA), std::forward<decltype(opB)>(opB), std::forward<decltype(useC)>(useC), std::forward<decltype(opC)>(opC), std::forward<decltype(m)>(m), std::forward<decltype(n)>(n), std::forward<decltype(k)>(k), std::forward<decltype(eltTypeC)>(eltTypeC), std::forward<decltype(eltTypeA)>(eltTypeA), std::forward<decltype(eltTypeB)>(eltTypeB), std::forward<decltype(layoutA)>(layoutA), std::forward<decltype(layoutB)>(layoutB));
  auto __res__ = ::llvm::dyn_cast<WGMMAOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAOp WGMMAOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value opA, ::mlir::Value opB, ::mlir::Value useC, /*optional*/::mlir::Value opC, uint32_t m, uint32_t n, uint32_t k, ::mlir::triton::nvgpu::WGMMAEltType eltTypeC, ::mlir::triton::nvgpu::WGMMAEltType eltTypeA, ::mlir::triton::nvgpu::WGMMAEltType eltTypeB, ::mlir::triton::nvgpu::WGMMALayout layoutA, ::mlir::triton::nvgpu::WGMMALayout layoutB) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(opA)>(opA), std::forward<decltype(opB)>(opB), std::forward<decltype(useC)>(useC), std::forward<decltype(opC)>(opC), std::forward<decltype(m)>(m), std::forward<decltype(n)>(n), std::forward<decltype(k)>(k), std::forward<decltype(eltTypeC)>(eltTypeC), std::forward<decltype(eltTypeA)>(eltTypeA), std::forward<decltype(eltTypeB)>(eltTypeB), std::forward<decltype(layoutA)>(layoutA), std::forward<decltype(layoutB)>(layoutB));
}

void WGMMAOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WGMMAOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

WGMMAOp WGMMAOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<WGMMAOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAOp WGMMAOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void WGMMAOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

WGMMAOp WGMMAOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<WGMMAOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAOp WGMMAOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult WGMMAOp::verifyInvariantsImpl() {
  auto tblgen_eltTypeA = getProperties().eltTypeA; (void)tblgen_eltTypeA;
  if (!tblgen_eltTypeA) return emitOpError("requires attribute 'eltTypeA'");
  auto tblgen_eltTypeB = getProperties().eltTypeB; (void)tblgen_eltTypeB;
  if (!tblgen_eltTypeB) return emitOpError("requires attribute 'eltTypeB'");
  auto tblgen_eltTypeC = getProperties().eltTypeC; (void)tblgen_eltTypeC;
  if (!tblgen_eltTypeC) return emitOpError("requires attribute 'eltTypeC'");
  auto tblgen_k = getProperties().k; (void)tblgen_k;
  if (!tblgen_k) return emitOpError("requires attribute 'k'");
  auto tblgen_layoutA = getProperties().layoutA; (void)tblgen_layoutA;
  if (!tblgen_layoutA) return emitOpError("requires attribute 'layoutA'");
  auto tblgen_layoutB = getProperties().layoutB; (void)tblgen_layoutB;
  if (!tblgen_layoutB) return emitOpError("requires attribute 'layoutB'");
  auto tblgen_m = getProperties().m; (void)tblgen_m;
  if (!tblgen_m) return emitOpError("requires attribute 'm'");
  auto tblgen_n = getProperties().n; (void)tblgen_n;
  if (!tblgen_n) return emitOpError("requires attribute 'n'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps3(*this, tblgen_m, "m")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps3(*this, tblgen_n, "n")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps3(*this, tblgen_k, "k")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps4(*this, tblgen_eltTypeC, "eltTypeC")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps4(*this, tblgen_eltTypeA, "eltTypeA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps4(*this, tblgen_eltTypeB, "eltTypeB")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps5(*this, tblgen_layoutA, "layoutA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps5(*this, tblgen_layoutB, "layoutB")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult WGMMAOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult WGMMAOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand opARawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> opAOperands(&opARawOperand, 1);  ::llvm::SMLoc opAOperandsLoc;
  (void)opAOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand opBRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> opBOperands(&opBRawOperand, 1);  ::llvm::SMLoc opBOperandsLoc;
  (void)opBOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand useCRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> useCOperands(&useCRawOperand, 1);  ::llvm::SMLoc useCOperandsLoc;
  (void)useCOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> opCOperands;
  ::llvm::SMLoc opCOperandsLoc;
  (void)opCOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  opAOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(opARawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  opBOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(opBRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  useCOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(useCRawOperand))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    opCOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      opCOperands.push_back(operand);
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

  ::mlir::FunctionType allOperand__res_functionType;
  if (parser.parseType(allOperand__res_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__res_functionType.getInputs();
  resTypes = allOperand__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(opAOperands, opBOperands, useCOperands, opCOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WGMMAOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOpA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getOpB();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getUseC();
  if (getOpC()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getOpC())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), ::llvm::ArrayRef<::mlir::Type>(getRes().getType()));
}


} // namespace mlir::triton::nvgpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::WGMMAOp)

namespace mlir::triton::nvgpu {


//===----------------------------------------------------------------------===//
// ::mlir::triton::nvgpu::WGMMAWaitGroupOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

WGMMAWaitGroupOpGenericAdaptorBase::WGMMAWaitGroupOpGenericAdaptorBase(WGMMAWaitGroupOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t WGMMAWaitGroupOpGenericAdaptorBase::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
WGMMAWaitGroupOpAdaptor::WGMMAWaitGroupOpAdaptor(WGMMAWaitGroupOp op) : WGMMAWaitGroupOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult WGMMAWaitGroupOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitError(loc, "'nvg.wgmma_wait_group' op requires attribute 'pendings'");

  if (tblgen_pendings && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pendings))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pendings).getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvg.wgmma_wait_group' op attribute 'pendings' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult WGMMAWaitGroupOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.pendings;
       auto attr = dict.get("pendings");
    if (attr) {
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `pendings` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute WGMMAWaitGroupOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.pendings;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("pendings",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code WGMMAWaitGroupOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.pendings.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> WGMMAWaitGroupOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "pendings")
      return prop.pendings;
  return std::nullopt;
}

void WGMMAWaitGroupOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "pendings") {
       prop.pendings = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pendings)>>(value);
       return;
    }
}

void WGMMAWaitGroupOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.pendings) attrs.append("pendings", prop.pendings);
}

::llvm::LogicalResult WGMMAWaitGroupOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getPendingsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps3(attr, "pendings", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult WGMMAWaitGroupOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.pendings)))
    return ::mlir::failure();
  return ::mlir::success();
}

void WGMMAWaitGroupOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.pendings);
}

uint32_t WGMMAWaitGroupOp::getPendings() {
  auto attr = getPendingsAttr();
  return attr.getValue().getZExtValue();
}

void WGMMAWaitGroupOp::setPendings(uint32_t attrValue) {
  getProperties().pendings = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().pendings = pendings;
  odsState.addTypes(output);
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(output)>(output), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  return create(builder, builder.getLoc(), std::forward<decltype(output)>(output), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().pendings = pendings;

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(WGMMAWaitGroupOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  return create(builder, builder.getLoc(), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().pendings = pendings;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::IntegerAttr pendings) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, uint32_t pendings) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);
  odsState.addTypes(output);
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type output, ::mlir::Value input, uint32_t pendings) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(output)>(output), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type output, ::mlir::Value input, uint32_t pendings) {
  return create(builder, builder.getLoc(), std::forward<decltype(output)>(output), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, uint32_t pendings) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(WGMMAWaitGroupOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value input, uint32_t pendings) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value input, uint32_t pendings) {
  return create(builder, builder.getLoc(), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t pendings) {
  odsState.addOperands(input);
  odsState.getOrAddProperties<Properties>().pendings = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pendings);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t pendings) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value input, uint32_t pendings) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(input)>(input), std::forward<decltype(pendings)>(pendings));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WGMMAWaitGroupOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  if (!attributes.empty()) {
    ::mlir::OpaqueProperties properties =
      &odsState.getOrAddProperties<WGMMAWaitGroupOp::Properties>();
    std::optional<::mlir::RegisteredOperationName> info =
      odsState.name.getRegisteredInfo();
    if (failed(info->setOpPropertiesFromAttribute(odsState.name, properties,
        odsState.attributes.getDictionary(odsState.getContext()), nullptr)))
      ::llvm::report_fatal_error("Property conversion failed.");
  }
  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(WGMMAWaitGroupOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

void WGMMAWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties &>(properties));
  odsState.addAttributes(discardableAttributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(WGMMAWaitGroupOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = ::llvm::dyn_cast<WGMMAWaitGroupOp>(builder.create(__state__));
  assert(__res__ && "builder didn't return the right type");
  return __res__;
}

WGMMAWaitGroupOp WGMMAWaitGroupOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult WGMMAWaitGroupOp::verifyInvariantsImpl() {
  auto tblgen_pendings = getProperties().pendings; (void)tblgen_pendings;
  if (!tblgen_pendings) return emitOpError("requires attribute 'pendings'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPUOps3(*this, tblgen_pendings, "pendings")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPUOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {input, output} have same type");
  return ::mlir::success();
}

::llvm::LogicalResult WGMMAWaitGroupOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult WGMMAWaitGroupOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  if (operands.size() <= 0)
    return ::mlir::failure();
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult WGMMAWaitGroupOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
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
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  result.addTypes(inputTypes[0]);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WGMMAWaitGroupOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}


} // namespace mlir::triton::nvgpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvgpu::WGMMAWaitGroupOp)


#endif // GET_OP_CLASSES

