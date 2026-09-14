/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::triton::amdgpu::SetFP8ClampingAttr,
::mlir::triton::amdgpu::SchedHintAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::amdgpu::SetFP8ClampingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::amdgpu::SetFP8ClampingAttr::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::amdgpu::SchedHintAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::amdgpu::SchedHintAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::triton::amdgpu::SetFP8ClampingAttr>([&](auto t) {
      printer << ::mlir::triton::amdgpu::SetFP8ClampingAttr::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::triton::amdgpu::SchedHintAttr>([&](auto t) {
      printer << ::mlir::triton::amdgpu::SchedHintAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::triton::amdgpu {


} // namespace mlir::triton::amdgpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::amdgpu::SetFP8ClampingAttr)
namespace mlir::triton::amdgpu {

namespace detail {

struct SchedHintAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::triton::amdgpu::SchedHint>;
  SchedHintAttrStorage(::mlir::triton::amdgpu::SchedHint value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SchedHintAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<SchedHintAttrStorage>()) SchedHintAttrStorage(std::move(value));
  }

  ::mlir::triton::amdgpu::SchedHint value;
};

} // namespace detail
SchedHintAttr SchedHintAttr::get(::mlir::MLIRContext *context, ::mlir::triton::amdgpu::SchedHint value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute SchedHintAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::triton::amdgpu::SchedHint> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::triton::amdgpu::SchedHint> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::triton::amdgpu::symbolizeSchedHint(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::triton::amdgpu::SchedHint" << " to be one of: " << "none" << ", " << "attention")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TritonAMDGPU_SchedHintVariantAttr parameter 'value' which is to be a `::mlir::triton::amdgpu::SchedHint`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return SchedHintAttr::get(odsParser.getContext(),
      ::mlir::triton::amdgpu::SchedHint((*_result_value)));
}

void SchedHintAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifySchedHint(getValue());
  odsPrinter << ">";
}

::mlir::triton::amdgpu::SchedHint SchedHintAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::triton::amdgpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::amdgpu::SchedHintAttr)
namespace mlir::triton::amdgpu {


/// Parse an attribute registered to this dialect.
::mlir::Attribute TritonAMDGPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void TritonAMDGPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::triton::amdgpu

#endif // GET_ATTRDEF_CLASSES

