/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::triton::nvws::ArefType,
::mlir::triton::nvws::TokenType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::nvws::ArefType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::nvws::ArefType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::nvws::TokenType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::nvws::TokenType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::triton::nvws::ArefType>([&](auto t) {
      printer << ::mlir::triton::nvws::ArefType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::nvws::TokenType>([&](auto t) {
      printer << ::mlir::triton::nvws::TokenType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::triton::nvws {

namespace detail {

struct ArefTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<TypeArrayAttr>;
  ArefTypeStorage(TypeArrayAttr baseType) : baseType(std::move(baseType)) {}

  KeyTy getAsKey() const {
    return KeyTy(baseType);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (baseType == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ArefTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto baseType = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ArefTypeStorage>()) ArefTypeStorage(std::move(baseType));
  }

  TypeArrayAttr baseType;
};

} // namespace detail
ArefType ArefType::get(::mlir::MLIRContext *context, TypeArrayAttr baseType) {
  return Base::get(context, std::move(baseType));
}

::mlir::Type ArefType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<TypeArrayAttr> _result_baseType;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'baseType'
  _result_baseType = ::mlir::FieldParser<TypeArrayAttr>::parse(odsParser);
  if (::mlir::failed(_result_baseType)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVWS_ArefType parameter 'baseType' which is to be a `TypeArrayAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_baseType));
  return ArefType::get(odsParser.getContext(),
      TypeArrayAttr((*_result_baseType)));
}

void ArefType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getBaseType());
  odsPrinter << ">";
}

TypeArrayAttr ArefType::getBaseType() const {
  return getImpl()->baseType;
}


} // namespace mlir::triton::nvws
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::ArefType)
namespace mlir::triton::nvws {


} // namespace mlir::triton::nvws
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::TokenType)
namespace mlir::triton::nvws {


/// Parse a type registered to this dialect.
::mlir::Type NVWSDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void NVWSDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}

} // namespace mlir::triton::nvws

#endif // GET_TYPEDEF_CLASSES

