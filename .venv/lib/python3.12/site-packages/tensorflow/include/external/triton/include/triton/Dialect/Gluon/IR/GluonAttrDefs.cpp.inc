/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::triton::gluon::AutoEncodingAttr,
::mlir::triton::gluon::CoalescedEncodingAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::gluon::AutoEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gluon::AutoEncodingAttr::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::gluon::CoalescedEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::gluon::CoalescedEncodingAttr::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::triton::gluon::AutoEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gluon::AutoEncodingAttr::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::triton::gluon::CoalescedEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::gluon::CoalescedEncodingAttr::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::triton::gluon {


} // namespace mlir::triton::gluon
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gluon::AutoEncodingAttr)
namespace mlir::triton::gluon {


} // namespace mlir::triton::gluon
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gluon::CoalescedEncodingAttr)
namespace mlir::triton::gluon {


/// Parse an attribute registered to this dialect.
::mlir::Attribute GluonDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void GluonDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::triton::gluon

#endif // GET_ATTRDEF_CLASSES

