/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::triton::nvws::TypeArrayAttr,
::mlir::triton::nvws::IntArrayAttr,
::mlir::triton::nvws::AsyncOpAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::nvws::TypeArrayAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::nvws::TypeArrayAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::nvws::IntArrayAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::nvws::IntArrayAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::nvws::AsyncOpAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::nvws::AsyncOpAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::triton::nvws::TypeArrayAttr>([&](auto t) {
      printer << ::mlir::triton::nvws::TypeArrayAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::nvws::IntArrayAttr>([&](auto t) {
      printer << ::mlir::triton::nvws::IntArrayAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::nvws::AsyncOpAttr>([&](auto t) {
      printer << ::mlir::triton::nvws::AsyncOpAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::triton::nvws {

namespace detail {

struct TypeArrayAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<Type>>;
  TypeArrayAttrStorage(::llvm::ArrayRef<Type> value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (::llvm::ArrayRef<Type>(value) == ::llvm::ArrayRef<Type>(std::get<0>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static TypeArrayAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    value = allocator.copyInto(value);
    return new (allocator.allocate<TypeArrayAttrStorage>()) TypeArrayAttrStorage(std::move(value));
  }

  ::llvm::ArrayRef<Type> value;
};

} // namespace detail
TypeArrayAttr TypeArrayAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Type> value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute TypeArrayAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<Type>> _result_value;
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};
  // Parse literal ']'
  if (odsParser.parseOptionalRSquare()) {

    // Parse variable 'value'
    _result_value = ::mlir::FieldParser<::llvm::SmallVector<Type>>::parse(odsParser);
    if (::mlir::failed(_result_value)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVWS_TypeArray parameter 'value' which is to be a `::llvm::ArrayRef<Type>`");
      return {};
    }
    // Parse literal ']'
    if (odsParser.parseRSquare()) return {};
  } else {
  }
  return TypeArrayAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<Type>((_result_value.value_or(::llvm::SmallVector<Type>()))));
}

void TypeArrayAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "[";
  if (!(!(::llvm::ArrayRef<Type>(getValue()) == ::llvm::ArrayRef<Type>(::llvm::SmallVector<Type>())))) {
    odsPrinter << "]";
  } else {
    if (!(::llvm::ArrayRef<Type>(getValue()) == ::llvm::ArrayRef<Type>(::llvm::SmallVector<Type>()))) {
      odsPrinter.printStrippedAttrOrType(getValue());
    }
    odsPrinter << "]";
  }
}

::llvm::ArrayRef<Type> TypeArrayAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::triton::nvws
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::TypeArrayAttr)
namespace mlir::triton::nvws {

namespace detail {

struct IntArrayAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int>>;
  IntArrayAttrStorage(::llvm::ArrayRef<int> value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (::llvm::ArrayRef<int>(value) == ::llvm::ArrayRef<int>(std::get<0>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static IntArrayAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    value = allocator.copyInto(value);
    return new (allocator.allocate<IntArrayAttrStorage>()) IntArrayAttrStorage(std::move(value));
  }

  ::llvm::ArrayRef<int> value;
};

} // namespace detail
IntArrayAttr IntArrayAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int> value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute IntArrayAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<int>> _result_value;
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};
  // Parse literal ']'
  if (odsParser.parseOptionalRSquare()) {

    // Parse variable 'value'
    _result_value = ::mlir::FieldParser<::llvm::SmallVector<int>>::parse(odsParser);
    if (::mlir::failed(_result_value)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVWS_IntArray parameter 'value' which is to be a `::llvm::ArrayRef<int>`");
      return {};
    }
    // Parse literal ']'
    if (odsParser.parseRSquare()) return {};
  } else {
  }
  return IntArrayAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<int>((_result_value.value_or(::llvm::SmallVector<int>()))));
}

void IntArrayAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "[";
  if (!(!(::llvm::ArrayRef<int>(getValue()) == ::llvm::ArrayRef<int>(::llvm::SmallVector<int>())))) {
    odsPrinter << "]";
  } else {
    if (!(::llvm::ArrayRef<int>(getValue()) == ::llvm::ArrayRef<int>(::llvm::SmallVector<int>()))) {
      odsPrinter.printStrippedAttrOrType(getValue());
    }
    odsPrinter << "]";
  }
}

::llvm::ArrayRef<int> IntArrayAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::triton::nvws
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::IntArrayAttr)
namespace mlir::triton::nvws {

namespace detail {

struct AsyncOpAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::triton::nvws::AsyncOp>;
  AsyncOpAttrStorage(::mlir::triton::nvws::AsyncOp value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AsyncOpAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<AsyncOpAttrStorage>()) AsyncOpAttrStorage(std::move(value));
  }

  ::mlir::triton::nvws::AsyncOp value;
};

} // namespace detail
AsyncOpAttr AsyncOpAttr::get(::mlir::MLIRContext *context, ::mlir::triton::nvws::AsyncOp value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute AsyncOpAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::triton::nvws::AsyncOp> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::triton::nvws::AsyncOp> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::triton::nvws::symbolizeAsyncOp(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::triton::nvws::AsyncOp" << " to be one of: " << "none" << ", " << "tma_load" << ", " << "tc5mma" << ", " << "tmem_copy" << ", " << "cp_async" << ", " << "wgmma")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse NVWS_AsyncOpEnum parameter 'value' which is to be a `::mlir::triton::nvws::AsyncOp`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return AsyncOpAttr::get(odsParser.getContext(),
      ::mlir::triton::nvws::AsyncOp((*_result_value)));
}

void AsyncOpAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyAsyncOp(getValue());
  odsPrinter << ">";
}

::mlir::triton::nvws::AsyncOp AsyncOpAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::triton::nvws
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::AsyncOpAttr)
namespace mlir::triton::nvws {


/// Parse an attribute registered to this dialect.
::mlir::Attribute NVWSDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void NVWSDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::triton::nvws

#endif // GET_ATTRDEF_CLASSES

