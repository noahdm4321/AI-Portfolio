/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::xegpu::TensorDescType,
::mlir::xegpu::NbarrierType,
::mlir::xegpu::MemDescType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::xegpu::TensorDescType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::TensorDescType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::NbarrierType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::NbarrierType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::MemDescType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::MemDescType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::xegpu::TensorDescType>([&](auto t) {
      printer << ::mlir::xegpu::TensorDescType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::NbarrierType>([&](auto t) {
      printer << ::mlir::xegpu::NbarrierType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::MemDescType>([&](auto t) {
      printer << ::mlir::xegpu::MemDescType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::xegpu {

namespace detail {

struct TensorDescTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, mlir::Type, mlir::Attribute, mlir::Attribute>;
  TensorDescTypeStorage(::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding, mlir::Attribute layout) : shape(std::move(shape)), elementType(std::move(elementType)), encoding(std::move(encoding)), layout(std::move(layout)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, encoding, layout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (encoding == std::get<2>(tblgenKey)) && (layout == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static TensorDescTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto encoding = std::move(std::get<2>(tblgenKey));
    auto layout = std::move(std::get<3>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<TensorDescTypeStorage>()) TensorDescTypeStorage(std::move(shape), std::move(elementType), std::move(encoding), std::move(layout));
  }

  ::llvm::ArrayRef<int64_t> shape;
  mlir::Type elementType;
  mlir::Attribute encoding;
  mlir::Attribute layout;
};

} // namespace detail
TensorDescType TensorDescType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding, mlir::Attribute layout) {
  return Base::get(context, std::move(shape), std::move(elementType), std::move(encoding), std::move(layout));
}

TensorDescType TensorDescType::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding, mlir::Attribute layout) {
  return Base::getChecked(emitError, context, std::move(shape), std::move(elementType), std::move(encoding), std::move(layout));
}

::llvm::LogicalResult TensorDescType::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, mlir::Attribute encoding, mlir::Attribute layout) {
  if (::mlir::failed(verify(emitError, shape, elementType, encoding, layout)))
    return ::mlir::failure();
  return ::mlir::success();
}

::llvm::ArrayRef<int64_t> TensorDescType::getShape() const {
  return getImpl()->shape;
}

mlir::Type TensorDescType::getElementType() const {
  return getImpl()->elementType;
}

mlir::Attribute TensorDescType::getEncoding() const {
  return getImpl()->encoding;
}

mlir::Attribute TensorDescType::getLayout() const {
  return getImpl()->layout;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::TensorDescType)
namespace mlir::xegpu {


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::NbarrierType)
namespace mlir::xegpu {

namespace detail {

struct MemDescTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, mlir::Type, MemLayoutAttr>;
  MemDescTypeStorage(::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, MemLayoutAttr mem_layout) : shape(std::move(shape)), elementType(std::move(elementType)), mem_layout(std::move(mem_layout)) {}

  KeyTy getAsKey() const {
    return KeyTy(shape, elementType, mem_layout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (elementType == std::get<1>(tblgenKey)) && (mem_layout == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static MemDescTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto shape = std::move(std::get<0>(tblgenKey));
    auto elementType = std::move(std::get<1>(tblgenKey));
    auto mem_layout = std::move(std::get<2>(tblgenKey));
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<MemDescTypeStorage>()) MemDescTypeStorage(std::move(shape), std::move(elementType), std::move(mem_layout));
  }

  ::llvm::ArrayRef<int64_t> shape;
  mlir::Type elementType;
  MemLayoutAttr mem_layout;
};

} // namespace detail
MemDescType MemDescType::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, mlir::Type elementType, MemLayoutAttr mem_layout) {
  return Base::get(context, std::move(shape), std::move(elementType), std::move(mem_layout));
}

::llvm::ArrayRef<int64_t> MemDescType::getShape() const {
  return getImpl()->shape;
}

mlir::Type MemDescType::getElementType() const {
  return getImpl()->elementType;
}

MemLayoutAttr MemDescType::getMemLayout() const {
  return getImpl()->mem_layout;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::MemDescType)
namespace mlir::xegpu {


/// Parse a type registered to this dialect.
::mlir::Type XeGPUDialect::parseType(::mlir::DialectAsmParser &parser) const {
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
void XeGPUDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}

} // namespace mlir::xegpu

#endif // GET_TYPEDEF_CLASSES

