/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::triton::gpu::CGAEncodingAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

namespace mlir::triton::gpu {

namespace detail {

struct CGAEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<LinearLayout>;
  CGAEncodingAttrStorage(LinearLayout linearLayout) : linearLayout(std::move(linearLayout)) {}

  KeyTy getAsKey() const {
    return KeyTy(linearLayout);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (linearLayout == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CGAEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto linearLayout = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CGAEncodingAttrStorage>()) CGAEncodingAttrStorage(std::move(linearLayout));
  }

  LinearLayout linearLayout;
};

} // namespace detail
CGAEncodingAttr CGAEncodingAttr::get(::mlir::MLIRContext *context, LinearLayout linearLayout) {
  return Base::get(context, std::move(linearLayout));
}

CGAEncodingAttr CGAEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, LinearLayout linearLayout) {
  return Base::getChecked(emitError, context, std::move(linearLayout));
}

::llvm::LogicalResult CGAEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout) {
  if (::mlir::failed(verify(emitError, linearLayout)))
    return ::mlir::failure();
  return ::mlir::success();
}

const LinearLayout &CGAEncodingAttr::getLinearLayout() const {
  return getImpl()->linearLayout;
}


} // namespace mlir::triton::gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::CGAEncodingAttr)
namespace mlir::triton::gpu {


/// Parse an attribute registered to this dialect.
::mlir::Attribute TritonGPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void TritonGPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::triton::gpu

#endif // GET_ATTRDEF_CLASSES

