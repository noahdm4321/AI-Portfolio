/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::xegpu::BlockTensorDescAttr,
::mlir::xegpu::ScatterTensorDescAttr,
::mlir::xegpu::MemorySpaceAttr,
::mlir::xegpu::CachePolicyAttr,
::mlir::xegpu::FenceScopeAttr,
::mlir::xegpu::LayoutAttr,
::mlir::xegpu::SliceAttr,
::mlir::xegpu::RangeAttr,
::mlir::xegpu::MemLayoutAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::xegpu::BlockTensorDescAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::BlockTensorDescAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::ScatterTensorDescAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::ScatterTensorDescAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::MemorySpaceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::MemorySpaceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::CachePolicyAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::CachePolicyAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::FenceScopeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::FenceScopeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::LayoutAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::LayoutAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::SliceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::SliceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::RangeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::RangeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xegpu::MemLayoutAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xegpu::MemLayoutAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::xegpu::BlockTensorDescAttr>([&](auto t) {
      printer << ::mlir::xegpu::BlockTensorDescAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::ScatterTensorDescAttr>([&](auto t) {
      printer << ::mlir::xegpu::ScatterTensorDescAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::MemorySpaceAttr>([&](auto t) {
      printer << ::mlir::xegpu::MemorySpaceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::CachePolicyAttr>([&](auto t) {
      printer << ::mlir::xegpu::CachePolicyAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::FenceScopeAttr>([&](auto t) {
      printer << ::mlir::xegpu::FenceScopeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::LayoutAttr>([&](auto t) {
      printer << ::mlir::xegpu::LayoutAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::SliceAttr>([&](auto t) {
      printer << ::mlir::xegpu::SliceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::RangeAttr>([&](auto t) {
      printer << ::mlir::xegpu::RangeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xegpu::MemLayoutAttr>([&](auto t) {
      printer << ::mlir::xegpu::MemLayoutAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::xegpu {

namespace detail {

struct BlockTensorDescAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<MemorySpaceAttr, IntegerAttr, BoolAttr>;
  BlockTensorDescAttrStorage(MemorySpaceAttr memory_space, IntegerAttr array_length, BoolAttr boundary_check) : memory_space(std::move(memory_space)), array_length(std::move(array_length)), boundary_check(std::move(boundary_check)) {}

  KeyTy getAsKey() const {
    return KeyTy(memory_space, array_length, boundary_check);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (memory_space == std::get<0>(tblgenKey)) && (array_length == std::get<1>(tblgenKey)) && (boundary_check == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static BlockTensorDescAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto memory_space = std::move(std::get<0>(tblgenKey));
    auto array_length = std::move(std::get<1>(tblgenKey));
    auto boundary_check = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<BlockTensorDescAttrStorage>()) BlockTensorDescAttrStorage(std::move(memory_space), std::move(array_length), std::move(boundary_check));
  }

  MemorySpaceAttr memory_space;
  IntegerAttr array_length;
  BoolAttr boundary_check;
};

} // namespace detail
BlockTensorDescAttr BlockTensorDescAttr::get(::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr array_length, BoolAttr boundary_check) {
  return Base::get(context, std::move(memory_space), std::move(array_length), std::move(boundary_check));
}

::mlir::Attribute BlockTensorDescAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<MemorySpaceAttr> _result_memory_space;
  ::mlir::FailureOr<IntegerAttr> _result_array_length;
  ::mlir::FailureOr<BoolAttr> _result_boundary_check;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_memory_space = false;
  bool _seen_array_length = false;
  bool _seen_boundary_check = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_memory_space && _paramKey == "memory_space") {
        _seen_memory_space = true;

        // Parse variable 'memory_space'
        _result_memory_space = ::mlir::FieldParser<MemorySpaceAttr>::parse(odsParser);
        if (::mlir::failed(_result_memory_space)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_BlockTensorDescAttr parameter 'memory_space' which is to be a `MemorySpaceAttr`");
          return {};
        }
      } else if (!_seen_array_length && _paramKey == "array_length") {
        _seen_array_length = true;

        // Parse variable 'array_length'
        _result_array_length = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_array_length)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_BlockTensorDescAttr parameter 'array_length' which is to be a `IntegerAttr`");
          return {};
        }
      } else if (!_seen_boundary_check && _paramKey == "boundary_check") {
        _seen_boundary_check = true;

        // Parse variable 'boundary_check'
        _result_boundary_check = ::mlir::FieldParser<BoolAttr>::parse(odsParser);
        if (::mlir::failed(_result_boundary_check)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_BlockTensorDescAttr parameter 'boundary_check' which is to be a `BoolAttr`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    ::llvm::StringRef _paramKey;
    if (!odsParser.parseOptionalKeyword(&_paramKey)) {
      if (!_loop_body(_paramKey)) return {};
      while (!odsParser.parseOptionalComma()) {
        ::llvm::StringRef _paramKey;
        if (odsParser.parseKeyword(&_paramKey)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                             "expected a parameter name in struct");
          return {};
        }
        if (!_loop_body(_paramKey)) return {};
      }
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return BlockTensorDescAttr::get(odsParser.getContext(),
      MemorySpaceAttr((_result_memory_space.value_or(MemorySpaceAttr::get(odsParser.getContext(), xegpu::MemorySpace::Global)))),
      IntegerAttr((_result_array_length.value_or(IntegerAttr::get(IntegerType::get(odsParser.getContext(), 64), 1)))),
      BoolAttr((_result_boundary_check.value_or(BoolAttr::get(odsParser.getContext(), true)))));
}

void BlockTensorDescAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getMemorySpace() == MemorySpaceAttr::get(getContext(), xegpu::MemorySpace::Global))) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "memory_space = ";
      if (!(getMemorySpace() == MemorySpaceAttr::get(getContext(), xegpu::MemorySpace::Global))) {
        odsPrinter.printStrippedAttrOrType(getMemorySpace());
      }
    }
    if (!(getArrayLength() == IntegerAttr::get(IntegerType::get(getContext(), 64), 1))) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "array_length = ";
      if (!(getArrayLength() == IntegerAttr::get(IntegerType::get(getContext(), 64), 1))) {
        odsPrinter.printStrippedAttrOrType(getArrayLength());
      }
    }
    if (!(getBoundaryCheck() == BoolAttr::get(getContext(), true))) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "boundary_check = ";
      if (!(getBoundaryCheck() == BoolAttr::get(getContext(), true))) {
        odsPrinter.printStrippedAttrOrType(getBoundaryCheck());
      }
    }
  }
  odsPrinter << ">";
}

MemorySpaceAttr BlockTensorDescAttr::getMemorySpace() const {
  return getImpl()->memory_space;
}

IntegerAttr BlockTensorDescAttr::getArrayLength() const {
  return getImpl()->array_length;
}

BoolAttr BlockTensorDescAttr::getBoundaryCheck() const {
  return getImpl()->boundary_check;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::BlockTensorDescAttr)
namespace mlir::xegpu {

namespace detail {

struct ScatterTensorDescAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<MemorySpaceAttr, IntegerAttr>;
  ScatterTensorDescAttrStorage(MemorySpaceAttr memory_space, IntegerAttr chunk_size) : memory_space(std::move(memory_space)), chunk_size(std::move(chunk_size)) {}

  KeyTy getAsKey() const {
    return KeyTy(memory_space, chunk_size);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (memory_space == std::get<0>(tblgenKey)) && (chunk_size == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static ScatterTensorDescAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto memory_space = std::move(std::get<0>(tblgenKey));
    auto chunk_size = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<ScatterTensorDescAttrStorage>()) ScatterTensorDescAttrStorage(std::move(memory_space), std::move(chunk_size));
  }

  MemorySpaceAttr memory_space;
  IntegerAttr chunk_size;
};

} // namespace detail
ScatterTensorDescAttr ScatterTensorDescAttr::get(::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr chunk_size) {
  return Base::get(context, std::move(memory_space), std::move(chunk_size));
}

ScatterTensorDescAttr ScatterTensorDescAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, MemorySpaceAttr memory_space, IntegerAttr chunk_size) {
  return Base::getChecked(emitError, context, std::move(memory_space), std::move(chunk_size));
}

::llvm::LogicalResult ScatterTensorDescAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, MemorySpaceAttr memory_space, IntegerAttr chunk_size) {
  if (::mlir::failed(verify(emitError, memory_space, chunk_size)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute ScatterTensorDescAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<MemorySpaceAttr> _result_memory_space;
  ::mlir::FailureOr<IntegerAttr> _result_chunk_size;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_memory_space = false;
  bool _seen_chunk_size = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_memory_space && _paramKey == "memory_space") {
        _seen_memory_space = true;

        // Parse variable 'memory_space'
        _result_memory_space = ::mlir::FieldParser<MemorySpaceAttr>::parse(odsParser);
        if (::mlir::failed(_result_memory_space)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_ScatterTensorDescAttr parameter 'memory_space' which is to be a `MemorySpaceAttr`");
          return {};
        }
      } else if (!_seen_chunk_size && _paramKey == "chunk_size") {
        _seen_chunk_size = true;

        // Parse variable 'chunk_size'
        _result_chunk_size = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
        if (::mlir::failed(_result_chunk_size)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_ScatterTensorDescAttr parameter 'chunk_size' which is to be a `IntegerAttr`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    ::llvm::StringRef _paramKey;
    if (!odsParser.parseOptionalKeyword(&_paramKey)) {
      if (!_loop_body(_paramKey)) return {};
      while (!odsParser.parseOptionalComma()) {
        ::llvm::StringRef _paramKey;
        if (odsParser.parseKeyword(&_paramKey)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                             "expected a parameter name in struct");
          return {};
        }
        if (!_loop_body(_paramKey)) return {};
      }
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return odsParser.getChecked<ScatterTensorDescAttr>(odsLoc, odsParser.getContext(),
      MemorySpaceAttr((_result_memory_space.value_or(MemorySpaceAttr::get(odsParser.getContext(), xegpu::MemorySpace::Global)))),
      IntegerAttr((_result_chunk_size.value_or(IntegerAttr::get(IntegerType::get(odsParser.getContext(), 64), 1)))));
}

void ScatterTensorDescAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getMemorySpace() == MemorySpaceAttr::get(getContext(), xegpu::MemorySpace::Global))) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "memory_space = ";
      if (!(getMemorySpace() == MemorySpaceAttr::get(getContext(), xegpu::MemorySpace::Global))) {
        odsPrinter.printStrippedAttrOrType(getMemorySpace());
      }
    }
    if (!(getChunkSize() == IntegerAttr::get(IntegerType::get(getContext(), 64), 1))) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "chunk_size = ";
      if (!(getChunkSize() == IntegerAttr::get(IntegerType::get(getContext(), 64), 1))) {
        odsPrinter.printStrippedAttrOrType(getChunkSize());
      }
    }
  }
  odsPrinter << ">";
}

MemorySpaceAttr ScatterTensorDescAttr::getMemorySpace() const {
  return getImpl()->memory_space;
}

IntegerAttr ScatterTensorDescAttr::getChunkSize() const {
  return getImpl()->chunk_size;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::ScatterTensorDescAttr)
namespace mlir::xegpu {

namespace detail {

struct MemorySpaceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xegpu::MemorySpace>;
  MemorySpaceAttrStorage(::mlir::xegpu::MemorySpace value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MemorySpaceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<MemorySpaceAttrStorage>()) MemorySpaceAttrStorage(std::move(value));
  }

  ::mlir::xegpu::MemorySpace value;
};

} // namespace detail
MemorySpaceAttr MemorySpaceAttr::get(::mlir::MLIRContext *context, ::mlir::xegpu::MemorySpace value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute MemorySpaceAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xegpu::MemorySpace> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xegpu::MemorySpace> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xegpu::symbolizeMemorySpace(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xegpu::MemorySpace" << " to be one of: " << "global" << ", " << "slm")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_MemorySpaceAttr parameter 'value' which is to be a `::mlir::xegpu::MemorySpace`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return MemorySpaceAttr::get(odsParser.getContext(),
      ::mlir::xegpu::MemorySpace((*_result_value)));
}

void MemorySpaceAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyMemorySpace(getValue());
}

::mlir::xegpu::MemorySpace MemorySpaceAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::MemorySpaceAttr)
namespace mlir::xegpu {

namespace detail {

struct CachePolicyAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xegpu::CachePolicy>;
  CachePolicyAttrStorage(::mlir::xegpu::CachePolicy value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CachePolicyAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CachePolicyAttrStorage>()) CachePolicyAttrStorage(std::move(value));
  }

  ::mlir::xegpu::CachePolicy value;
};

} // namespace detail
CachePolicyAttr CachePolicyAttr::get(::mlir::MLIRContext *context, ::mlir::xegpu::CachePolicy value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute CachePolicyAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xegpu::CachePolicy> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xegpu::CachePolicy> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xegpu::symbolizeCachePolicy(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xegpu::CachePolicy" << " to be one of: " << "cached" << ", " << "uncached" << ", " << "streaming" << ", " << "read_invalidate" << ", " << "write_back" << ", " << "write_through")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_CacheHintAttr parameter 'value' which is to be a `::mlir::xegpu::CachePolicy`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return CachePolicyAttr::get(odsParser.getContext(),
      ::mlir::xegpu::CachePolicy((*_result_value)));
}

void CachePolicyAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyCachePolicy(getValue());
  odsPrinter << ">";
}

::mlir::xegpu::CachePolicy CachePolicyAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::CachePolicyAttr)
namespace mlir::xegpu {

namespace detail {

struct FenceScopeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xegpu::FenceScope>;
  FenceScopeAttrStorage(::mlir::xegpu::FenceScope value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static FenceScopeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<FenceScopeAttrStorage>()) FenceScopeAttrStorage(std::move(value));
  }

  ::mlir::xegpu::FenceScope value;
};

} // namespace detail
FenceScopeAttr FenceScopeAttr::get(::mlir::MLIRContext *context, ::mlir::xegpu::FenceScope value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute FenceScopeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xegpu::FenceScope> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xegpu::FenceScope> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xegpu::symbolizeFenceScope(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xegpu::FenceScope" << " to be one of: " << "workgroup" << ", " << "gpu")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_FenceScopeAttr parameter 'value' which is to be a `::mlir::xegpu::FenceScope`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return FenceScopeAttr::get(odsParser.getContext(),
      ::mlir::xegpu::FenceScope((*_result_value)));
}

void FenceScopeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyFenceScope(getValue());
}

::mlir::xegpu::FenceScope FenceScopeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::FenceScopeAttr)
namespace mlir::xegpu {

namespace detail {

struct LayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<DenseI32ArrayAttr, DenseI32ArrayAttr, DenseI32ArrayAttr, DenseI32ArrayAttr, DenseI32ArrayAttr, DenseI32ArrayAttr>;
  LayoutAttrStorage(DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order) : sg_layout(std::move(sg_layout)), sg_data(std::move(sg_data)), inst_data(std::move(inst_data)), lane_layout(std::move(lane_layout)), lane_data(std::move(lane_data)), order(std::move(order)) {}

  KeyTy getAsKey() const {
    return KeyTy(sg_layout, sg_data, inst_data, lane_layout, lane_data, order);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (sg_layout == std::get<0>(tblgenKey)) && (sg_data == std::get<1>(tblgenKey)) && (inst_data == std::get<2>(tblgenKey)) && (lane_layout == std::get<3>(tblgenKey)) && (lane_data == std::get<4>(tblgenKey)) && (order == std::get<5>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey));
  }

  static LayoutAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto sg_layout = std::move(std::get<0>(tblgenKey));
    auto sg_data = std::move(std::get<1>(tblgenKey));
    auto inst_data = std::move(std::get<2>(tblgenKey));
    auto lane_layout = std::move(std::get<3>(tblgenKey));
    auto lane_data = std::move(std::get<4>(tblgenKey));
    auto order = std::move(std::get<5>(tblgenKey));
    return new (allocator.allocate<LayoutAttrStorage>()) LayoutAttrStorage(std::move(sg_layout), std::move(sg_data), std::move(inst_data), std::move(lane_layout), std::move(lane_data), std::move(order));
  }

  DenseI32ArrayAttr sg_layout;
  DenseI32ArrayAttr sg_data;
  DenseI32ArrayAttr inst_data;
  DenseI32ArrayAttr lane_layout;
  DenseI32ArrayAttr lane_data;
  DenseI32ArrayAttr order;
};

} // namespace detail
LayoutAttr LayoutAttr::get(::mlir::MLIRContext *context, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order) {
  return Base::get(context, std::move(sg_layout), std::move(sg_data), std::move(inst_data), std::move(lane_layout), std::move(lane_data), std::move(order));
}

LayoutAttr LayoutAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order) {
  return Base::getChecked(emitError, context, std::move(sg_layout), std::move(sg_data), std::move(inst_data), std::move(lane_layout), std::move(lane_data), std::move(order));
}

LayoutAttr LayoutAttr::get(::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> inst_data) {
  auto sg_layout = DenseI32ArrayAttr();
  auto sg_data = DenseI32ArrayAttr();
  auto order = DenseI32ArrayAttr();
  auto lane_layout = DenseI32ArrayAttr();
  auto lane_data = DenseI32ArrayAttr();
  return Base::get(context, sg_layout, sg_data,
               DenseI32ArrayAttr::get(context, inst_data),
               lane_layout, lane_data, order);
}

LayoutAttr LayoutAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> inst_data) {
  auto sg_layout = DenseI32ArrayAttr();
  auto sg_data = DenseI32ArrayAttr();
  auto order = DenseI32ArrayAttr();
  auto lane_layout = DenseI32ArrayAttr();
  auto lane_data = DenseI32ArrayAttr();
  return Base::getChecked(emitError, context, sg_layout, sg_data,
               DenseI32ArrayAttr::get(context, inst_data),
               lane_layout, lane_data, order);
}

LayoutAttr LayoutAttr::get(::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> inst_data, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data) {
  auto sg_layout = DenseI32ArrayAttr();
  auto sg_data = DenseI32ArrayAttr();
  auto order = DenseI32ArrayAttr();
  return Base::get(context, sg_layout, sg_data,
               DenseI32ArrayAttr::get(context, inst_data),
               DenseI32ArrayAttr::get(context, lane_layout),
               DenseI32ArrayAttr::get(context, lane_data), order);
}

LayoutAttr LayoutAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> inst_data, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data) {
  auto sg_layout = DenseI32ArrayAttr();
  auto sg_data = DenseI32ArrayAttr();
  auto order = DenseI32ArrayAttr();
  return Base::getChecked(emitError, context, sg_layout, sg_data,
               DenseI32ArrayAttr::get(context, inst_data),
               DenseI32ArrayAttr::get(context, lane_layout),
               DenseI32ArrayAttr::get(context, lane_data), order);
}

LayoutAttr LayoutAttr::get(::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data) {
  auto sg_layout = DenseI32ArrayAttr();
  auto sg_data = DenseI32ArrayAttr();
  auto inst_data = DenseI32ArrayAttr();
  auto order = DenseI32ArrayAttr();
  return Base::get(context, sg_layout, sg_data, inst_data,
               DenseI32ArrayAttr::get(context, lane_layout),
               DenseI32ArrayAttr::get(context, lane_data), order);
}

LayoutAttr LayoutAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, llvm::ArrayRef<int32_t> lane_layout, llvm::ArrayRef<int32_t> lane_data) {
  auto sg_layout = DenseI32ArrayAttr();
  auto sg_data = DenseI32ArrayAttr();
  auto inst_data = DenseI32ArrayAttr();
  auto order = DenseI32ArrayAttr();
  return Base::getChecked(emitError, context, sg_layout, sg_data, inst_data,
               DenseI32ArrayAttr::get(context, lane_layout),
               DenseI32ArrayAttr::get(context, lane_data), order);
}

LayoutAttr LayoutAttr::get(::mlir::MLIRContext *context, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order) {
  return Base::get(context, /*sg_layout =*/ nullptr, /*sg_data =*/ nullptr,
            /*inst_data =*/ nullptr, lane_layout, lane_data, order);
}

LayoutAttr LayoutAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order) {
  return Base::getChecked(emitError, context, /*sg_layout =*/ nullptr, /*sg_data =*/ nullptr,
            /*inst_data =*/ nullptr, lane_layout, lane_data, order);
}

::llvm::LogicalResult LayoutAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data, DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout, DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order) {
  if (::mlir::failed(verify(emitError, sg_layout, sg_data, inst_data, lane_layout, lane_data, order)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute LayoutAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<DenseI32ArrayAttr> _result_sg_layout;
  ::mlir::FailureOr<DenseI32ArrayAttr> _result_sg_data;
  ::mlir::FailureOr<DenseI32ArrayAttr> _result_inst_data;
  ::mlir::FailureOr<DenseI32ArrayAttr> _result_lane_layout;
  ::mlir::FailureOr<DenseI32ArrayAttr> _result_lane_data;
  ::mlir::FailureOr<DenseI32ArrayAttr> _result_order;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_sg_layout = false;
  bool _seen_sg_data = false;
  bool _seen_inst_data = false;
  bool _seen_lane_layout = false;
  bool _seen_lane_data = false;
  bool _seen_order = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_sg_layout && _paramKey == "sg_layout") {
        _seen_sg_layout = true;

        // Parse variable 'sg_layout'
        _result_sg_layout = ::mlir::FieldParser<DenseI32ArrayAttr>::parse(odsParser);
        if (::mlir::failed(_result_sg_layout)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_LayoutAttr parameter 'sg_layout' which is to be a `DenseI32ArrayAttr`");
          return {};
        }
      } else if (!_seen_sg_data && _paramKey == "sg_data") {
        _seen_sg_data = true;

        // Parse variable 'sg_data'
        _result_sg_data = ::mlir::FieldParser<DenseI32ArrayAttr>::parse(odsParser);
        if (::mlir::failed(_result_sg_data)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_LayoutAttr parameter 'sg_data' which is to be a `DenseI32ArrayAttr`");
          return {};
        }
      } else if (!_seen_inst_data && _paramKey == "inst_data") {
        _seen_inst_data = true;

        // Parse variable 'inst_data'
        _result_inst_data = ::mlir::FieldParser<DenseI32ArrayAttr>::parse(odsParser);
        if (::mlir::failed(_result_inst_data)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_LayoutAttr parameter 'inst_data' which is to be a `DenseI32ArrayAttr`");
          return {};
        }
      } else if (!_seen_lane_layout && _paramKey == "lane_layout") {
        _seen_lane_layout = true;

        // Parse variable 'lane_layout'
        _result_lane_layout = ::mlir::FieldParser<DenseI32ArrayAttr>::parse(odsParser);
        if (::mlir::failed(_result_lane_layout)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_LayoutAttr parameter 'lane_layout' which is to be a `DenseI32ArrayAttr`");
          return {};
        }
      } else if (!_seen_lane_data && _paramKey == "lane_data") {
        _seen_lane_data = true;

        // Parse variable 'lane_data'
        _result_lane_data = ::mlir::FieldParser<DenseI32ArrayAttr>::parse(odsParser);
        if (::mlir::failed(_result_lane_data)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_LayoutAttr parameter 'lane_data' which is to be a `DenseI32ArrayAttr`");
          return {};
        }
      } else if (!_seen_order && _paramKey == "order") {
        _seen_order = true;

        // Parse variable 'order'
        _result_order = ::mlir::FieldParser<DenseI32ArrayAttr>::parse(odsParser);
        if (::mlir::failed(_result_order)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_LayoutAttr parameter 'order' which is to be a `DenseI32ArrayAttr`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    ::llvm::StringRef _paramKey;
    if (!odsParser.parseOptionalKeyword(&_paramKey)) {
      if (!_loop_body(_paramKey)) return {};
      while (!odsParser.parseOptionalComma()) {
        ::llvm::StringRef _paramKey;
        if (odsParser.parseKeyword(&_paramKey)) {
          odsParser.emitError(odsParser.getCurrentLocation(),
                             "expected a parameter name in struct");
          return {};
        }
        if (!_loop_body(_paramKey)) return {};
      }
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  return odsParser.getChecked<LayoutAttr>(odsLoc, odsParser.getContext(),
      DenseI32ArrayAttr((_result_sg_layout.value_or(DenseI32ArrayAttr()))),
      DenseI32ArrayAttr((_result_sg_data.value_or(DenseI32ArrayAttr()))),
      DenseI32ArrayAttr((_result_inst_data.value_or(DenseI32ArrayAttr()))),
      DenseI32ArrayAttr((_result_lane_layout.value_or(DenseI32ArrayAttr()))),
      DenseI32ArrayAttr((_result_lane_data.value_or(DenseI32ArrayAttr()))),
      DenseI32ArrayAttr((_result_order.value_or(DenseI32ArrayAttr()))));
}

void LayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getSgLayout() == DenseI32ArrayAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "sg_layout = ";
      if (!(getSgLayout() == DenseI32ArrayAttr())) {
        odsPrinter.printStrippedAttrOrType(getSgLayout());
      }
    }
    if (!(getSgData() == DenseI32ArrayAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "sg_data = ";
      if (!(getSgData() == DenseI32ArrayAttr())) {
        odsPrinter.printStrippedAttrOrType(getSgData());
      }
    }
    if (!(getInstData() == DenseI32ArrayAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "inst_data = ";
      if (!(getInstData() == DenseI32ArrayAttr())) {
        odsPrinter.printStrippedAttrOrType(getInstData());
      }
    }
    if (!(getLaneLayout() == DenseI32ArrayAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "lane_layout = ";
      if (!(getLaneLayout() == DenseI32ArrayAttr())) {
        odsPrinter.printStrippedAttrOrType(getLaneLayout());
      }
    }
    if (!(getLaneData() == DenseI32ArrayAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "lane_data = ";
      if (!(getLaneData() == DenseI32ArrayAttr())) {
        odsPrinter.printStrippedAttrOrType(getLaneData());
      }
    }
    if (!(getOrder() == DenseI32ArrayAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "order = ";
      if (!(getOrder() == DenseI32ArrayAttr())) {
        odsPrinter.printStrippedAttrOrType(getOrder());
      }
    }
  }
  odsPrinter << ">";
}

DenseI32ArrayAttr LayoutAttr::getSgLayout() const {
  return getImpl()->sg_layout;
}

DenseI32ArrayAttr LayoutAttr::getSgData() const {
  return getImpl()->sg_data;
}

DenseI32ArrayAttr LayoutAttr::getInstData() const {
  return getImpl()->inst_data;
}

DenseI32ArrayAttr LayoutAttr::getLaneLayout() const {
  return getImpl()->lane_layout;
}

DenseI32ArrayAttr LayoutAttr::getLaneData() const {
  return getImpl()->lane_data;
}

DenseI32ArrayAttr LayoutAttr::getOrder() const {
  return getImpl()->order;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::LayoutAttr)
namespace mlir::xegpu {

namespace detail {

struct SliceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<xegpu::DistributeLayoutAttr, DenseI64ArrayAttr>;
  SliceAttrStorage(xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims) : parent(std::move(parent)), dims(std::move(dims)) {}

  KeyTy getAsKey() const {
    return KeyTy(parent, dims);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (parent == std::get<0>(tblgenKey)) && (dims == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static SliceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto parent = std::move(std::get<0>(tblgenKey));
    auto dims = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<SliceAttrStorage>()) SliceAttrStorage(std::move(parent), std::move(dims));
  }

  xegpu::DistributeLayoutAttr parent;
  DenseI64ArrayAttr dims;
};

} // namespace detail
SliceAttr SliceAttr::get(::mlir::MLIRContext *context, xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims) {
  return Base::get(context, std::move(parent), std::move(dims));
}

SliceAttr SliceAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims) {
  return Base::getChecked(emitError, context, std::move(parent), std::move(dims));
}

::llvm::LogicalResult SliceAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims) {
  if (::mlir::failed(verify(emitError, parent, dims)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute SliceAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<xegpu::DistributeLayoutAttr> _result_parent;
  ::mlir::FailureOr<DenseI64ArrayAttr> _result_dims;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'parent'
  _result_parent = ::mlir::FieldParser<xegpu::DistributeLayoutAttr>::parse(odsParser);
  if (::mlir::failed(_result_parent)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_SliceAttr parameter 'parent' which is to be a `xegpu::DistributeLayoutAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'dims'
  if (odsParser.parseKeyword("dims")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'dims'
  _result_dims = ::mlir::FieldParser<DenseI64ArrayAttr>::parse(odsParser);
  if (::mlir::failed(_result_dims)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_SliceAttr parameter 'dims' which is to be a `DenseI64ArrayAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_parent));
  assert(::mlir::succeeded(_result_dims));
  return odsParser.getChecked<SliceAttr>(odsLoc, odsParser.getContext(),
      xegpu::DistributeLayoutAttr((*_result_parent)),
      DenseI64ArrayAttr((*_result_dims)));
}

void SliceAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << getParent();
  odsPrinter << ",";
  odsPrinter << ' ' << "dims";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getDims());
  odsPrinter << ">";
}

xegpu::DistributeLayoutAttr SliceAttr::getParent() const {
  return getImpl()->parent;
}

DenseI64ArrayAttr SliceAttr::getDims() const {
  return getImpl()->dims;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::SliceAttr)
namespace mlir::xegpu {

namespace detail {

struct RangeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<IntegerAttr, IntegerAttr>;
  RangeAttrStorage(IntegerAttr start, IntegerAttr end) : start(std::move(start)), end(std::move(end)) {}

  KeyTy getAsKey() const {
    return KeyTy(start, end);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (start == std::get<0>(tblgenKey)) && (end == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static RangeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto start = std::move(std::get<0>(tblgenKey));
    auto end = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<RangeAttrStorage>()) RangeAttrStorage(std::move(start), std::move(end));
  }

  IntegerAttr start;
  IntegerAttr end;
};

} // namespace detail
RangeAttr RangeAttr::get(::mlir::MLIRContext *context, IntegerAttr start, IntegerAttr end) {
  return Base::get(context, std::move(start), std::move(end));
}

RangeAttr RangeAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, IntegerAttr start, IntegerAttr end) {
  return Base::getChecked(emitError, context, std::move(start), std::move(end));
}

::llvm::LogicalResult RangeAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, IntegerAttr start, IntegerAttr end) {
  if (::mlir::failed(verify(emitError, start, end)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute RangeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<IntegerAttr> _result_start;
  ::mlir::FailureOr<IntegerAttr> _result_end;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'start'
  _result_start = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_start)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_RangeAttr parameter 'start' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'end'
  _result_end = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_end)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeGPU_RangeAttr parameter 'end' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_start));
  assert(::mlir::succeeded(_result_end));
  return odsParser.getChecked<RangeAttr>(odsLoc, odsParser.getContext(),
      IntegerAttr((*_result_start)),
      IntegerAttr((*_result_end)));
}

void RangeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "[";
  odsPrinter.printStrippedAttrOrType(getStart());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getEnd());
  odsPrinter << "]";
  odsPrinter << ">";
}

IntegerAttr RangeAttr::getStart() const {
  return getImpl()->start;
}

IntegerAttr RangeAttr::getEnd() const {
  return getImpl()->end;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::RangeAttr)
namespace mlir::xegpu {

namespace detail {

struct MemLayoutAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<DictionaryAttr>;
  MemLayoutAttrStorage(DictionaryAttr attrs) : attrs(std::move(attrs)) {}

  KeyTy getAsKey() const {
    return KeyTy(attrs);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (attrs == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MemLayoutAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto attrs = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<MemLayoutAttrStorage>()) MemLayoutAttrStorage(std::move(attrs));
  }

  DictionaryAttr attrs;
};

} // namespace detail
MemLayoutAttr MemLayoutAttr::get(::mlir::MLIRContext *context, DictionaryAttr attrs) {
  return Base::get(context, std::move(attrs));
}

DictionaryAttr MemLayoutAttr::getAttrs() const {
  return getImpl()->attrs;
}


} // namespace mlir::xegpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xegpu::MemLayoutAttr)
namespace mlir::xegpu {


/// Parse an attribute registered to this dialect.
::mlir::Attribute XeGPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void XeGPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::xegpu

#endif // GET_ATTRDEF_CLASSES

