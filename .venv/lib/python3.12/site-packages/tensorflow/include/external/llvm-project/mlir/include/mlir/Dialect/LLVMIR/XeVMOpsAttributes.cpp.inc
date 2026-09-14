/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::xevm::LoadCacheControlAttr,
::mlir::xevm::StoreCacheControlAttr,
::mlir::xevm::MemScopeAttr,
::mlir::xevm::AddrSpaceAttr,
::mlir::xevm::MMAShapeAttr,
::mlir::xevm::MMATypesAttr,
::mlir::xevm::XeVMTargetAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::xevm::LoadCacheControlAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xevm::LoadCacheControlAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xevm::StoreCacheControlAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xevm::StoreCacheControlAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xevm::MemScopeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xevm::MemScopeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xevm::AddrSpaceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xevm::AddrSpaceAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xevm::MMAShapeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xevm::MMAShapeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xevm::MMATypesAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xevm::MMATypesAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::xevm::XeVMTargetAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::xevm::XeVMTargetAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::xevm::LoadCacheControlAttr>([&](auto t) {
      printer << ::mlir::xevm::LoadCacheControlAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xevm::StoreCacheControlAttr>([&](auto t) {
      printer << ::mlir::xevm::StoreCacheControlAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xevm::MemScopeAttr>([&](auto t) {
      printer << ::mlir::xevm::MemScopeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xevm::AddrSpaceAttr>([&](auto t) {
      printer << ::mlir::xevm::AddrSpaceAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xevm::MMAShapeAttr>([&](auto t) {
      printer << ::mlir::xevm::MMAShapeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xevm::MMATypesAttr>([&](auto t) {
      printer << ::mlir::xevm::MMATypesAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::xevm::XeVMTargetAttr>([&](auto t) {
      printer << ::mlir::xevm::XeVMTargetAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::xevm {

namespace detail {

struct LoadCacheControlAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xevm::LoadCacheControl>;
  LoadCacheControlAttrStorage(::mlir::xevm::LoadCacheControl value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static LoadCacheControlAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<LoadCacheControlAttrStorage>()) LoadCacheControlAttrStorage(std::move(value));
  }

  ::mlir::xevm::LoadCacheControl value;
};

} // namespace detail
LoadCacheControlAttr LoadCacheControlAttr::get(::mlir::MLIRContext *context, ::mlir::xevm::LoadCacheControl value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute LoadCacheControlAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xevm::LoadCacheControl> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xevm::LoadCacheControl> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xevm::symbolizeLoadCacheControl(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xevm::LoadCacheControl" << " to be one of: " << "L1uc_L2uc_L3uc" << ", " << "L1uc_L2uc_L3c" << ", " << "L1uc_L2c_L3uc" << ", " << "L1uc_L2c_L3c" << ", " << "L1c_L2uc_L3uc" << ", " << "L1c_L2uc_L3c" << ", " << "L1c_L2c_L3uc" << ", " << "L1c_L2c_L3c" << ", " << "L1s_L2uc_L3uc" << ", " << "L1s_L2uc_L3c" << ", " << "L1s_L2c_L3uc" << ", " << "L1s_L2c_L3c" << ", " << "ir")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_LoadCacheControlAttr parameter 'value' which is to be a `::mlir::xevm::LoadCacheControl`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return LoadCacheControlAttr::get(odsParser.getContext(),
      ::mlir::xevm::LoadCacheControl((*_result_value)));
}

void LoadCacheControlAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyLoadCacheControl(getValue());
  odsPrinter << ">";
}

::mlir::xevm::LoadCacheControl LoadCacheControlAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::xevm
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xevm::LoadCacheControlAttr)
namespace mlir::xevm {

namespace detail {

struct StoreCacheControlAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xevm::StoreCacheControl>;
  StoreCacheControlAttrStorage(::mlir::xevm::StoreCacheControl value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static StoreCacheControlAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<StoreCacheControlAttrStorage>()) StoreCacheControlAttrStorage(std::move(value));
  }

  ::mlir::xevm::StoreCacheControl value;
};

} // namespace detail
StoreCacheControlAttr StoreCacheControlAttr::get(::mlir::MLIRContext *context, ::mlir::xevm::StoreCacheControl value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute StoreCacheControlAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xevm::StoreCacheControl> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xevm::StoreCacheControl> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xevm::symbolizeStoreCacheControl(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xevm::StoreCacheControl" << " to be one of: " << "L1uc_L2uc_L3uc" << ", " << "L1uc_L2uc_L3wb" << ", " << "L1uc_L2wb_L3uc" << ", " << "L1uc_L2wb_L3wb" << ", " << "L1wt_L2uc_L3uc" << ", " << "L1wt_L2uc_L3wb" << ", " << "L1wt_L2wb_L3uc" << ", " << "L1wt_L2wb_L3wb" << ", " << "L1s_L2uc_L3uc" << ", " << "L1s_L2uc_L3wb" << ", " << "L1s_L2wb_L3uc" << ", " << "L1s_L2wb_L3wb" << ", " << "L1wb_L2uc_L3uc" << ", " << "L1wb_L2wb_L3uc" << ", " << "L1wb_L2uc_L3wb")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_StoreCacheControlAttr parameter 'value' which is to be a `::mlir::xevm::StoreCacheControl`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return StoreCacheControlAttr::get(odsParser.getContext(),
      ::mlir::xevm::StoreCacheControl((*_result_value)));
}

void StoreCacheControlAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyStoreCacheControl(getValue());
  odsPrinter << ">";
}

::mlir::xevm::StoreCacheControl StoreCacheControlAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::xevm
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xevm::StoreCacheControlAttr)
namespace mlir::xevm {

namespace detail {

struct MemScopeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::xevm::MemScope>;
  MemScopeAttrStorage(::mlir::xevm::MemScope value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static MemScopeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<MemScopeAttrStorage>()) MemScopeAttrStorage(std::move(value));
  }

  ::mlir::xevm::MemScope value;
};

} // namespace detail
MemScopeAttr MemScopeAttr::get(::mlir::MLIRContext *context, ::mlir::xevm::MemScope value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute MemScopeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::xevm::MemScope> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::xevm::MemScope> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::xevm::symbolizeMemScope(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::xevm::MemScope" << " to be one of: " << "lane" << ", " << "subgroup" << ", " << "workgroup" << ", " << "cluster" << ", " << "device" << ", " << "system")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_MemScopeAttr parameter 'value' which is to be a `::mlir::xevm::MemScope`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return MemScopeAttr::get(odsParser.getContext(),
      ::mlir::xevm::MemScope((*_result_value)));
}

void MemScopeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyMemScope(getValue());
  odsPrinter << ">";
}

::mlir::xevm::MemScope MemScopeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::xevm
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xevm::MemScopeAttr)
namespace mlir::xevm {

namespace detail {

struct AddrSpaceAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<mlir::xevm::AddrSpace>;
  AddrSpaceAttrStorage(mlir::xevm::AddrSpace value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static AddrSpaceAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<AddrSpaceAttrStorage>()) AddrSpaceAttrStorage(std::move(value));
  }

  mlir::xevm::AddrSpace value;
};

} // namespace detail
AddrSpaceAttr AddrSpaceAttr::get(::mlir::MLIRContext *context, mlir::xevm::AddrSpace value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute AddrSpaceAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<mlir::xevm::AddrSpace> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<mlir::xevm::AddrSpace> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = mlir::xevm::symbolizeAddrSpace(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "mlir::xevm::AddrSpace" << " to be one of: " << "private" << ", " << "global" << ", " << "constant" << ", " << "shared" << ", " << "generic")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_AddrSpaceAttr parameter 'value' which is to be a `mlir::xevm::AddrSpace`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return AddrSpaceAttr::get(odsParser.getContext(),
      mlir::xevm::AddrSpace((*_result_value)));
}

void AddrSpaceAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyAddrSpace(getValue());
  odsPrinter << ">";
}

mlir::xevm::AddrSpace AddrSpaceAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::xevm
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xevm::AddrSpaceAttr)
namespace mlir::xevm {

namespace detail {

struct MMAShapeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int, int, int>;
  MMAShapeAttrStorage(int m, int n, int k) : m(std::move(m)), n(std::move(n)), k(std::move(k)) {}

  KeyTy getAsKey() const {
    return KeyTy(m, n, k);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (m == std::get<0>(tblgenKey)) && (n == std::get<1>(tblgenKey)) && (k == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static MMAShapeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto m = std::move(std::get<0>(tblgenKey));
    auto n = std::move(std::get<1>(tblgenKey));
    auto k = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<MMAShapeAttrStorage>()) MMAShapeAttrStorage(std::move(m), std::move(n), std::move(k));
  }

  int m;
  int n;
  int k;
};

} // namespace detail
MMAShapeAttr MMAShapeAttr::get(::mlir::MLIRContext *context, int m, int n, int k) {
  return Base::get(context, std::move(m), std::move(n), std::move(k));
}

::mlir::Attribute MMAShapeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int> _result_m;
  ::mlir::FailureOr<int> _result_n;
  ::mlir::FailureOr<int> _result_k;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_m = false;
  bool _seen_n = false;
  bool _seen_k = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_m && _paramKey == "m") {
        _seen_m = true;

        // Parse variable 'm'
        _result_m = ::mlir::FieldParser<int>::parse(odsParser);
        if (::mlir::failed(_result_m)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_MMAShapeAttr parameter 'm' which is to be a `int`");
          return {};
        }
      } else if (!_seen_n && _paramKey == "n") {
        _seen_n = true;

        // Parse variable 'n'
        _result_n = ::mlir::FieldParser<int>::parse(odsParser);
        if (::mlir::failed(_result_n)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_MMAShapeAttr parameter 'n' which is to be a `int`");
          return {};
        }
      } else if (!_seen_k && _paramKey == "k") {
        _seen_k = true;

        // Parse variable 'k'
        _result_k = ::mlir::FieldParser<int>::parse(odsParser);
        if (::mlir::failed(_result_k)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_MMAShapeAttr parameter 'k' which is to be a `int`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 3; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 3 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_m));
  assert(::mlir::succeeded(_result_n));
  assert(::mlir::succeeded(_result_k));
  return MMAShapeAttr::get(odsParser.getContext(),
      int((*_result_m)),
      int((*_result_n)),
      int((*_result_k)));
}

void MMAShapeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "m = ";
    odsPrinter.printStrippedAttrOrType(getM());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "n = ";
    odsPrinter.printStrippedAttrOrType(getN());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "k = ";
    odsPrinter.printStrippedAttrOrType(getK());
  }
  odsPrinter << ">";
}

int MMAShapeAttr::getM() const {
  return getImpl()->m;
}

int MMAShapeAttr::getN() const {
  return getImpl()->n;
}

int MMAShapeAttr::getK() const {
  return getImpl()->k;
}


} // namespace mlir::xevm
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xevm::MMAShapeAttr)
namespace mlir::xevm {

namespace detail {

struct MMATypesAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<xevm::ElemType, xevm::ElemType, xevm::ElemType, xevm::ElemType>;
  MMATypesAttrStorage(xevm::ElemType d, xevm::ElemType a, xevm::ElemType b, xevm::ElemType c) : d(std::move(d)), a(std::move(a)), b(std::move(b)), c(std::move(c)) {}

  KeyTy getAsKey() const {
    return KeyTy(d, a, b, c);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (d == std::get<0>(tblgenKey)) && (a == std::get<1>(tblgenKey)) && (b == std::get<2>(tblgenKey)) && (c == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static MMATypesAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto d = std::move(std::get<0>(tblgenKey));
    auto a = std::move(std::get<1>(tblgenKey));
    auto b = std::move(std::get<2>(tblgenKey));
    auto c = std::move(std::get<3>(tblgenKey));
    return new (allocator.allocate<MMATypesAttrStorage>()) MMATypesAttrStorage(std::move(d), std::move(a), std::move(b), std::move(c));
  }

  xevm::ElemType d;
  xevm::ElemType a;
  xevm::ElemType b;
  xevm::ElemType c;
};

} // namespace detail
MMATypesAttr MMATypesAttr::get(::mlir::MLIRContext *context, xevm::ElemType d, xevm::ElemType a, xevm::ElemType b, xevm::ElemType c) {
  return Base::get(context, std::move(d), std::move(a), std::move(b), std::move(c));
}

::mlir::Attribute MMATypesAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<xevm::ElemType> _result_d;
  ::mlir::FailureOr<xevm::ElemType> _result_a;
  ::mlir::FailureOr<xevm::ElemType> _result_b;
  ::mlir::FailureOr<xevm::ElemType> _result_c;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_d = false;
  bool _seen_a = false;
  bool _seen_b = false;
  bool _seen_c = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_d && _paramKey == "d") {
        _seen_d = true;

        // Parse variable 'd'
        _result_d = ::mlir::FieldParser<xevm::ElemType>::parse(odsParser);
        if (::mlir::failed(_result_d)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_MMATypesAttr parameter 'd' which is to be a `xevm::ElemType`");
          return {};
        }
      } else if (!_seen_a && _paramKey == "a") {
        _seen_a = true;

        // Parse variable 'a'
        _result_a = ::mlir::FieldParser<xevm::ElemType>::parse(odsParser);
        if (::mlir::failed(_result_a)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_MMATypesAttr parameter 'a' which is to be a `xevm::ElemType`");
          return {};
        }
      } else if (!_seen_b && _paramKey == "b") {
        _seen_b = true;

        // Parse variable 'b'
        _result_b = ::mlir::FieldParser<xevm::ElemType>::parse(odsParser);
        if (::mlir::failed(_result_b)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_MMATypesAttr parameter 'b' which is to be a `xevm::ElemType`");
          return {};
        }
      } else if (!_seen_c && _paramKey == "c") {
        _seen_c = true;

        // Parse variable 'c'
        _result_c = ::mlir::FieldParser<xevm::ElemType>::parse(odsParser);
        if (::mlir::failed(_result_c)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_MMATypesAttr parameter 'c' which is to be a `xevm::ElemType`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    do {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
    } while(!odsParser.parseOptionalComma());
    if (!_seen_d) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "d";
      return {};
    }
    if (!_seen_a) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "a";
      return {};
    }
    if (!_seen_b) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "b";
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_d));
  assert(::mlir::succeeded(_result_a));
  assert(::mlir::succeeded(_result_b));
  return MMATypesAttr::get(odsParser.getContext(),
      xevm::ElemType((*_result_d)),
      xevm::ElemType((*_result_a)),
      xevm::ElemType((*_result_b)),
      xevm::ElemType((_result_c.value_or(xevm::ElemType()))));
}

void MMATypesAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "d = ";
    odsPrinter.printStrippedAttrOrType(getD());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "a = ";
    odsPrinter.printStrippedAttrOrType(getA());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "b = ";
    odsPrinter.printStrippedAttrOrType(getB());
    if (!(getC() == xevm::ElemType())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "c = ";
      if (!(getC() == xevm::ElemType())) {
        odsPrinter.printStrippedAttrOrType(getC());
      }
    }
  }
  odsPrinter << ">";
}

xevm::ElemType MMATypesAttr::getD() const {
  return getImpl()->d;
}

xevm::ElemType MMATypesAttr::getA() const {
  return getImpl()->a;
}

xevm::ElemType MMATypesAttr::getB() const {
  return getImpl()->b;
}

xevm::ElemType MMATypesAttr::getC() const {
  return getImpl()->c;
}


} // namespace mlir::xevm
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xevm::MMATypesAttr)
namespace mlir::xevm {

namespace detail {

struct XeVMTargetAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int, ::llvm::StringRef, ::llvm::StringRef, ::mlir::DictionaryAttr, ::mlir::ArrayAttr>;
  XeVMTargetAttrStorage(int O, ::llvm::StringRef triple, ::llvm::StringRef chip, ::mlir::DictionaryAttr flags, ::mlir::ArrayAttr linkFiles) : O(std::move(O)), triple(std::move(triple)), chip(std::move(chip)), flags(std::move(flags)), linkFiles(std::move(linkFiles)) {}

  KeyTy getAsKey() const {
    return KeyTy(O, triple, chip, flags, linkFiles);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (O == std::get<0>(tblgenKey)) && (triple == std::get<1>(tblgenKey)) && (chip == std::get<2>(tblgenKey)) && (flags == std::get<3>(tblgenKey)) && (linkFiles == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static XeVMTargetAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto O = std::move(std::get<0>(tblgenKey));
    auto triple = std::move(std::get<1>(tblgenKey));
    auto chip = std::move(std::get<2>(tblgenKey));
    auto flags = std::move(std::get<3>(tblgenKey));
    auto linkFiles = std::move(std::get<4>(tblgenKey));
    triple = allocator.copyInto(triple);
    chip = allocator.copyInto(chip);
    return new (allocator.allocate<XeVMTargetAttrStorage>()) XeVMTargetAttrStorage(std::move(O), std::move(triple), std::move(chip), std::move(flags), std::move(linkFiles));
  }

  int O;
  ::llvm::StringRef triple;
  ::llvm::StringRef chip;
  ::mlir::DictionaryAttr flags;
  ::mlir::ArrayAttr linkFiles;
};

} // namespace detail
XeVMTargetAttr XeVMTargetAttr::get(::mlir::MLIRContext *context, int optLevel, ::llvm::StringRef triple, ::llvm::StringRef chip, ::mlir::DictionaryAttr targetFlags, ::mlir::ArrayAttr linkFiles) {
  return Base::get(context, optLevel, triple, chip, targetFlags, linkFiles);
}

XeVMTargetAttr XeVMTargetAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int optLevel, ::llvm::StringRef triple, ::llvm::StringRef chip, ::mlir::DictionaryAttr targetFlags, ::mlir::ArrayAttr linkFiles) {
  return Base::get(context, optLevel, triple, chip, targetFlags, linkFiles);
}

::llvm::LogicalResult XeVMTargetAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int O, ::llvm::StringRef triple, ::llvm::StringRef chip, ::mlir::DictionaryAttr flags, ::mlir::ArrayAttr linkFiles) {
  if (::mlir::failed(verify(emitError, O, triple, chip, flags, linkFiles)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute XeVMTargetAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int> _result_O;
  ::mlir::FailureOr<std::string> _result_triple;
  ::mlir::FailureOr<std::string> _result_chip;
  ::mlir::FailureOr<::mlir::DictionaryAttr> _result_flags;
  ::mlir::FailureOr<::mlir::ArrayAttr> _result_linkFiles;
  // Parse literal '<'
  if (odsParser.parseOptionalLess()) {
  } else {
    // Parse parameter struct
    bool _seen_O = false;
    bool _seen_triple = false;
    bool _seen_chip = false;
    bool _seen_flags = false;
    bool _seen_linkFiles = false;
    {
      const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
        // Parse literal '='
        if (odsParser.parseEqual()) return {};
        if (!_seen_O && _paramKey == "O") {
          _seen_O = true;

          // Parse variable 'O'
          _result_O = ::mlir::FieldParser<int>::parse(odsParser);
          if (::mlir::failed(_result_O)) {
            odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_TargetAttr parameter 'O' which is to be a `int`");
            return {};
          }
        } else if (!_seen_triple && _paramKey == "triple") {
          _seen_triple = true;

          // Parse variable 'triple'
          _result_triple = ::mlir::FieldParser<std::string>::parse(odsParser);
          if (::mlir::failed(_result_triple)) {
            odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_TargetAttr parameter 'triple' which is to be a `::llvm::StringRef`");
            return {};
          }
        } else if (!_seen_chip && _paramKey == "chip") {
          _seen_chip = true;

          // Parse variable 'chip'
          _result_chip = ::mlir::FieldParser<std::string>::parse(odsParser);
          if (::mlir::failed(_result_chip)) {
            odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_TargetAttr parameter 'chip' which is to be a `::llvm::StringRef`");
            return {};
          }
        } else if (!_seen_flags && _paramKey == "flags") {
          _seen_flags = true;

          // Parse variable 'flags'
          _result_flags = ::mlir::FieldParser<::mlir::DictionaryAttr>::parse(odsParser);
          if (::mlir::failed(_result_flags)) {
            odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_TargetAttr parameter 'flags' which is to be a `::mlir::DictionaryAttr`");
            return {};
          }
        } else if (!_seen_linkFiles && _paramKey == "linkFiles") {
          _seen_linkFiles = true;

          // Parse variable 'linkFiles'
          _result_linkFiles = ::mlir::FieldParser<::mlir::ArrayAttr>::parse(odsParser);
          if (::mlir::failed(_result_linkFiles)) {
            odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse XeVM_TargetAttr parameter 'linkFiles' which is to be a `::mlir::ArrayAttr`");
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
  }
  return odsParser.getChecked<XeVMTargetAttr>(odsLoc, odsParser.getContext(),
      int((_result_O.value_or(2))),
      ::llvm::StringRef((_result_triple.value_or("spirv64-unknown-unknown"))),
      ::llvm::StringRef((_result_chip.value_or("bmg"))),
      ::mlir::DictionaryAttr((_result_flags.value_or(::mlir::DictionaryAttr()))),
      ::mlir::ArrayAttr((_result_linkFiles.value_or(::mlir::ArrayAttr()))));
}

void XeVMTargetAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  if (!(getO() == 2) || !(getTriple() == "spirv64-unknown-unknown") || !(getChip() == "bmg") || !(getFlags() == ::mlir::DictionaryAttr()) || !(getLinkFiles() == ::mlir::ArrayAttr())) {
    odsPrinter << "<";
    {
      bool _firstPrinted = true;
      if (!(getO() == 2)) {
        if (!_firstPrinted) odsPrinter << ", ";
        _firstPrinted = false;
        odsPrinter << "O = ";
        if (!(getO() == 2)) {
          odsPrinter.printStrippedAttrOrType(getO());
        }
      }
      if (!(getTriple() == "spirv64-unknown-unknown")) {
        if (!_firstPrinted) odsPrinter << ", ";
        _firstPrinted = false;
        odsPrinter << "triple = ";
        if (!(getTriple() == "spirv64-unknown-unknown")) {
          odsPrinter.printString(getTriple());;
        }
      }
      if (!(getChip() == "bmg")) {
        if (!_firstPrinted) odsPrinter << ", ";
        _firstPrinted = false;
        odsPrinter << "chip = ";
        if (!(getChip() == "bmg")) {
          odsPrinter.printString(getChip());;
        }
      }
      if (!(getFlags() == ::mlir::DictionaryAttr())) {
        if (!_firstPrinted) odsPrinter << ", ";
        _firstPrinted = false;
        odsPrinter << "flags = ";
        if (!(getFlags() == ::mlir::DictionaryAttr())) {
          odsPrinter.printStrippedAttrOrType(getFlags());
        }
      }
      if (!(getLinkFiles() == ::mlir::ArrayAttr())) {
        if (!_firstPrinted) odsPrinter << ", ";
        _firstPrinted = false;
        odsPrinter << "linkFiles = ";
        if (!(getLinkFiles() == ::mlir::ArrayAttr())) {
          odsPrinter.printStrippedAttrOrType(getLinkFiles());
        }
      }
    }
    odsPrinter << ">";
  } else {
  }
}

int XeVMTargetAttr::getO() const {
  return getImpl()->O;
}

::llvm::StringRef XeVMTargetAttr::getTriple() const {
  return getImpl()->triple;
}

::llvm::StringRef XeVMTargetAttr::getChip() const {
  return getImpl()->chip;
}

::mlir::DictionaryAttr XeVMTargetAttr::getFlags() const {
  return getImpl()->flags;
}

::mlir::ArrayAttr XeVMTargetAttr::getLinkFiles() const {
  return getImpl()->linkFiles;
}


} // namespace mlir::xevm
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::xevm::XeVMTargetAttr)
namespace mlir::xevm {


/// Parse an attribute registered to this dialect.
::mlir::Attribute XeVMDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void XeVMDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::xevm

#endif // GET_ATTRDEF_CLASSES

