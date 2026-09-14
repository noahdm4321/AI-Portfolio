/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::acc::ClauseDefaultValueAttr,
::mlir::acc::ReductionOperatorAttr,
::mlir::acc::ParLevelAttr,
::mlir::acc::RecipeKindAttr,
::mlir::acc::DataClauseAttr,
::mlir::acc::DataClauseModifierAttr,
::mlir::acc::DeclareAttr,
::mlir::acc::DeclareActionAttr,
::mlir::acc::DeviceTypeAttr,
::mlir::acc::GangArgTypeAttr,
::mlir::acc::CombinedConstructsTypeAttr,
::mlir::acc::ConstructAttr,
::mlir::acc::VarNameAttr,
::mlir::acc::RoutineInfoAttr,
::mlir::acc::SpecializedRoutineAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::acc::ClauseDefaultValueAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::ClauseDefaultValueAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::ReductionOperatorAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::ReductionOperatorAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::ParLevelAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::ParLevelAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::RecipeKindAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::RecipeKindAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::DataClauseAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::DataClauseAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::DataClauseModifierAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::DataClauseModifierAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::DeclareAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::DeclareAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::DeclareActionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::DeclareActionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::DeviceTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::DeviceTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::GangArgTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::GangArgTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::CombinedConstructsTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::CombinedConstructsTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::ConstructAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::ConstructAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::VarNameAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::VarNameAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::RoutineInfoAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::RoutineInfoAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::acc::SpecializedRoutineAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::acc::SpecializedRoutineAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::acc::ClauseDefaultValueAttr>([&](auto t) {
      printer << ::mlir::acc::ClauseDefaultValueAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::ReductionOperatorAttr>([&](auto t) {
      printer << ::mlir::acc::ReductionOperatorAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::ParLevelAttr>([&](auto t) {
      printer << ::mlir::acc::ParLevelAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::RecipeKindAttr>([&](auto t) {
      printer << ::mlir::acc::RecipeKindAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::DataClauseAttr>([&](auto t) {
      printer << ::mlir::acc::DataClauseAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::DataClauseModifierAttr>([&](auto t) {
      printer << ::mlir::acc::DataClauseModifierAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::DeclareAttr>([&](auto t) {
      printer << ::mlir::acc::DeclareAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::DeclareActionAttr>([&](auto t) {
      printer << ::mlir::acc::DeclareActionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::DeviceTypeAttr>([&](auto t) {
      printer << ::mlir::acc::DeviceTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::GangArgTypeAttr>([&](auto t) {
      printer << ::mlir::acc::GangArgTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::CombinedConstructsTypeAttr>([&](auto t) {
      printer << ::mlir::acc::CombinedConstructsTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::ConstructAttr>([&](auto t) {
      printer << ::mlir::acc::ConstructAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::VarNameAttr>([&](auto t) {
      printer << ::mlir::acc::VarNameAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::RoutineInfoAttr>([&](auto t) {
      printer << ::mlir::acc::RoutineInfoAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::acc::SpecializedRoutineAttr>([&](auto t) {
      printer << ::mlir::acc::SpecializedRoutineAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::acc {

namespace detail {

struct ClauseDefaultValueAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::ClauseDefaultValue>;
  ClauseDefaultValueAttrStorage(::mlir::acc::ClauseDefaultValue value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ClauseDefaultValueAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ClauseDefaultValueAttrStorage>()) ClauseDefaultValueAttrStorage(std::move(value));
  }

  ::mlir::acc::ClauseDefaultValue value;
};

} // namespace detail
ClauseDefaultValueAttr ClauseDefaultValueAttr::get(::mlir::MLIRContext *context, ::mlir::acc::ClauseDefaultValue value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ClauseDefaultValueAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::ClauseDefaultValue> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::ClauseDefaultValue> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeClauseDefaultValue(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::ClauseDefaultValue" << " to be one of: " << "present" << ", " << "none")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DefaultValueAttr parameter 'value' which is to be a `::mlir::acc::ClauseDefaultValue`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ClauseDefaultValueAttr::get(odsParser.getContext(),
      ::mlir::acc::ClauseDefaultValue((*_result_value)));
}

void ClauseDefaultValueAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyClauseDefaultValue(getValue());
}

::mlir::acc::ClauseDefaultValue ClauseDefaultValueAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ClauseDefaultValueAttr)
namespace mlir::acc {

namespace detail {

struct ReductionOperatorAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::ReductionOperator>;
  ReductionOperatorAttrStorage(::mlir::acc::ReductionOperator value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ReductionOperatorAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ReductionOperatorAttrStorage>()) ReductionOperatorAttrStorage(std::move(value));
  }

  ::mlir::acc::ReductionOperator value;
};

} // namespace detail
ReductionOperatorAttr ReductionOperatorAttr::get(::mlir::MLIRContext *context, ::mlir::acc::ReductionOperator value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ReductionOperatorAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::ReductionOperator> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::ReductionOperator> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeReductionOperator(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::ReductionOperator" << " to be one of: " << "none" << ", " << "add" << ", " << "mul" << ", " << "max" << ", " << "min" << ", " << "iand" << ", " << "ior" << ", " << "xor" << ", " << "eqv" << ", " << "neqv" << ", " << "land" << ", " << "lor")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_ReductionOperatorAttr parameter 'value' which is to be a `::mlir::acc::ReductionOperator`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return ReductionOperatorAttr::get(odsParser.getContext(),
      ::mlir::acc::ReductionOperator((*_result_value)));
}

void ReductionOperatorAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyReductionOperator(getValue());
  odsPrinter << ">";
}

::mlir::acc::ReductionOperator ReductionOperatorAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ReductionOperatorAttr)
namespace mlir::acc {

namespace detail {

struct ParLevelAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::ParLevel>;
  ParLevelAttrStorage(::mlir::acc::ParLevel value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ParLevelAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ParLevelAttrStorage>()) ParLevelAttrStorage(std::move(value));
  }

  ::mlir::acc::ParLevel value;
};

} // namespace detail
ParLevelAttr ParLevelAttr::get(::mlir::MLIRContext *context, ::mlir::acc::ParLevel value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ParLevelAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::ParLevel> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::ParLevel> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeParLevel(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::ParLevel" << " to be one of: " << "seq" << ", " << "gang_dim1" << ", " << "gang_dim2" << ", " << "gang_dim3" << ", " << "worker" << ", " << "vector")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_ParLevelAttr parameter 'value' which is to be a `::mlir::acc::ParLevel`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return ParLevelAttr::get(odsParser.getContext(),
      ::mlir::acc::ParLevel((*_result_value)));
}

void ParLevelAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyParLevel(getValue());
  odsPrinter << ">";
}

::mlir::acc::ParLevel ParLevelAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ParLevelAttr)
namespace mlir::acc {

namespace detail {

struct RecipeKindAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::RecipeKind>;
  RecipeKindAttrStorage(::mlir::acc::RecipeKind value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RecipeKindAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RecipeKindAttrStorage>()) RecipeKindAttrStorage(std::move(value));
  }

  ::mlir::acc::RecipeKind value;
};

} // namespace detail
RecipeKindAttr RecipeKindAttr::get(::mlir::MLIRContext *context, ::mlir::acc::RecipeKind value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute RecipeKindAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::RecipeKind> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::RecipeKind> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeRecipeKind(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::RecipeKind" << " to be one of: " << "private_recipe" << ", " << "firstprivate_recipe" << ", " << "reduction_recipe")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_RecipeKindAttr parameter 'value' which is to be a `::mlir::acc::RecipeKind`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return RecipeKindAttr::get(odsParser.getContext(),
      ::mlir::acc::RecipeKind((*_result_value)));
}

void RecipeKindAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyRecipeKind(getValue());
  odsPrinter << ">";
}

::mlir::acc::RecipeKind RecipeKindAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::RecipeKindAttr)
namespace mlir::acc {

namespace detail {

struct DataClauseAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::DataClause>;
  DataClauseAttrStorage(::mlir::acc::DataClause value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DataClauseAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<DataClauseAttrStorage>()) DataClauseAttrStorage(std::move(value));
  }

  ::mlir::acc::DataClause value;
};

} // namespace detail
DataClauseAttr DataClauseAttr::get(::mlir::MLIRContext *context, ::mlir::acc::DataClause value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute DataClauseAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::DataClause> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::DataClause> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeDataClause(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::DataClause" << " to be one of: " << "acc_copyin" << ", " << "acc_copyin_readonly" << ", " << "acc_copy" << ", " << "acc_copyout" << ", " << "acc_copyout_zero" << ", " << "acc_present" << ", " << "acc_create" << ", " << "acc_create_zero" << ", " << "acc_delete" << ", " << "acc_attach" << ", " << "acc_detach" << ", " << "acc_no_create" << ", " << "acc_private" << ", " << "acc_firstprivate" << ", " << "acc_deviceptr" << ", " << "acc_getdeviceptr" << ", " << "acc_update_host" << ", " << "acc_update_self" << ", " << "acc_update_device" << ", " << "acc_use_device" << ", " << "acc_reduction" << ", " << "acc_declare_device_resident" << ", " << "acc_declare_link" << ", " << "acc_cache" << ", " << "acc_cache_readonly")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_DataClauseAttr parameter 'value' which is to be a `::mlir::acc::DataClause`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return DataClauseAttr::get(odsParser.getContext(),
      ::mlir::acc::DataClause((*_result_value)));
}

void DataClauseAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyDataClause(getValue());
}

::mlir::acc::DataClause DataClauseAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DataClauseAttr)
namespace mlir::acc {

namespace detail {

struct DataClauseModifierAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::DataClauseModifier>;
  DataClauseModifierAttrStorage(::mlir::acc::DataClauseModifier value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DataClauseModifierAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<DataClauseModifierAttrStorage>()) DataClauseModifierAttrStorage(std::move(value));
  }

  ::mlir::acc::DataClauseModifier value;
};

} // namespace detail
DataClauseModifierAttr DataClauseModifierAttr::get(::mlir::MLIRContext *context, ::mlir::acc::DataClauseModifier value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute DataClauseModifierAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::DataClauseModifier> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::DataClauseModifier> {
      ::mlir::acc::DataClauseModifier flags = {};
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      do {
        if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
          return ::mlir::failure();
        auto maybeEnum = ::mlir::acc::symbolizeDataClauseModifier(enumKeyword);
        if (!maybeEnum) {
            return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::DataClauseModifier" << " to be one of: " << "none" << ", " << "zero" << ", " << "readonly" << ", " << "alwaysin" << ", " << "alwaysout" << ", " << "always" << ", " << "capture")};
        }
        flags = flags | *maybeEnum;
      } while(::mlir::succeeded(odsParser.parseOptionalComma()));
      return flags;
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_DataClauseModifierAttr parameter 'value' which is to be a `::mlir::acc::DataClauseModifier`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return DataClauseModifierAttr::get(odsParser.getContext(),
      ::mlir::acc::DataClauseModifier((*_result_value)));
}

void DataClauseModifierAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyDataClauseModifier(getValue());
}

::mlir::acc::DataClauseModifier DataClauseModifierAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DataClauseModifierAttr)
namespace mlir::acc {

namespace detail {

struct DeclareAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<DataClauseAttr, bool>;
  DeclareAttrStorage(DataClauseAttr dataClause, bool implicit) : dataClause(std::move(dataClause)), implicit(std::move(implicit)) {}

  KeyTy getAsKey() const {
    return KeyTy(dataClause, implicit);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (dataClause == std::get<0>(tblgenKey)) && (implicit == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static DeclareAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto dataClause = std::move(std::get<0>(tblgenKey));
    auto implicit = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<DeclareAttrStorage>()) DeclareAttrStorage(std::move(dataClause), std::move(implicit));
  }

  DataClauseAttr dataClause;
  bool implicit;
};

} // namespace detail
DeclareAttr DeclareAttr::get(::mlir::MLIRContext *context, DataClauseAttr dataClause, bool implicit) {
  return Base::get(context, std::move(dataClause), std::move(implicit));
}

DeclareAttr DeclareAttr::get(::mlir::MLIRContext *context, DataClauseAttr dataClause) {
  return Base::get(context, dataClause, /*implicit=*/false);
}

::mlir::Attribute DeclareAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<DataClauseAttr> _result_dataClause;
  ::mlir::FailureOr<bool> _result_implicit;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_dataClause = false;
  bool _seen_implicit = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_dataClause && _paramKey == "dataClause") {
        _seen_dataClause = true;

        // Parse variable 'dataClause'
        _result_dataClause = ::mlir::FieldParser<DataClauseAttr>::parse(odsParser);
        if (::mlir::failed(_result_dataClause)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareAttr parameter 'dataClause' which is to be a `DataClauseAttr`");
          return {};
        }
      } else if (!_seen_implicit && _paramKey == "implicit") {
        _seen_implicit = true;

        // Parse variable 'implicit'
        _result_implicit = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_implicit)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareAttr parameter 'implicit' which is to be a `bool`");
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
    if (!_seen_dataClause) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "dataClause";
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_dataClause));
  return DeclareAttr::get(odsParser.getContext(),
      DataClauseAttr((*_result_dataClause)),
      bool((_result_implicit.value_or(false))));
}

void DeclareAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "dataClause = ";
    odsPrinter.printStrippedAttrOrType(getDataClause());
    if (!(getImplicit() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "implicit = ";
      if (!(getImplicit() == false)) {
        odsPrinter.printStrippedAttrOrType(getImplicit());
      }
    }
  }
  odsPrinter << ">";
}

DataClauseAttr DeclareAttr::getDataClause() const {
  return getImpl()->dataClause;
}

bool DeclareAttr::getImplicit() const {
  return getImpl()->implicit;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DeclareAttr)
namespace mlir::acc {

namespace detail {

struct DeclareActionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<SymbolRefAttr, SymbolRefAttr, SymbolRefAttr, SymbolRefAttr>;
  DeclareActionAttrStorage(SymbolRefAttr preAlloc, SymbolRefAttr postAlloc, SymbolRefAttr preDealloc, SymbolRefAttr postDealloc) : preAlloc(std::move(preAlloc)), postAlloc(std::move(postAlloc)), preDealloc(std::move(preDealloc)), postDealloc(std::move(postDealloc)) {}

  KeyTy getAsKey() const {
    return KeyTy(preAlloc, postAlloc, preDealloc, postDealloc);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (preAlloc == std::get<0>(tblgenKey)) && (postAlloc == std::get<1>(tblgenKey)) && (preDealloc == std::get<2>(tblgenKey)) && (postDealloc == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static DeclareActionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto preAlloc = std::move(std::get<0>(tblgenKey));
    auto postAlloc = std::move(std::get<1>(tblgenKey));
    auto preDealloc = std::move(std::get<2>(tblgenKey));
    auto postDealloc = std::move(std::get<3>(tblgenKey));
    return new (allocator.allocate<DeclareActionAttrStorage>()) DeclareActionAttrStorage(std::move(preAlloc), std::move(postAlloc), std::move(preDealloc), std::move(postDealloc));
  }

  SymbolRefAttr preAlloc;
  SymbolRefAttr postAlloc;
  SymbolRefAttr preDealloc;
  SymbolRefAttr postDealloc;
};

} // namespace detail
DeclareActionAttr DeclareActionAttr::get(::mlir::MLIRContext *context, SymbolRefAttr preAlloc, SymbolRefAttr postAlloc, SymbolRefAttr preDealloc, SymbolRefAttr postDealloc) {
  return Base::get(context, std::move(preAlloc), std::move(postAlloc), std::move(preDealloc), std::move(postDealloc));
}

::mlir::Attribute DeclareActionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<SymbolRefAttr> _result_preAlloc;
  ::mlir::FailureOr<SymbolRefAttr> _result_postAlloc;
  ::mlir::FailureOr<SymbolRefAttr> _result_preDealloc;
  ::mlir::FailureOr<SymbolRefAttr> _result_postDealloc;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_preAlloc = false;
  bool _seen_postAlloc = false;
  bool _seen_preDealloc = false;
  bool _seen_postDealloc = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_preAlloc && _paramKey == "preAlloc") {
        _seen_preAlloc = true;

        // Parse variable 'preAlloc'
        _result_preAlloc = ::mlir::FieldParser<SymbolRefAttr>::parse(odsParser);
        if (::mlir::failed(_result_preAlloc)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareActionAttr parameter 'preAlloc' which is to be a `SymbolRefAttr`");
          return {};
        }
      } else if (!_seen_postAlloc && _paramKey == "postAlloc") {
        _seen_postAlloc = true;

        // Parse variable 'postAlloc'
        _result_postAlloc = ::mlir::FieldParser<SymbolRefAttr>::parse(odsParser);
        if (::mlir::failed(_result_postAlloc)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareActionAttr parameter 'postAlloc' which is to be a `SymbolRefAttr`");
          return {};
        }
      } else if (!_seen_preDealloc && _paramKey == "preDealloc") {
        _seen_preDealloc = true;

        // Parse variable 'preDealloc'
        _result_preDealloc = ::mlir::FieldParser<SymbolRefAttr>::parse(odsParser);
        if (::mlir::failed(_result_preDealloc)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareActionAttr parameter 'preDealloc' which is to be a `SymbolRefAttr`");
          return {};
        }
      } else if (!_seen_postDealloc && _paramKey == "postDealloc") {
        _seen_postDealloc = true;

        // Parse variable 'postDealloc'
        _result_postDealloc = ::mlir::FieldParser<SymbolRefAttr>::parse(odsParser);
        if (::mlir::failed(_result_postDealloc)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse DeclareActionAttr parameter 'postDealloc' which is to be a `SymbolRefAttr`");
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
  return DeclareActionAttr::get(odsParser.getContext(),
      SymbolRefAttr((_result_preAlloc.value_or(SymbolRefAttr()))),
      SymbolRefAttr((_result_postAlloc.value_or(SymbolRefAttr()))),
      SymbolRefAttr((_result_preDealloc.value_or(SymbolRefAttr()))),
      SymbolRefAttr((_result_postDealloc.value_or(SymbolRefAttr()))));
}

void DeclareActionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getPreAlloc() == SymbolRefAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "preAlloc = ";
      if (!(getPreAlloc() == SymbolRefAttr())) {
        odsPrinter.printStrippedAttrOrType(getPreAlloc());
      }
    }
    if (!(getPostAlloc() == SymbolRefAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "postAlloc = ";
      if (!(getPostAlloc() == SymbolRefAttr())) {
        odsPrinter.printStrippedAttrOrType(getPostAlloc());
      }
    }
    if (!(getPreDealloc() == SymbolRefAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "preDealloc = ";
      if (!(getPreDealloc() == SymbolRefAttr())) {
        odsPrinter.printStrippedAttrOrType(getPreDealloc());
      }
    }
    if (!(getPostDealloc() == SymbolRefAttr())) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "postDealloc = ";
      if (!(getPostDealloc() == SymbolRefAttr())) {
        odsPrinter.printStrippedAttrOrType(getPostDealloc());
      }
    }
  }
  odsPrinter << ">";
}

SymbolRefAttr DeclareActionAttr::getPreAlloc() const {
  return getImpl()->preAlloc;
}

SymbolRefAttr DeclareActionAttr::getPostAlloc() const {
  return getImpl()->postAlloc;
}

SymbolRefAttr DeclareActionAttr::getPreDealloc() const {
  return getImpl()->preDealloc;
}

SymbolRefAttr DeclareActionAttr::getPostDealloc() const {
  return getImpl()->postDealloc;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DeclareActionAttr)
namespace mlir::acc {

namespace detail {

struct DeviceTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::DeviceType>;
  DeviceTypeAttrStorage(::mlir::acc::DeviceType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static DeviceTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<DeviceTypeAttrStorage>()) DeviceTypeAttrStorage(std::move(value));
  }

  ::mlir::acc::DeviceType value;
};

} // namespace detail
DeviceTypeAttr DeviceTypeAttr::get(::mlir::MLIRContext *context, ::mlir::acc::DeviceType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute DeviceTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::DeviceType> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::DeviceType> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeDeviceType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::DeviceType" << " to be one of: " << "none" << ", " << "star" << ", " << "default" << ", " << "host" << ", " << "multicore" << ", " << "nvidia" << ", " << "radeon")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_DeviceTypeAttr parameter 'value' which is to be a `::mlir::acc::DeviceType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return DeviceTypeAttr::get(odsParser.getContext(),
      ::mlir::acc::DeviceType((*_result_value)));
}

void DeviceTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyDeviceType(getValue());
  odsPrinter << ">";
}

::mlir::acc::DeviceType DeviceTypeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DeviceTypeAttr)
namespace mlir::acc {

namespace detail {

struct GangArgTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::GangArgType>;
  GangArgTypeAttrStorage(::mlir::acc::GangArgType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static GangArgTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<GangArgTypeAttrStorage>()) GangArgTypeAttrStorage(std::move(value));
  }

  ::mlir::acc::GangArgType value;
};

} // namespace detail
GangArgTypeAttr GangArgTypeAttr::get(::mlir::MLIRContext *context, ::mlir::acc::GangArgType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute GangArgTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::GangArgType> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::GangArgType> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeGangArgType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::GangArgType" << " to be one of: " << "Num" << ", " << "Dim" << ", " << "Static")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_GangArgTypeAttr parameter 'value' which is to be a `::mlir::acc::GangArgType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return GangArgTypeAttr::get(odsParser.getContext(),
      ::mlir::acc::GangArgType((*_result_value)));
}

void GangArgTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyGangArgType(getValue());
  odsPrinter << ">";
}

::mlir::acc::GangArgType GangArgTypeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::GangArgTypeAttr)
namespace mlir::acc {

namespace detail {

struct CombinedConstructsTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::CombinedConstructsType>;
  CombinedConstructsTypeAttrStorage(::mlir::acc::CombinedConstructsType value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CombinedConstructsTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<CombinedConstructsTypeAttrStorage>()) CombinedConstructsTypeAttrStorage(std::move(value));
  }

  ::mlir::acc::CombinedConstructsType value;
};

} // namespace detail
CombinedConstructsTypeAttr CombinedConstructsTypeAttr::get(::mlir::MLIRContext *context, ::mlir::acc::CombinedConstructsType value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute CombinedConstructsTypeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::CombinedConstructsType> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::CombinedConstructsType> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeCombinedConstructsType(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::CombinedConstructsType" << " to be one of: " << "kernels_loop" << ", " << "parallel_loop" << ", " << "serial_loop")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_CombinedConstructsAttr parameter 'value' which is to be a `::mlir::acc::CombinedConstructsType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return CombinedConstructsTypeAttr::get(odsParser.getContext(),
      ::mlir::acc::CombinedConstructsType((*_result_value)));
}

void CombinedConstructsTypeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyCombinedConstructsType(getValue());
  odsPrinter << ">";
}

::mlir::acc::CombinedConstructsType CombinedConstructsTypeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::CombinedConstructsTypeAttr)
namespace mlir::acc {

namespace detail {

struct ConstructAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::acc::Construct>;
  ConstructAttrStorage(::mlir::acc::Construct value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ConstructAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ConstructAttrStorage>()) ConstructAttrStorage(std::move(value));
  }

  ::mlir::acc::Construct value;
};

} // namespace detail
ConstructAttr ConstructAttr::get(::mlir::MLIRContext *context, ::mlir::acc::Construct value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ConstructAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::acc::Construct> _result_value;

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::acc::Construct> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::acc::symbolizeConstruct(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::acc::Construct" << " to be one of: " << "acc_construct_parallel" << ", " << "acc_construct_kernels" << ", " << "acc_construct_loop" << ", " << "acc_construct_data" << ", " << "acc_construct_enter_data" << ", " << "acc_construct_exit_data" << ", " << "acc_construct_host_data" << ", " << "acc_construct_atomic" << ", " << "acc_construct_declare" << ", " << "acc_construct_init" << ", " << "acc_construct_shutdown" << ", " << "acc_construct_set" << ", " << "acc_construct_update" << ", " << "acc_construct_routine" << ", " << "acc_construct_wait" << ", " << "acc_construct_runtime_api" << ", " << "acc_construct_serial")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_ConstructAttr parameter 'value' which is to be a `::mlir::acc::Construct`");
    return {};
  }
  assert(::mlir::succeeded(_result_value));
  return ConstructAttr::get(odsParser.getContext(),
      ::mlir::acc::Construct((*_result_value)));
}

void ConstructAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << ' ';
  odsPrinter << stringifyConstruct(getValue());
}

::mlir::acc::Construct ConstructAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ConstructAttr)
namespace mlir::acc {

namespace detail {

struct VarNameAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::StringRef>;
  VarNameAttrStorage(::llvm::StringRef name) : name(std::move(name)) {}

  KeyTy getAsKey() const {
    return KeyTy(name);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (name == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static VarNameAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto name = std::move(std::get<0>(tblgenKey));
    name = allocator.copyInto(name);
    return new (allocator.allocate<VarNameAttrStorage>()) VarNameAttrStorage(std::move(name));
  }

  ::llvm::StringRef name;
};

} // namespace detail
VarNameAttr VarNameAttr::get(::mlir::MLIRContext *context, ::llvm::StringRef name) {
  return Base::get(context, std::move(name));
}

::mlir::Attribute VarNameAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<std::string> _result_name;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'name'
  _result_name = ::mlir::FieldParser<std::string>::parse(odsParser);
  if (::mlir::failed(_result_name)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse OpenACC_VarNameAttr parameter 'name' which is to be a `::llvm::StringRef`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_name));
  return VarNameAttr::get(odsParser.getContext(),
      ::llvm::StringRef((*_result_name)));
}

void VarNameAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printString(getName());;
  odsPrinter << ">";
}

::llvm::StringRef VarNameAttr::getName() const {
  return getImpl()->name;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::VarNameAttr)
namespace mlir::acc {

namespace detail {

struct RoutineInfoAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<SymbolRefAttr>>;
  RoutineInfoAttrStorage(::llvm::ArrayRef<SymbolRefAttr> accRoutines) : accRoutines(std::move(accRoutines)) {}

  KeyTy getAsKey() const {
    return KeyTy(accRoutines);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (accRoutines == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RoutineInfoAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto accRoutines = std::move(std::get<0>(tblgenKey));
    accRoutines = allocator.copyInto(accRoutines);
    return new (allocator.allocate<RoutineInfoAttrStorage>()) RoutineInfoAttrStorage(std::move(accRoutines));
  }

  ::llvm::ArrayRef<SymbolRefAttr> accRoutines;
};

} // namespace detail
RoutineInfoAttr RoutineInfoAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<SymbolRefAttr> accRoutines) {
  return Base::get(context, std::move(accRoutines));
}

::mlir::Attribute RoutineInfoAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::llvm::SmallVector<SymbolRefAttr>> _result_accRoutines;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'accRoutines'
  _result_accRoutines = ::mlir::FieldParser<::llvm::SmallVector<SymbolRefAttr>>::parse(odsParser);
  if (::mlir::failed(_result_accRoutines)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse RoutineInfoAttr parameter 'accRoutines' which is to be a `::llvm::ArrayRef<SymbolRefAttr>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_accRoutines));
  return RoutineInfoAttr::get(odsParser.getContext(),
      ::llvm::ArrayRef<SymbolRefAttr>((*_result_accRoutines)));
}

void RoutineInfoAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "[";
  odsPrinter.printStrippedAttrOrType(getAccRoutines());
  odsPrinter << "]";
  odsPrinter << ">";
}

::llvm::ArrayRef<SymbolRefAttr> RoutineInfoAttr::getAccRoutines() const {
  return getImpl()->accRoutines;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::RoutineInfoAttr)
namespace mlir::acc {

namespace detail {

struct SpecializedRoutineAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<SymbolRefAttr, ParLevelAttr, StringAttr>;
  SpecializedRoutineAttrStorage(SymbolRefAttr routine, ParLevelAttr level, StringAttr funcName) : routine(std::move(routine)), level(std::move(level)), funcName(std::move(funcName)) {}

  KeyTy getAsKey() const {
    return KeyTy(routine, level, funcName);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (routine == std::get<0>(tblgenKey)) && (level == std::get<1>(tblgenKey)) && (funcName == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static SpecializedRoutineAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto routine = std::move(std::get<0>(tblgenKey));
    auto level = std::move(std::get<1>(tblgenKey));
    auto funcName = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<SpecializedRoutineAttrStorage>()) SpecializedRoutineAttrStorage(std::move(routine), std::move(level), std::move(funcName));
  }

  SymbolRefAttr routine;
  ParLevelAttr level;
  StringAttr funcName;
};

} // namespace detail
SpecializedRoutineAttr SpecializedRoutineAttr::get(::mlir::MLIRContext *context, SymbolRefAttr routine, ParLevelAttr level, StringAttr funcName) {
  return Base::get(context, std::move(routine), std::move(level), std::move(funcName));
}

::mlir::Attribute SpecializedRoutineAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<SymbolRefAttr> _result_routine;
  ::mlir::FailureOr<ParLevelAttr> _result_level;
  ::mlir::FailureOr<StringAttr> _result_funcName;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'routine'
  _result_routine = ::mlir::FieldParser<SymbolRefAttr>::parse(odsParser);
  if (::mlir::failed(_result_routine)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SpecializedRoutineAttr parameter 'routine' which is to be a `SymbolRefAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'level'
  _result_level = ::mlir::FieldParser<ParLevelAttr>::parse(odsParser);
  if (::mlir::failed(_result_level)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SpecializedRoutineAttr parameter 'level' which is to be a `ParLevelAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'funcName'
  _result_funcName = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_funcName)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse SpecializedRoutineAttr parameter 'funcName' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_routine));
  assert(::mlir::succeeded(_result_level));
  assert(::mlir::succeeded(_result_funcName));
  return SpecializedRoutineAttr::get(odsParser.getContext(),
      SymbolRefAttr((*_result_routine)),
      ParLevelAttr((*_result_level)),
      StringAttr((*_result_funcName)));
}

void SpecializedRoutineAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getRoutine());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLevel());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getFuncName());
  odsPrinter << ">";
}

SymbolRefAttr SpecializedRoutineAttr::getRoutine() const {
  return getImpl()->routine;
}

ParLevelAttr SpecializedRoutineAttr::getLevel() const {
  return getImpl()->level;
}

StringAttr SpecializedRoutineAttr::getFuncName() const {
  return getImpl()->funcName;
}


} // namespace mlir::acc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::SpecializedRoutineAttr)
namespace mlir::acc {


/// Parse an attribute registered to this dialect.
::mlir::Attribute OpenACCDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void OpenACCDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::acc

#endif // GET_ATTRDEF_CLASSES

