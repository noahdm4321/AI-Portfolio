/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

mlir::tosa::UnaryOpQuantizationAttr,
mlir::tosa::ConvOpQuantizationAttr,
mlir::tosa::MatMulOpQuantizationAttr,
mlir::tosa::PadOpQuantizationAttr,
mlir::tosa::ProfileAttr,
mlir::tosa::ExtensionAttr,
mlir::tosa::LevelAttr,
mlir::tosa::SpecificationVersionAttr,
mlir::tosa::TargetEnvAttr,
mlir::tosa::ResizeModeAttr,
mlir::tosa::NanPropagationModeAttr,
mlir::tosa::RoundingModeAttr,
mlir::tosa::BlockSizeAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(mlir::tosa::UnaryOpQuantizationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::UnaryOpQuantizationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::ConvOpQuantizationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::ConvOpQuantizationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::MatMulOpQuantizationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::MatMulOpQuantizationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::PadOpQuantizationAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::PadOpQuantizationAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::ProfileAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::ProfileAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::ExtensionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::ExtensionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::LevelAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::LevelAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::SpecificationVersionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::SpecificationVersionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::TargetEnvAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::TargetEnvAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::ResizeModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::ResizeModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::NanPropagationModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::NanPropagationModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::RoundingModeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::RoundingModeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(mlir::tosa::BlockSizeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = mlir::tosa::BlockSizeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<mlir::tosa::UnaryOpQuantizationAttr>([&](auto t) {
      printer << mlir::tosa::UnaryOpQuantizationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::ConvOpQuantizationAttr>([&](auto t) {
      printer << mlir::tosa::ConvOpQuantizationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::MatMulOpQuantizationAttr>([&](auto t) {
      printer << mlir::tosa::MatMulOpQuantizationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::PadOpQuantizationAttr>([&](auto t) {
      printer << mlir::tosa::PadOpQuantizationAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::ProfileAttr>([&](auto t) {
      printer << mlir::tosa::ProfileAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::ExtensionAttr>([&](auto t) {
      printer << mlir::tosa::ExtensionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::LevelAttr>([&](auto t) {
      printer << mlir::tosa::LevelAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::SpecificationVersionAttr>([&](auto t) {
      printer << mlir::tosa::SpecificationVersionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::TargetEnvAttr>([&](auto t) {
      printer << mlir::tosa::TargetEnvAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::ResizeModeAttr>([&](auto t) {
      printer << mlir::tosa::ResizeModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::NanPropagationModeAttr>([&](auto t) {
      printer << mlir::tosa::NanPropagationModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::RoundingModeAttr>([&](auto t) {
      printer << mlir::tosa::RoundingModeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<mlir::tosa::BlockSizeAttr>([&](auto t) {
      printer << mlir::tosa::BlockSizeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::tosa {

namespace detail {

struct UnaryOpQuantizationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t>;
  UnaryOpQuantizationAttrStorage(int64_t input_zp, int64_t output_zp) : input_zp(std::move(input_zp)), output_zp(std::move(output_zp)) {}

  KeyTy getAsKey() const {
    return KeyTy(input_zp, output_zp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (input_zp == std::get<0>(tblgenKey)) && (output_zp == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static UnaryOpQuantizationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto input_zp = std::move(std::get<0>(tblgenKey));
    auto output_zp = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<UnaryOpQuantizationAttrStorage>()) UnaryOpQuantizationAttrStorage(std::move(input_zp), std::move(output_zp));
  }

  int64_t input_zp;
  int64_t output_zp;
};

} // namespace detail
UnaryOpQuantizationAttr UnaryOpQuantizationAttr::get(::mlir::MLIRContext *context, int64_t input_zp, int64_t output_zp) {
  return Base::get(context, std::move(input_zp), std::move(output_zp));
}

::mlir::Attribute UnaryOpQuantizationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_input_zp;
  ::mlir::FailureOr<int64_t> _result_output_zp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_input_zp = false;
  bool _seen_output_zp = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_input_zp && _paramKey == "input_zp") {
        _seen_input_zp = true;

        // Parse variable 'input_zp'
        _result_input_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_input_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_UnaryOpQuantizationAttr parameter 'input_zp' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_output_zp && _paramKey == "output_zp") {
        _seen_output_zp = true;

        // Parse variable 'output_zp'
        _result_output_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_output_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_UnaryOpQuantizationAttr parameter 'output_zp' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_input_zp));
  assert(::mlir::succeeded(_result_output_zp));
  return UnaryOpQuantizationAttr::get(odsParser.getContext(),
      int64_t((*_result_input_zp)),
      int64_t((*_result_output_zp)));
}

void UnaryOpQuantizationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "input_zp = ";
    odsPrinter.printStrippedAttrOrType(getInputZp());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "output_zp = ";
    odsPrinter.printStrippedAttrOrType(getOutputZp());
  }
  odsPrinter << ">";
}

int64_t UnaryOpQuantizationAttr::getInputZp() const {
  return getImpl()->input_zp;
}

int64_t UnaryOpQuantizationAttr::getOutputZp() const {
  return getImpl()->output_zp;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::UnaryOpQuantizationAttr)
namespace mlir::tosa {

namespace detail {

struct ConvOpQuantizationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t>;
  ConvOpQuantizationAttrStorage(int64_t input_zp, int64_t weight_zp) : input_zp(std::move(input_zp)), weight_zp(std::move(weight_zp)) {}

  KeyTy getAsKey() const {
    return KeyTy(input_zp, weight_zp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (input_zp == std::get<0>(tblgenKey)) && (weight_zp == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static ConvOpQuantizationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto input_zp = std::move(std::get<0>(tblgenKey));
    auto weight_zp = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<ConvOpQuantizationAttrStorage>()) ConvOpQuantizationAttrStorage(std::move(input_zp), std::move(weight_zp));
  }

  int64_t input_zp;
  int64_t weight_zp;
};

} // namespace detail
ConvOpQuantizationAttr ConvOpQuantizationAttr::get(::mlir::MLIRContext *context, int64_t input_zp, int64_t weight_zp) {
  return Base::get(context, std::move(input_zp), std::move(weight_zp));
}

::mlir::Attribute ConvOpQuantizationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_input_zp;
  ::mlir::FailureOr<int64_t> _result_weight_zp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_input_zp = false;
  bool _seen_weight_zp = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_input_zp && _paramKey == "input_zp") {
        _seen_input_zp = true;

        // Parse variable 'input_zp'
        _result_input_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_input_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_ConvOpQuantizationAttr parameter 'input_zp' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_weight_zp && _paramKey == "weight_zp") {
        _seen_weight_zp = true;

        // Parse variable 'weight_zp'
        _result_weight_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_weight_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_ConvOpQuantizationAttr parameter 'weight_zp' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_input_zp));
  assert(::mlir::succeeded(_result_weight_zp));
  return ConvOpQuantizationAttr::get(odsParser.getContext(),
      int64_t((*_result_input_zp)),
      int64_t((*_result_weight_zp)));
}

void ConvOpQuantizationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "input_zp = ";
    odsPrinter.printStrippedAttrOrType(getInputZp());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "weight_zp = ";
    odsPrinter.printStrippedAttrOrType(getWeightZp());
  }
  odsPrinter << ">";
}

int64_t ConvOpQuantizationAttr::getInputZp() const {
  return getImpl()->input_zp;
}

int64_t ConvOpQuantizationAttr::getWeightZp() const {
  return getImpl()->weight_zp;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ConvOpQuantizationAttr)
namespace mlir::tosa {

namespace detail {

struct MatMulOpQuantizationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t, int64_t>;
  MatMulOpQuantizationAttrStorage(int64_t a_zp, int64_t b_zp) : a_zp(std::move(a_zp)), b_zp(std::move(b_zp)) {}

  KeyTy getAsKey() const {
    return KeyTy(a_zp, b_zp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (a_zp == std::get<0>(tblgenKey)) && (b_zp == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static MatMulOpQuantizationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto a_zp = std::move(std::get<0>(tblgenKey));
    auto b_zp = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<MatMulOpQuantizationAttrStorage>()) MatMulOpQuantizationAttrStorage(std::move(a_zp), std::move(b_zp));
  }

  int64_t a_zp;
  int64_t b_zp;
};

} // namespace detail
MatMulOpQuantizationAttr MatMulOpQuantizationAttr::get(::mlir::MLIRContext *context, int64_t a_zp, int64_t b_zp) {
  return Base::get(context, std::move(a_zp), std::move(b_zp));
}

::mlir::Attribute MatMulOpQuantizationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_a_zp;
  ::mlir::FailureOr<int64_t> _result_b_zp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_a_zp = false;
  bool _seen_b_zp = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_a_zp && _paramKey == "a_zp") {
        _seen_a_zp = true;

        // Parse variable 'a_zp'
        _result_a_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_a_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_MatMulOpQuantizationAttr parameter 'a_zp' which is to be a `int64_t`");
          return {};
        }
      } else if (!_seen_b_zp && _paramKey == "b_zp") {
        _seen_b_zp = true;

        // Parse variable 'b_zp'
        _result_b_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_b_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_MatMulOpQuantizationAttr parameter 'b_zp' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_a_zp));
  assert(::mlir::succeeded(_result_b_zp));
  return MatMulOpQuantizationAttr::get(odsParser.getContext(),
      int64_t((*_result_a_zp)),
      int64_t((*_result_b_zp)));
}

void MatMulOpQuantizationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "a_zp = ";
    odsPrinter.printStrippedAttrOrType(getAZp());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "b_zp = ";
    odsPrinter.printStrippedAttrOrType(getBZp());
  }
  odsPrinter << ">";
}

int64_t MatMulOpQuantizationAttr::getAZp() const {
  return getImpl()->a_zp;
}

int64_t MatMulOpQuantizationAttr::getBZp() const {
  return getImpl()->b_zp;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::MatMulOpQuantizationAttr)
namespace mlir::tosa {

namespace detail {

struct PadOpQuantizationAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int64_t>;
  PadOpQuantizationAttrStorage(int64_t input_zp) : input_zp(std::move(input_zp)) {}

  KeyTy getAsKey() const {
    return KeyTy(input_zp);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (input_zp == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PadOpQuantizationAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto input_zp = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<PadOpQuantizationAttrStorage>()) PadOpQuantizationAttrStorage(std::move(input_zp));
  }

  int64_t input_zp;
};

} // namespace detail
PadOpQuantizationAttr PadOpQuantizationAttr::get(::mlir::MLIRContext *context, int64_t input_zp) {
  return Base::get(context, std::move(input_zp));
}

::mlir::Attribute PadOpQuantizationAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int64_t> _result_input_zp;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_input_zp = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_input_zp && _paramKey == "input_zp") {
        _seen_input_zp = true;

        // Parse variable 'input_zp'
        _result_input_zp = ::mlir::FieldParser<int64_t>::parse(odsParser);
        if (::mlir::failed(_result_input_zp)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_PadOpQuantizationAttr parameter 'input_zp' which is to be a `int64_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 1; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 1 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_input_zp));
  return PadOpQuantizationAttr::get(odsParser.getContext(),
      int64_t((*_result_input_zp)));
}

void PadOpQuantizationAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "input_zp = ";
    odsPrinter.printStrippedAttrOrType(getInputZp());
  }
  odsPrinter << ">";
}

int64_t PadOpQuantizationAttr::getInputZp() const {
  return getImpl()->input_zp;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::PadOpQuantizationAttr)
namespace mlir::tosa {

namespace detail {

struct ProfileAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::tosa::Profile>;
  ProfileAttrStorage(::mlir::tosa::Profile value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ProfileAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ProfileAttrStorage>()) ProfileAttrStorage(std::move(value));
  }

  ::mlir::tosa::Profile value;
};

} // namespace detail
ProfileAttr ProfileAttr::get(::mlir::MLIRContext *context, ::mlir::tosa::Profile value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ProfileAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::tosa::Profile> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::tosa::Profile> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::tosa::symbolizeProfile(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::tosa::Profile" << " to be one of: " << "pro_int" << ", " << "pro_fp" << ", " << "none")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_ProfileAttr parameter 'value' which is to be a `::mlir::tosa::Profile`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return ProfileAttr::get(odsParser.getContext(),
      ::mlir::tosa::Profile((*_result_value)));
}

void ProfileAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyProfile(getValue());
  odsPrinter << ">";
}

::mlir::tosa::Profile ProfileAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ProfileAttr)
namespace mlir::tosa {

namespace detail {

struct ExtensionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::tosa::Extension>;
  ExtensionAttrStorage(::mlir::tosa::Extension value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ExtensionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ExtensionAttrStorage>()) ExtensionAttrStorage(std::move(value));
  }

  ::mlir::tosa::Extension value;
};

} // namespace detail
ExtensionAttr ExtensionAttr::get(::mlir::MLIRContext *context, ::mlir::tosa::Extension value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ExtensionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::tosa::Extension> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::tosa::Extension> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::tosa::symbolizeExtension(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::tosa::Extension" << " to be one of: " << "none" << ", " << "int16" << ", " << "int4" << ", " << "bf16" << ", " << "fp8e4m3" << ", " << "fp8e5m2" << ", " << "fft" << ", " << "variable" << ", " << "controlflow" << ", " << "doubleround" << ", " << "inexactround" << ", " << "dynamic" << ", " << "mxfp" << ", " << "int64" << ", " << "mxfp_conv" << ", " << "shape")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_ExtensionAttr parameter 'value' which is to be a `::mlir::tosa::Extension`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return ExtensionAttr::get(odsParser.getContext(),
      ::mlir::tosa::Extension((*_result_value)));
}

void ExtensionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyExtension(getValue());
  odsPrinter << ">";
}

::mlir::tosa::Extension ExtensionAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ExtensionAttr)
namespace mlir::tosa {

namespace detail {

struct LevelAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::tosa::Level>;
  LevelAttrStorage(::mlir::tosa::Level value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static LevelAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<LevelAttrStorage>()) LevelAttrStorage(std::move(value));
  }

  ::mlir::tosa::Level value;
};

} // namespace detail
LevelAttr LevelAttr::get(::mlir::MLIRContext *context, ::mlir::tosa::Level value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute LevelAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::tosa::Level> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::tosa::Level> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::tosa::symbolizeLevel(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::tosa::Level" << " to be one of: " << "none" << ", " << "8k")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_LevelAttr parameter 'value' which is to be a `::mlir::tosa::Level`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return LevelAttr::get(odsParser.getContext(),
      ::mlir::tosa::Level((*_result_value)));
}

void LevelAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyLevel(getValue());
  odsPrinter << ">";
}

::mlir::tosa::Level LevelAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::LevelAttr)
namespace mlir::tosa {

namespace detail {

struct SpecificationVersionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::tosa::SpecificationVersion>;
  SpecificationVersionAttrStorage(::mlir::tosa::SpecificationVersion value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static SpecificationVersionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<SpecificationVersionAttrStorage>()) SpecificationVersionAttrStorage(std::move(value));
  }

  ::mlir::tosa::SpecificationVersion value;
};

} // namespace detail
SpecificationVersionAttr SpecificationVersionAttr::get(::mlir::MLIRContext *context, ::mlir::tosa::SpecificationVersion value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute SpecificationVersionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::tosa::SpecificationVersion> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::tosa::SpecificationVersion> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::tosa::symbolizeSpecificationVersion(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::tosa::SpecificationVersion" << " to be one of: " << "1.0" << ", " << "1.1.draft")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_SpecificationVersion parameter 'value' which is to be a `::mlir::tosa::SpecificationVersion`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return SpecificationVersionAttr::get(odsParser.getContext(),
      ::mlir::tosa::SpecificationVersion((*_result_value)));
}

void SpecificationVersionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifySpecificationVersion(getValue());
  odsPrinter << ">";
}

::mlir::tosa::SpecificationVersion SpecificationVersionAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::SpecificationVersionAttr)
namespace mlir::tosa {

namespace detail {

struct TargetEnvAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<SpecificationVersion, Level, ::llvm::ArrayRef<Profile>, ::llvm::ArrayRef<Extension>>;
  TargetEnvAttrStorage(SpecificationVersion specification_version, Level level, ::llvm::ArrayRef<Profile> profiles, ::llvm::ArrayRef<Extension> extensions) : specification_version(std::move(specification_version)), level(std::move(level)), profiles(std::move(profiles)), extensions(std::move(extensions)) {}

  KeyTy getAsKey() const {
    return KeyTy(specification_version, level, profiles, extensions);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (specification_version == std::get<0>(tblgenKey)) && (level == std::get<1>(tblgenKey)) && (profiles == std::get<2>(tblgenKey)) && (extensions == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static TargetEnvAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto specification_version = std::move(std::get<0>(tblgenKey));
    auto level = std::move(std::get<1>(tblgenKey));
    auto profiles = std::move(std::get<2>(tblgenKey));
    auto extensions = std::move(std::get<3>(tblgenKey));
    profiles = allocator.copyInto(profiles);
    extensions = allocator.copyInto(extensions);
    return new (allocator.allocate<TargetEnvAttrStorage>()) TargetEnvAttrStorage(std::move(specification_version), std::move(level), std::move(profiles), std::move(extensions));
  }

  SpecificationVersion specification_version;
  Level level;
  ::llvm::ArrayRef<Profile> profiles;
  ::llvm::ArrayRef<Extension> extensions;
};

} // namespace detail
TargetEnvAttr TargetEnvAttr::get(::mlir::MLIRContext *context, SpecificationVersion specification_version, Level level, ::llvm::ArrayRef<Profile> profiles, ::llvm::ArrayRef<Extension> extensions) {
  return Base::get(context, std::move(specification_version), std::move(level), std::move(profiles), std::move(extensions));
}

::mlir::Attribute TargetEnvAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<SpecificationVersion> _result_specification_version;
  ::mlir::FailureOr<Level> _result_level;
  ::mlir::FailureOr<::llvm::SmallVector<Profile>> _result_profiles;
  ::mlir::FailureOr<::llvm::SmallVector<Extension>> _result_extensions;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse literal 'specification_version'
  if (odsParser.parseKeyword("specification_version")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'specification_version'
  _result_specification_version = ::mlir::FieldParser<SpecificationVersion>::parse(odsParser);
  if (::mlir::failed(_result_specification_version)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_TargetEnv parameter 'specification_version' which is to be a `SpecificationVersion`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'level'
  if (odsParser.parseKeyword("level")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};

  // Parse variable 'level'
  _result_level = ::mlir::FieldParser<Level>::parse(odsParser);
  if (::mlir::failed(_result_level)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_TargetEnv parameter 'level' which is to be a `Level`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'profiles'
  if (odsParser.parseKeyword("profiles")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'profiles'
  _result_profiles = ::mlir::FieldParser<::llvm::SmallVector<Profile>>::parse(odsParser);
  if (::mlir::failed(_result_profiles)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_TargetEnv parameter 'profiles' which is to be a `::llvm::ArrayRef<Profile>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal ','
  if (odsParser.parseComma()) return {};
  // Parse literal 'extensions'
  if (odsParser.parseKeyword("extensions")) return {};
  // Parse literal '='
  if (odsParser.parseEqual()) return {};
  // Parse literal '['
  if (odsParser.parseLSquare()) return {};

  // Parse variable 'extensions'
  _result_extensions = ::mlir::FieldParser<::llvm::SmallVector<Extension>>::parse(odsParser);
  if (::mlir::failed(_result_extensions)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_TargetEnv parameter 'extensions' which is to be a `::llvm::ArrayRef<Extension>`");
    return {};
  }
  // Parse literal ']'
  if (odsParser.parseRSquare()) return {};
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_specification_version));
  assert(::mlir::succeeded(_result_level));
  assert(::mlir::succeeded(_result_profiles));
  assert(::mlir::succeeded(_result_extensions));
  return TargetEnvAttr::get(odsParser.getContext(),
      SpecificationVersion((*_result_specification_version)),
      Level((*_result_level)),
      ::llvm::ArrayRef<Profile>((*_result_profiles)),
      ::llvm::ArrayRef<Extension>((*_result_extensions)));
}

void TargetEnvAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << "specification_version";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getSpecificationVersion());
  odsPrinter << ",";
  odsPrinter << ' ' << "level";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLevel());
  odsPrinter << ",";
  odsPrinter << ' ' << "profiles";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ' << "[";
  odsPrinter.printStrippedAttrOrType(getProfiles());
  odsPrinter << "]";
  odsPrinter << ",";
  odsPrinter << ' ' << "extensions";
  odsPrinter << ' ' << "=";
  odsPrinter << ' ' << "[";
  odsPrinter.printStrippedAttrOrType(getExtensions());
  odsPrinter << "]";
  odsPrinter << ">";
}

SpecificationVersion TargetEnvAttr::getSpecificationVersion() const {
  return getImpl()->specification_version;
}

Level TargetEnvAttr::getLevel() const {
  return getImpl()->level;
}

::llvm::ArrayRef<Profile> TargetEnvAttr::getProfiles() const {
  return getImpl()->profiles;
}

::llvm::ArrayRef<Extension> TargetEnvAttr::getExtensions() const {
  return getImpl()->extensions;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::TargetEnvAttr)
namespace mlir::tosa {

namespace detail {

struct ResizeModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::tosa::ResizeMode>;
  ResizeModeAttrStorage(::mlir::tosa::ResizeMode value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static ResizeModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<ResizeModeAttrStorage>()) ResizeModeAttrStorage(std::move(value));
  }

  ::mlir::tosa::ResizeMode value;
};

} // namespace detail
ResizeModeAttr ResizeModeAttr::get(::mlir::MLIRContext *context, ::mlir::tosa::ResizeMode value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute ResizeModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::tosa::ResizeMode> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::tosa::ResizeMode> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::tosa::symbolizeResizeMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::tosa::ResizeMode" << " to be one of: " << "NEAREST_NEIGHBOR" << ", " << "BILINEAR")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_ResizeModeAttr parameter 'value' which is to be a `::mlir::tosa::ResizeMode`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return ResizeModeAttr::get(odsParser.getContext(),
      ::mlir::tosa::ResizeMode((*_result_value)));
}

void ResizeModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyResizeMode(getValue());
  odsPrinter << ">";
}

::mlir::tosa::ResizeMode ResizeModeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::ResizeModeAttr)
namespace mlir::tosa {

namespace detail {

struct NanPropagationModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::tosa::NanPropagationMode>;
  NanPropagationModeAttrStorage(::mlir::tosa::NanPropagationMode value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static NanPropagationModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<NanPropagationModeAttrStorage>()) NanPropagationModeAttrStorage(std::move(value));
  }

  ::mlir::tosa::NanPropagationMode value;
};

} // namespace detail
NanPropagationModeAttr NanPropagationModeAttr::get(::mlir::MLIRContext *context, ::mlir::tosa::NanPropagationMode value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute NanPropagationModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::tosa::NanPropagationMode> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::tosa::NanPropagationMode> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::tosa::symbolizeNanPropagationMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::tosa::NanPropagationMode" << " to be one of: " << "PROPAGATE" << ", " << "IGNORE")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_NanPropagationModeAttr parameter 'value' which is to be a `::mlir::tosa::NanPropagationMode`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return NanPropagationModeAttr::get(odsParser.getContext(),
      ::mlir::tosa::NanPropagationMode((*_result_value)));
}

void NanPropagationModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyNanPropagationMode(getValue());
  odsPrinter << ">";
}

::mlir::tosa::NanPropagationMode NanPropagationModeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::NanPropagationModeAttr)
namespace mlir::tosa {

namespace detail {

struct RoundingModeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::tosa::RoundingMode>;
  RoundingModeAttrStorage(::mlir::tosa::RoundingMode value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static RoundingModeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<RoundingModeAttrStorage>()) RoundingModeAttrStorage(std::move(value));
  }

  ::mlir::tosa::RoundingMode value;
};

} // namespace detail
RoundingModeAttr RoundingModeAttr::get(::mlir::MLIRContext *context, ::mlir::tosa::RoundingMode value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute RoundingModeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::tosa::RoundingMode> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::tosa::RoundingMode> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::tosa::symbolizeRoundingMode(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::tosa::RoundingMode" << " to be one of: " << "SINGLE_ROUND" << ", " << "INEXACT_ROUND" << ", " << "DOUBLE_ROUND")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_RoundingModeAttr parameter 'value' which is to be a `::mlir::tosa::RoundingMode`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return RoundingModeAttr::get(odsParser.getContext(),
      ::mlir::tosa::RoundingMode((*_result_value)));
}

void RoundingModeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyRoundingMode(getValue());
  odsPrinter << ">";
}

::mlir::tosa::RoundingMode RoundingModeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::RoundingModeAttr)
namespace mlir::tosa {

namespace detail {

struct BlockSizeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::mlir::tosa::BlockSize>;
  BlockSizeAttrStorage(::mlir::tosa::BlockSize value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static BlockSizeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<BlockSizeAttrStorage>()) BlockSizeAttrStorage(std::move(value));
  }

  ::mlir::tosa::BlockSize value;
};

} // namespace detail
BlockSizeAttr BlockSizeAttr::get(::mlir::MLIRContext *context, ::mlir::tosa::BlockSize value) {
  return Base::get(context, std::move(value));
}

::mlir::Attribute BlockSizeAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<::mlir::tosa::BlockSize> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = [&]() -> ::mlir::FailureOr<::mlir::tosa::BlockSize> {
      auto loc = odsParser.getCurrentLocation();
      std::string enumKeyword;
      if (::mlir::failed(odsParser.parseKeywordOrString(&enumKeyword)))
        return ::mlir::failure();
      auto maybeEnum = ::mlir::tosa::symbolizeBlockSize(enumKeyword);
      if (maybeEnum)
        return *maybeEnum;
      return {(::llvm::LogicalResult)(odsParser.emitError(loc) << "expected " << "::mlir::tosa::BlockSize" << " to be one of: " << "BLOCK_SIZE_32")};
    }();
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Tosa_BlockSizeAttr parameter 'value' which is to be a `::mlir::tosa::BlockSize`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return BlockSizeAttr::get(odsParser.getContext(),
      ::mlir::tosa::BlockSize((*_result_value)));
}

void BlockSizeAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter << stringifyBlockSize(getValue());
  odsPrinter << ">";
}

::mlir::tosa::BlockSize BlockSizeAttr::getValue() const {
  return getImpl()->value;
}


} // namespace mlir::tosa
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::tosa::BlockSizeAttr)
namespace mlir::tosa {


/// Parse an attribute registered to this dialect.
::mlir::Attribute TosaDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void TosaDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::tosa

#endif // GET_ATTRDEF_CLASSES

