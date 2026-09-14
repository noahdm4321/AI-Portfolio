/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr,
::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr,
::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr

#endif // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::llvm::LogicalResult>(def)    .Case<::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr>([&](auto t) {
      printer << ::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>([&](auto t) {
      printer << ::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::triton::nvidia_gpu {


} // namespace mlir::triton::nvidia_gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemorySpaceAttr)
namespace mlir::triton::nvidia_gpu {

namespace detail {

struct TensorMemoryEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, bool>;
  TensorMemoryEncodingAttrStorage(unsigned blockM, unsigned blockN, unsigned colStride, unsigned CTASplitM, unsigned CTASplitN, bool twoCTAs) : blockM(std::move(blockM)), blockN(std::move(blockN)), colStride(std::move(colStride)), CTASplitM(std::move(CTASplitM)), CTASplitN(std::move(CTASplitN)), twoCTAs(std::move(twoCTAs)) {}

  KeyTy getAsKey() const {
    return KeyTy(blockM, blockN, colStride, CTASplitM, CTASplitN, twoCTAs);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (blockM == std::get<0>(tblgenKey)) && (blockN == std::get<1>(tblgenKey)) && (colStride == std::get<2>(tblgenKey)) && (CTASplitM == std::get<3>(tblgenKey)) && (CTASplitN == std::get<4>(tblgenKey)) && (twoCTAs == std::get<5>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey));
  }

  static TensorMemoryEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto blockM = std::move(std::get<0>(tblgenKey));
    auto blockN = std::move(std::get<1>(tblgenKey));
    auto colStride = std::move(std::get<2>(tblgenKey));
    auto CTASplitM = std::move(std::get<3>(tblgenKey));
    auto CTASplitN = std::move(std::get<4>(tblgenKey));
    auto twoCTAs = std::move(std::get<5>(tblgenKey));
    return new (allocator.allocate<TensorMemoryEncodingAttrStorage>()) TensorMemoryEncodingAttrStorage(std::move(blockM), std::move(blockN), std::move(colStride), std::move(CTASplitM), std::move(CTASplitN), std::move(twoCTAs));
  }

  unsigned blockM;
  unsigned blockN;
  unsigned colStride;
  unsigned CTASplitM;
  unsigned CTASplitN;
  bool twoCTAs;
};

} // namespace detail
TensorMemoryEncodingAttr TensorMemoryEncodingAttr::get(::mlir::MLIRContext *context, unsigned blockM, unsigned blockN, unsigned colStride, unsigned CTASplitM, unsigned CTASplitN, bool twoCTAs) {
  return Base::get(context, std::move(blockM), std::move(blockN), std::move(colStride), std::move(CTASplitM), std::move(CTASplitN), std::move(twoCTAs));
}

TensorMemoryEncodingAttr TensorMemoryEncodingAttr::getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned blockM, unsigned blockN, unsigned colStride, unsigned CTASplitM, unsigned CTASplitN, bool twoCTAs) {
  return Base::getChecked(emitError, context, std::move(blockM), std::move(blockN), std::move(colStride), std::move(CTASplitM), std::move(CTASplitN), std::move(twoCTAs));
}

::llvm::LogicalResult TensorMemoryEncodingAttr::verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned blockM, unsigned blockN, unsigned colStride, unsigned CTASplitM, unsigned CTASplitN, bool twoCTAs) {
  if (::mlir::failed(verify(emitError, blockM, blockN, colStride, CTASplitM, CTASplitN, twoCTAs)))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::Attribute TensorMemoryEncodingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<unsigned> _result_blockM;
  ::mlir::FailureOr<unsigned> _result_blockN;
  ::mlir::FailureOr<unsigned> _result_colStride;
  ::mlir::FailureOr<unsigned> _result_CTASplitM;
  ::mlir::FailureOr<unsigned> _result_CTASplitN;
  ::mlir::FailureOr<bool> _result_twoCTAs;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_blockM = false;
  bool _seen_blockN = false;
  bool _seen_colStride = false;
  bool _seen_CTASplitM = false;
  bool _seen_CTASplitN = false;
  bool _seen_twoCTAs = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_blockM && _paramKey == "blockM") {
        _seen_blockM = true;

        // Parse variable 'blockM'
        _result_blockM = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_blockM)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TTG_TensorMemoryEncodingAttr parameter 'blockM' which is to be a `unsigned`");
          return {};
        }
      } else if (!_seen_blockN && _paramKey == "blockN") {
        _seen_blockN = true;

        // Parse variable 'blockN'
        _result_blockN = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_blockN)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TTG_TensorMemoryEncodingAttr parameter 'blockN' which is to be a `unsigned`");
          return {};
        }
      } else if (!_seen_colStride && _paramKey == "colStride") {
        _seen_colStride = true;

        // Parse variable 'colStride'
        _result_colStride = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_colStride)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TTG_TensorMemoryEncodingAttr parameter 'colStride' which is to be a `unsigned`");
          return {};
        }
      } else if (!_seen_CTASplitM && _paramKey == "CTASplitM") {
        _seen_CTASplitM = true;

        // Parse variable 'CTASplitM'
        _result_CTASplitM = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_CTASplitM)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TTG_TensorMemoryEncodingAttr parameter 'CTASplitM' which is to be a `unsigned`");
          return {};
        }
      } else if (!_seen_CTASplitN && _paramKey == "CTASplitN") {
        _seen_CTASplitN = true;

        // Parse variable 'CTASplitN'
        _result_CTASplitN = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_CTASplitN)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TTG_TensorMemoryEncodingAttr parameter 'CTASplitN' which is to be a `unsigned`");
          return {};
        }
      } else if (!_seen_twoCTAs && _paramKey == "twoCTAs") {
        _seen_twoCTAs = true;

        // Parse variable 'twoCTAs'
        _result_twoCTAs = ::mlir::FieldParser<bool>::parse(odsParser);
        if (::mlir::failed(_result_twoCTAs)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TTG_TensorMemoryEncodingAttr parameter 'twoCTAs' which is to be a `bool`");
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
    if (!_seen_blockM) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "blockM";
      return {};
    }
    if (!_seen_blockN) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "blockN";
      return {};
    }
    if (!_seen_colStride) {
      odsParser.emitError(odsParser.getCurrentLocation(), "struct is missing required parameter: ") << "colStride";
      return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_blockM));
  assert(::mlir::succeeded(_result_blockN));
  assert(::mlir::succeeded(_result_colStride));
  return odsParser.getChecked<TensorMemoryEncodingAttr>(odsLoc, odsParser.getContext(),
      unsigned((*_result_blockM)),
      unsigned((*_result_blockN)),
      unsigned((*_result_colStride)),
      unsigned((_result_CTASplitM.value_or(1))),
      unsigned((_result_CTASplitN.value_or(1))),
      bool((_result_twoCTAs.value_or(false))));
}

void TensorMemoryEncodingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "blockM = ";
    odsPrinter.printStrippedAttrOrType(getBlockM());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "blockN = ";
    odsPrinter.printStrippedAttrOrType(getBlockN());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "colStride = ";
    odsPrinter.printStrippedAttrOrType(getColStride());
    if (!(getCTASplitM() == 1)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "CTASplitM = ";
      if (!(getCTASplitM() == 1)) {
        odsPrinter.printStrippedAttrOrType(getCTASplitM());
      }
    }
    if (!(getCTASplitN() == 1)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "CTASplitN = ";
      if (!(getCTASplitN() == 1)) {
        odsPrinter.printStrippedAttrOrType(getCTASplitN());
      }
    }
    if (!(getTwoCTAs() == false)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "twoCTAs = ";
      if (!(getTwoCTAs() == false)) {
        odsPrinter.printStrippedAttrOrType(getTwoCTAs());
      }
    }
  }
  odsPrinter << ">";
}

unsigned TensorMemoryEncodingAttr::getBlockM() const {
  return getImpl()->blockM;
}

unsigned TensorMemoryEncodingAttr::getBlockN() const {
  return getImpl()->blockN;
}

unsigned TensorMemoryEncodingAttr::getColStride() const {
  return getImpl()->colStride;
}

unsigned TensorMemoryEncodingAttr::getCTASplitM() const {
  return getImpl()->CTASplitM;
}

unsigned TensorMemoryEncodingAttr::getCTASplitN() const {
  return getImpl()->CTASplitN;
}

bool TensorMemoryEncodingAttr::getTwoCTAs() const {
  return getImpl()->twoCTAs;
}


} // namespace mlir::triton::nvidia_gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr)
namespace mlir::triton::nvidia_gpu {

namespace detail {

struct TensorMemoryScalesEncodingAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<unsigned, unsigned>;
  TensorMemoryScalesEncodingAttrStorage(unsigned CTASplitM, unsigned CTASplitN) : CTASplitM(std::move(CTASplitM)), CTASplitN(std::move(CTASplitN)) {}

  KeyTy getAsKey() const {
    return KeyTy(CTASplitM, CTASplitN);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (CTASplitM == std::get<0>(tblgenKey)) && (CTASplitN == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static TensorMemoryScalesEncodingAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto CTASplitM = std::move(std::get<0>(tblgenKey));
    auto CTASplitN = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<TensorMemoryScalesEncodingAttrStorage>()) TensorMemoryScalesEncodingAttrStorage(std::move(CTASplitM), std::move(CTASplitN));
  }

  unsigned CTASplitM;
  unsigned CTASplitN;
};

} // namespace detail
TensorMemoryScalesEncodingAttr TensorMemoryScalesEncodingAttr::get(::mlir::MLIRContext *context, unsigned CTASplitM, unsigned CTASplitN) {
  return Base::get(context, std::move(CTASplitM), std::move(CTASplitN));
}

::mlir::Attribute TensorMemoryScalesEncodingAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<unsigned> _result_CTASplitM;
  ::mlir::FailureOr<unsigned> _result_CTASplitN;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_CTASplitM = false;
  bool _seen_CTASplitN = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_CTASplitM && _paramKey == "CTASplitM") {
        _seen_CTASplitM = true;

        // Parse variable 'CTASplitM'
        _result_CTASplitM = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_CTASplitM)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TTG_TensorMemoryScalesEncodingAttr parameter 'CTASplitM' which is to be a `unsigned`");
          return {};
        }
      } else if (!_seen_CTASplitN && _paramKey == "CTASplitN") {
        _seen_CTASplitN = true;

        // Parse variable 'CTASplitN'
        _result_CTASplitN = ::mlir::FieldParser<unsigned>::parse(odsParser);
        if (::mlir::failed(_result_CTASplitN)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TTG_TensorMemoryScalesEncodingAttr parameter 'CTASplitN' which is to be a `unsigned`");
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
  return TensorMemoryScalesEncodingAttr::get(odsParser.getContext(),
      unsigned((_result_CTASplitM.value_or(1))),
      unsigned((_result_CTASplitN.value_or(1))));
}

void TensorMemoryScalesEncodingAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!(getCTASplitM() == 1)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "CTASplitM = ";
      if (!(getCTASplitM() == 1)) {
        odsPrinter.printStrippedAttrOrType(getCTASplitM());
      }
    }
    if (!(getCTASplitN() == 1)) {
      if (!_firstPrinted) odsPrinter << ", ";
      _firstPrinted = false;
      odsPrinter << "CTASplitN = ";
      if (!(getCTASplitN() == 1)) {
        odsPrinter.printStrippedAttrOrType(getCTASplitN());
      }
    }
  }
  odsPrinter << ">";
}

unsigned TensorMemoryScalesEncodingAttr::getCTASplitM() const {
  return getImpl()->CTASplitM;
}

unsigned TensorMemoryScalesEncodingAttr::getCTASplitN() const {
  return getImpl()->CTASplitN;
}


} // namespace mlir::triton::nvidia_gpu
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr)
namespace mlir::triton::nvidia_gpu {


/// Parse an attribute registered to this dialect.
::mlir::Attribute TritonNvidiaGPUDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void TritonNvidiaGPUDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}

} // namespace mlir::triton::nvidia_gpu

#endif // GET_ATTRDEF_CLASSES

