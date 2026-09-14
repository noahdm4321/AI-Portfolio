/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonInstrumentOps.td                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::instrument {

// allowed 32-bit signless integer cases: 0, 1
enum class MemType : uint32_t {
  SHARED_MEM = 0,
  TENSOR_MEM = 1,
};

::std::optional<MemType> symbolizeMemType(uint32_t);
::llvm::StringRef stringifyMemType(MemType);
::std::optional<MemType> symbolizeMemType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMemType() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(MemType enumValue) {
  return stringifyMemType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MemType> symbolizeEnum<MemType>(::llvm::StringRef str) {
  return symbolizeMemType(str);
}

class MemTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MemType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MemTypeAttr get(::mlir::MLIRContext *context, MemType val);
  MemType getValue() const;
};

} // namespace mlir::triton::instrument

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::instrument::MemType, ::mlir::triton::instrument::MemType> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::instrument::MemType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::instrument::MemType> attr = ::mlir::triton::instrument::symbolizeEnum<::mlir::triton::instrument::MemType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [shared_mem, tensor_mem] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::instrument::MemType>, std::optional<::mlir::triton::instrument::MemType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::instrument::MemType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::instrument::MemType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::instrument::MemType> attr = ::mlir::triton::instrument::symbolizeEnum<::mlir::triton::instrument::MemType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [shared_mem, tensor_mem] for allowed 32-bit signless integer cases: 0, 1, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::instrument::MemType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::instrument::MemType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::instrument::MemType getEmptyKey() {
    return static_cast<::mlir::triton::instrument::MemType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::instrument::MemType getTombstoneKey() {
    return static_cast<::mlir::triton::instrument::MemType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::instrument::MemType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::instrument::MemType &lhs, const ::mlir::triton::instrument::MemType &rhs) {
    return lhs == rhs;
  }
};
}

