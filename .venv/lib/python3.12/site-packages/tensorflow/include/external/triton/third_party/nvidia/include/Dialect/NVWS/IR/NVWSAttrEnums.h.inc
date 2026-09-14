/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVWSAttrDefs.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvws {

// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5
enum class AsyncOp : uint32_t {
  NONE = 0,
  TMALoad = 1,
  TC5MMA = 2,
  TMEMCopy = 3,
  CpAsync = 4,
  WGMMA = 5,
};

::std::optional<AsyncOp> symbolizeAsyncOp(uint32_t);
::llvm::StringRef stringifyAsyncOp(AsyncOp);
::std::optional<AsyncOp> symbolizeAsyncOp(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAsyncOp() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(AsyncOp enumValue) {
  return stringifyAsyncOp(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AsyncOp> symbolizeEnum<AsyncOp>(::llvm::StringRef str) {
  return symbolizeAsyncOp(str);
}

} // namespace mlir::triton::nvws

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::nvws::AsyncOp, ::mlir::triton::nvws::AsyncOp> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::nvws::AsyncOp> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvws::AsyncOp> attr = ::mlir::triton::nvws::symbolizeEnum<::mlir::triton::nvws::AsyncOp>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, tma_load, tc5mma, tmem_copy, cp_async, wgmma] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::nvws::AsyncOp>, std::optional<::mlir::triton::nvws::AsyncOp>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::nvws::AsyncOp>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::nvws::AsyncOp>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvws::AsyncOp> attr = ::mlir::triton::nvws::symbolizeEnum<::mlir::triton::nvws::AsyncOp>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, tma_load, tc5mma, tmem_copy, cp_async, wgmma] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::nvws::AsyncOp value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::nvws::AsyncOp> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::nvws::AsyncOp getEmptyKey() {
    return static_cast<::mlir::triton::nvws::AsyncOp>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::nvws::AsyncOp getTombstoneKey() {
    return static_cast<::mlir::triton::nvws::AsyncOp>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::nvws::AsyncOp &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::nvws::AsyncOp &lhs, const ::mlir::triton::nvws::AsyncOp &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::nvws {

// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4
enum class TokenLoadType : uint32_t {
  None = 0,
  AsyncLoadOp = 1,
  TMALoadOp = 2,
  LocalStoreOp = 3,
  TmemLoadOp = 4,
};

::std::optional<TokenLoadType> symbolizeTokenLoadType(uint32_t);
::llvm::StringRef stringifyTokenLoadType(TokenLoadType);
::std::optional<TokenLoadType> symbolizeTokenLoadType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTokenLoadType() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(TokenLoadType enumValue) {
  return stringifyTokenLoadType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TokenLoadType> symbolizeEnum<TokenLoadType>(::llvm::StringRef str) {
  return symbolizeTokenLoadType(str);
}

class TokenLoadTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = TokenLoadType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static TokenLoadTypeAttr get(::mlir::MLIRContext *context, TokenLoadType val);
  TokenLoadType getValue() const;
};

} // namespace mlir::triton::nvws

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::nvws::TokenLoadType, ::mlir::triton::nvws::TokenLoadType> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::nvws::TokenLoadType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvws::TokenLoadType> attr = ::mlir::triton::nvws::symbolizeEnum<::mlir::triton::nvws::TokenLoadType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, asyncLoadOp, tmaLoadOp, localStoreOp, TmemLoadOp] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::nvws::TokenLoadType>, std::optional<::mlir::triton::nvws::TokenLoadType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::nvws::TokenLoadType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::nvws::TokenLoadType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvws::TokenLoadType> attr = ::mlir::triton::nvws::symbolizeEnum<::mlir::triton::nvws::TokenLoadType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, asyncLoadOp, tmaLoadOp, localStoreOp, TmemLoadOp] for allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::nvws::TokenLoadType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::nvws::TokenLoadType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::nvws::TokenLoadType getEmptyKey() {
    return static_cast<::mlir::triton::nvws::TokenLoadType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::nvws::TokenLoadType getTombstoneKey() {
    return static_cast<::mlir::triton::nvws::TokenLoadType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::nvws::TokenLoadType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::nvws::TokenLoadType &lhs, const ::mlir::triton::nvws::TokenLoadType &rhs) {
    return lhs == rhs;
  }
};
}

