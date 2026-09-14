/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: triton_xla_attrs.td                                                  *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::xla {

// A comparison operator for instructions.
enum class Comparator : uint32_t {
  EQ = 0,
  LT = 1,
};

::std::optional<Comparator> symbolizeComparator(uint32_t);
::llvm::StringRef stringifyComparator(Comparator);
::std::optional<Comparator> symbolizeComparator(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForComparator() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(Comparator enumValue) {
  return stringifyComparator(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Comparator> symbolizeEnum<Comparator>(::llvm::StringRef str) {
  return symbolizeComparator(str);
}

} // namespace mlir::triton::xla

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::xla::Comparator, ::mlir::triton::xla::Comparator> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::xla::Comparator> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for A comparison operator for instructions.");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::xla::Comparator> attr = ::mlir::triton::xla::symbolizeEnum<::mlir::triton::xla::Comparator>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [equal_to, less_than] for A comparison operator for instructions., got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::xla::Comparator>, std::optional<::mlir::triton::xla::Comparator>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::xla::Comparator>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::xla::Comparator>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::xla::Comparator> attr = ::mlir::triton::xla::symbolizeEnum<::mlir::triton::xla::Comparator>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [equal_to, less_than] for A comparison operator for instructions., got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::xla::Comparator value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::xla::Comparator> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::xla::Comparator getEmptyKey() {
    return static_cast<::mlir::triton::xla::Comparator>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::xla::Comparator getTombstoneKey() {
    return static_cast<::mlir::triton::xla::Comparator>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::xla::Comparator &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::xla::Comparator &lhs, const ::mlir::triton::xla::Comparator &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::xla {

// tensor map swizzle mode
enum class SwizzleMode : uint32_t {
  kNone = 0,
  k32b = 1,
  k64b = 2,
  k128b = 3,
};

::std::optional<SwizzleMode> symbolizeSwizzleMode(uint32_t);
::llvm::StringRef stringifySwizzleMode(SwizzleMode);
::std::optional<SwizzleMode> symbolizeSwizzleMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSwizzleMode() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(SwizzleMode enumValue) {
  return stringifySwizzleMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SwizzleMode> symbolizeEnum<SwizzleMode>(::llvm::StringRef str) {
  return symbolizeSwizzleMode(str);
}

} // namespace mlir::triton::xla

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::xla::SwizzleMode, ::mlir::triton::xla::SwizzleMode> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::xla::SwizzleMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for tensor map swizzle mode");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::xla::SwizzleMode> attr = ::mlir::triton::xla::symbolizeEnum<::mlir::triton::xla::SwizzleMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, \"32b\"32b, \"64b\"64b, \"128b\"128b] for tensor map swizzle mode, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::xla::SwizzleMode>, std::optional<::mlir::triton::xla::SwizzleMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::xla::SwizzleMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::xla::SwizzleMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::xla::SwizzleMode> attr = ::mlir::triton::xla::symbolizeEnum<::mlir::triton::xla::SwizzleMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, \"32b\"32b, \"64b\"64b, \"128b\"128b] for tensor map swizzle mode, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::xla::SwizzleMode value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::triton::xla::SwizzleMode::kNone:
    break;
  default:
    return p << '"' << valueStr << '"';
  }
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::xla::SwizzleMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::xla::SwizzleMode getEmptyKey() {
    return static_cast<::mlir::triton::xla::SwizzleMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::xla::SwizzleMode getTombstoneKey() {
    return static_cast<::mlir::triton::xla::SwizzleMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::xla::SwizzleMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::xla::SwizzleMode &lhs, const ::mlir::triton::xla::SwizzleMode &rhs) {
    return lhs == rhs;
  }
};
}

