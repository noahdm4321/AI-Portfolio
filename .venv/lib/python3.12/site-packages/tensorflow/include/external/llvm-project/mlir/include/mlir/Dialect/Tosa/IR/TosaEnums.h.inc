/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TosaOpBase.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::tosa {

// supported TOSA profiles
enum class Profile : uint32_t {
  pro_int = 1,
  pro_fp = 2,
  none = 0,
};

::std::optional<Profile> symbolizeProfile(uint32_t);
::llvm::StringRef stringifyProfile(Profile);
::std::optional<Profile> symbolizeProfile(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForProfile() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(Profile enumValue) {
  return stringifyProfile(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Profile> symbolizeEnum<Profile>(::llvm::StringRef str) {
  return symbolizeProfile(str);
}

} // namespace mlir::tosa

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::Profile, ::mlir::tosa::Profile> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::Profile> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for supported TOSA profiles");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Profile> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Profile>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [pro_int, pro_fp, none] for supported TOSA profiles, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::Profile>, std::optional<::mlir::tosa::Profile>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::Profile>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::Profile>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Profile> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Profile>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [pro_int, pro_fp, none] for supported TOSA profiles, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::Profile value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::Profile> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::Profile getEmptyKey() {
    return static_cast<::mlir::tosa::Profile>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::Profile getTombstoneKey() {
    return static_cast<::mlir::tosa::Profile>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::Profile &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::Profile &lhs, const ::mlir::tosa::Profile &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::tosa {

// supported TOSA extensions
enum class Extension : uint32_t {
  none = 0,
  int16 = 1,
  int4 = 2,
  bf16 = 3,
  fp8e4m3 = 4,
  fp8e5m2 = 5,
  fft = 6,
  variable = 7,
  controlflow = 8,
  doubleround = 9,
  inexactround = 10,
  dynamic = 11,
  mxfp = 12,
  int64 = 13,
  mxfp_conv = 14,
  shape = 15,
};

::std::optional<Extension> symbolizeExtension(uint32_t);
::llvm::StringRef stringifyExtension(Extension);
::std::optional<Extension> symbolizeExtension(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForExtension() {
  return 15;
}


inline ::llvm::StringRef stringifyEnum(Extension enumValue) {
  return stringifyExtension(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Extension> symbolizeEnum<Extension>(::llvm::StringRef str) {
  return symbolizeExtension(str);
}

} // namespace mlir::tosa

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::Extension, ::mlir::tosa::Extension> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::Extension> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for supported TOSA extensions");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Extension> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Extension>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, int16, int4, bf16, fp8e4m3, fp8e5m2, fft, variable, controlflow, doubleround, inexactround, dynamic, mxfp, int64, mxfp_conv, shape] for supported TOSA extensions, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::Extension>, std::optional<::mlir::tosa::Extension>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::Extension>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::Extension>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Extension> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Extension>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, int16, int4, bf16, fp8e4m3, fp8e5m2, fft, variable, controlflow, doubleround, inexactround, dynamic, mxfp, int64, mxfp_conv, shape] for supported TOSA extensions, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::Extension value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::Extension> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::Extension getEmptyKey() {
    return static_cast<::mlir::tosa::Extension>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::Extension getTombstoneKey() {
    return static_cast<::mlir::tosa::Extension>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::Extension &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::Extension &lhs, const ::mlir::tosa::Extension &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::tosa {

// supported TOSA levels
enum class Level : uint32_t {
  none = 0,
  eightK = 1,
};

::std::optional<Level> symbolizeLevel(uint32_t);
::llvm::StringRef stringifyLevel(Level);
::std::optional<Level> symbolizeLevel(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLevel() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(Level enumValue) {
  return stringifyLevel(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Level> symbolizeEnum<Level>(::llvm::StringRef str) {
  return symbolizeLevel(str);
}

} // namespace mlir::tosa

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::Level, ::mlir::tosa::Level> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::Level> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for supported TOSA levels");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Level> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Level>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, \"8k\"8k] for supported TOSA levels, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::Level>, std::optional<::mlir::tosa::Level>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::Level>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::Level>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::Level> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::Level>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, \"8k\"8k] for supported TOSA levels, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::Level value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::tosa::Level::none:
    break;
  default:
    return p << '"' << valueStr << '"';
  }
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::Level> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::Level getEmptyKey() {
    return static_cast<::mlir::tosa::Level>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::Level getTombstoneKey() {
    return static_cast<::mlir::tosa::Level>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::Level &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::Level &lhs, const ::mlir::tosa::Level &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::tosa {

// TOSA specification version
enum class SpecificationVersion : uint32_t {
  V_1_0 = 0,
  V_1_1_DRAFT = 1,
};

::std::optional<SpecificationVersion> symbolizeSpecificationVersion(uint32_t);
::llvm::StringRef stringifySpecificationVersion(SpecificationVersion);
::std::optional<SpecificationVersion> symbolizeSpecificationVersion(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSpecificationVersion() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(SpecificationVersion enumValue) {
  return stringifySpecificationVersion(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SpecificationVersion> symbolizeEnum<SpecificationVersion>(::llvm::StringRef str) {
  return symbolizeSpecificationVersion(str);
}

} // namespace mlir::tosa

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::SpecificationVersion, ::mlir::tosa::SpecificationVersion> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::SpecificationVersion> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for TOSA specification version");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::SpecificationVersion> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::SpecificationVersion>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [\"1.0\"1.0, \"1.1.draft\"1.1.draft] for TOSA specification version, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::SpecificationVersion>, std::optional<::mlir::tosa::SpecificationVersion>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::SpecificationVersion>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::SpecificationVersion>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::SpecificationVersion> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::SpecificationVersion>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [\"1.0\"1.0, \"1.1.draft\"1.1.draft] for TOSA specification version, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::SpecificationVersion value) {
  auto valueStr = stringifyEnum(value);
  return p << '"' << valueStr << '"';
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::SpecificationVersion> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::SpecificationVersion getEmptyKey() {
    return static_cast<::mlir::tosa::SpecificationVersion>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::SpecificationVersion getTombstoneKey() {
    return static_cast<::mlir::tosa::SpecificationVersion>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::SpecificationVersion &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::SpecificationVersion &lhs, const ::mlir::tosa::SpecificationVersion &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::tosa {

// Supported resize/upsampling strategies
enum class ResizeMode : uint32_t {
  NEAREST_NEIGHBOR = 1,
  BILINEAR = 2,
};

::std::optional<ResizeMode> symbolizeResizeMode(uint32_t);
::llvm::StringRef stringifyResizeMode(ResizeMode);
::std::optional<ResizeMode> symbolizeResizeMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForResizeMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(ResizeMode enumValue) {
  return stringifyResizeMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ResizeMode> symbolizeEnum<ResizeMode>(::llvm::StringRef str) {
  return symbolizeResizeMode(str);
}

} // namespace mlir::tosa

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::ResizeMode, ::mlir::tosa::ResizeMode> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::ResizeMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Supported resize/upsampling strategies");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::ResizeMode> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::ResizeMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [NEAREST_NEIGHBOR, BILINEAR] for Supported resize/upsampling strategies, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::ResizeMode>, std::optional<::mlir::tosa::ResizeMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::ResizeMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::ResizeMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::ResizeMode> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::ResizeMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [NEAREST_NEIGHBOR, BILINEAR] for Supported resize/upsampling strategies, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::ResizeMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::ResizeMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::ResizeMode getEmptyKey() {
    return static_cast<::mlir::tosa::ResizeMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::ResizeMode getTombstoneKey() {
    return static_cast<::mlir::tosa::ResizeMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::ResizeMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::ResizeMode &lhs, const ::mlir::tosa::ResizeMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::tosa {

// Supported NaN propagation strategies
enum class NanPropagationMode : uint32_t {
  PROPAGATE = 1,
  IGNORE = 2,
};

::std::optional<NanPropagationMode> symbolizeNanPropagationMode(uint32_t);
::llvm::StringRef stringifyNanPropagationMode(NanPropagationMode);
::std::optional<NanPropagationMode> symbolizeNanPropagationMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForNanPropagationMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(NanPropagationMode enumValue) {
  return stringifyNanPropagationMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<NanPropagationMode> symbolizeEnum<NanPropagationMode>(::llvm::StringRef str) {
  return symbolizeNanPropagationMode(str);
}

} // namespace mlir::tosa

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::NanPropagationMode, ::mlir::tosa::NanPropagationMode> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::NanPropagationMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Supported NaN propagation strategies");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::NanPropagationMode> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::NanPropagationMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [PROPAGATE, IGNORE] for Supported NaN propagation strategies, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::NanPropagationMode>, std::optional<::mlir::tosa::NanPropagationMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::NanPropagationMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::NanPropagationMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::NanPropagationMode> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::NanPropagationMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [PROPAGATE, IGNORE] for Supported NaN propagation strategies, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::NanPropagationMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::NanPropagationMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::NanPropagationMode getEmptyKey() {
    return static_cast<::mlir::tosa::NanPropagationMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::NanPropagationMode getTombstoneKey() {
    return static_cast<::mlir::tosa::NanPropagationMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::NanPropagationMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::NanPropagationMode &lhs, const ::mlir::tosa::NanPropagationMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::tosa {

// Supported rounding modes
enum class RoundingMode : uint32_t {
  SINGLE_ROUND = 1,
  INEXACT_ROUND = 2,
  DOUBLE_ROUND = 3,
};

::std::optional<RoundingMode> symbolizeRoundingMode(uint32_t);
::llvm::StringRef stringifyRoundingMode(RoundingMode);
::std::optional<RoundingMode> symbolizeRoundingMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRoundingMode() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(RoundingMode enumValue) {
  return stringifyRoundingMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RoundingMode> symbolizeEnum<RoundingMode>(::llvm::StringRef str) {
  return symbolizeRoundingMode(str);
}

} // namespace mlir::tosa

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::RoundingMode, ::mlir::tosa::RoundingMode> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::RoundingMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Supported rounding modes");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::RoundingMode> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::RoundingMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [SINGLE_ROUND, INEXACT_ROUND, DOUBLE_ROUND] for Supported rounding modes, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::RoundingMode>, std::optional<::mlir::tosa::RoundingMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::RoundingMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::RoundingMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::RoundingMode> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::RoundingMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [SINGLE_ROUND, INEXACT_ROUND, DOUBLE_ROUND] for Supported rounding modes, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::RoundingMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::RoundingMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::RoundingMode getEmptyKey() {
    return static_cast<::mlir::tosa::RoundingMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::RoundingMode getTombstoneKey() {
    return static_cast<::mlir::tosa::RoundingMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::RoundingMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::RoundingMode &lhs, const ::mlir::tosa::RoundingMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::tosa {

// Block size for the block_scaled formats
enum class BlockSize : uint32_t {
  BLOCK_SIZE_32 = 32,
};

::std::optional<BlockSize> symbolizeBlockSize(uint32_t);
::llvm::StringRef stringifyBlockSize(BlockSize);
::std::optional<BlockSize> symbolizeBlockSize(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForBlockSize() {
  return 32;
}


inline ::llvm::StringRef stringifyEnum(BlockSize enumValue) {
  return stringifyBlockSize(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<BlockSize> symbolizeEnum<BlockSize>(::llvm::StringRef str) {
  return symbolizeBlockSize(str);
}

} // namespace mlir::tosa

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::tosa::BlockSize, ::mlir::tosa::BlockSize> {
  template <typename ParserT>
  static FailureOr<::mlir::tosa::BlockSize> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Block size for the block_scaled formats");

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::BlockSize> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::BlockSize>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [BLOCK_SIZE_32] for Block size for the block_scaled formats, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::tosa::BlockSize>, std::optional<::mlir::tosa::BlockSize>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::tosa::BlockSize>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::tosa::BlockSize>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::tosa::BlockSize> attr = ::mlir::tosa::symbolizeEnum<::mlir::tosa::BlockSize>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [BLOCK_SIZE_32] for Block size for the block_scaled formats, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::tosa::BlockSize value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::tosa::BlockSize> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tosa::BlockSize getEmptyKey() {
    return static_cast<::mlir::tosa::BlockSize>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tosa::BlockSize getTombstoneKey() {
    return static_cast<::mlir::tosa::BlockSize>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tosa::BlockSize &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tosa::BlockSize &lhs, const ::mlir::tosa::BlockSize &rhs) {
    return lhs == rhs;
  }
};
}

