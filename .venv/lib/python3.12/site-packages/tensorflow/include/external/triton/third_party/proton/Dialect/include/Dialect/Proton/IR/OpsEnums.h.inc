/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ProtonAttrDefs.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::proton {

// The type of metric to be profiled
enum class MetricType : uint32_t {
  CYCLE = 0,
};

::std::optional<MetricType> symbolizeMetricType(uint32_t);
::llvm::StringRef stringifyMetricType(MetricType);
::std::optional<MetricType> symbolizeMetricType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMetricType() {
  return 0;
}


inline ::llvm::StringRef stringifyEnum(MetricType enumValue) {
  return stringifyMetricType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MetricType> symbolizeEnum<MetricType>(::llvm::StringRef str) {
  return symbolizeMetricType(str);
}

class MetricTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = MetricType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static MetricTypeAttr get(::mlir::MLIRContext *context, MetricType val);
  MetricType getValue() const;
};

} // namespace mlir::triton::proton

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::proton::MetricType, ::mlir::triton::proton::MetricType> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::proton::MetricType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for The type of metric to be profiled");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::proton::MetricType> attr = ::mlir::triton::proton::symbolizeEnum<::mlir::triton::proton::MetricType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [cycle] for The type of metric to be profiled, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::proton::MetricType>, std::optional<::mlir::triton::proton::MetricType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::proton::MetricType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::proton::MetricType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::proton::MetricType> attr = ::mlir::triton::proton::symbolizeEnum<::mlir::triton::proton::MetricType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [cycle] for The type of metric to be profiled, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::proton::MetricType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::proton::MetricType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::proton::MetricType getEmptyKey() {
    return static_cast<::mlir::triton::proton::MetricType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::proton::MetricType getTombstoneKey() {
    return static_cast<::mlir::triton::proton::MetricType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::proton::MetricType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::proton::MetricType &lhs, const ::mlir::triton::proton::MetricType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::proton {

// The mode of profiling
enum class Mode : uint32_t {
  DEFAULT = 0,
  MMA = 1,
};

::std::optional<Mode> symbolizeMode(uint32_t);
::llvm::StringRef stringifyMode(Mode);
::std::optional<Mode> symbolizeMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMode() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(Mode enumValue) {
  return stringifyMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Mode> symbolizeEnum<Mode>(::llvm::StringRef str) {
  return symbolizeMode(str);
}

class ModeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = Mode;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ModeAttr get(::mlir::MLIRContext *context, Mode val);
  Mode getValue() const;
};

} // namespace mlir::triton::proton

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::proton::Mode, ::mlir::triton::proton::Mode> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::proton::Mode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for The mode of profiling");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::proton::Mode> attr = ::mlir::triton::proton::symbolizeEnum<::mlir::triton::proton::Mode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [default, mma] for The mode of profiling, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::proton::Mode>, std::optional<::mlir::triton::proton::Mode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::proton::Mode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::proton::Mode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::proton::Mode> attr = ::mlir::triton::proton::symbolizeEnum<::mlir::triton::proton::Mode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [default, mma] for The mode of profiling, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::proton::Mode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::proton::Mode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::proton::Mode getEmptyKey() {
    return static_cast<::mlir::triton::proton::Mode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::proton::Mode getTombstoneKey() {
    return static_cast<::mlir::triton::proton::Mode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::proton::Mode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::proton::Mode &lhs, const ::mlir::triton::proton::Mode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::triton::proton {

// The strategy for sampling the profiling data
enum class SamplingStrategy : uint32_t {
  NONE = 0,
  SELECTIVE = 1,
};

::std::optional<SamplingStrategy> symbolizeSamplingStrategy(uint32_t);
::llvm::StringRef stringifySamplingStrategy(SamplingStrategy);
::std::optional<SamplingStrategy> symbolizeSamplingStrategy(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSamplingStrategy() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(SamplingStrategy enumValue) {
  return stringifySamplingStrategy(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SamplingStrategy> symbolizeEnum<SamplingStrategy>(::llvm::StringRef str) {
  return symbolizeSamplingStrategy(str);
}

class SamplingStrategyAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = SamplingStrategy;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static SamplingStrategyAttr get(::mlir::MLIRContext *context, SamplingStrategy val);
  SamplingStrategy getValue() const;
};

} // namespace mlir::triton::proton

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::proton::SamplingStrategy, ::mlir::triton::proton::SamplingStrategy> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::proton::SamplingStrategy> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for The strategy for sampling the profiling data");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::proton::SamplingStrategy> attr = ::mlir::triton::proton::symbolizeEnum<::mlir::triton::proton::SamplingStrategy>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, selective] for The strategy for sampling the profiling data, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::proton::SamplingStrategy>, std::optional<::mlir::triton::proton::SamplingStrategy>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::proton::SamplingStrategy>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::proton::SamplingStrategy>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::proton::SamplingStrategy> attr = ::mlir::triton::proton::symbolizeEnum<::mlir::triton::proton::SamplingStrategy>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, selective] for The strategy for sampling the profiling data, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::proton::SamplingStrategy value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::proton::SamplingStrategy> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::proton::SamplingStrategy getEmptyKey() {
    return static_cast<::mlir::triton::proton::SamplingStrategy>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::proton::SamplingStrategy getTombstoneKey() {
    return static_cast<::mlir::triton::proton::SamplingStrategy>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::proton::SamplingStrategy &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::proton::SamplingStrategy &lhs, const ::mlir::triton::proton::SamplingStrategy &rhs) {
    return lhs == rhs;
  }
};
}

