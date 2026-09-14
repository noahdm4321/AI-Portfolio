/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OpenACCOps.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::acc {

// DefaultValue Clause
enum class ClauseDefaultValue : uint32_t {
  Present = 0,
  None = 1,
};

::std::optional<ClauseDefaultValue> symbolizeClauseDefaultValue(uint32_t);
::llvm::StringRef stringifyClauseDefaultValue(ClauseDefaultValue);
::std::optional<ClauseDefaultValue> symbolizeClauseDefaultValue(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForClauseDefaultValue() {
  return 1;
}


inline ::llvm::StringRef stringifyEnum(ClauseDefaultValue enumValue) {
  return stringifyClauseDefaultValue(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ClauseDefaultValue> symbolizeEnum<ClauseDefaultValue>(::llvm::StringRef str) {
  return symbolizeClauseDefaultValue(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::ClauseDefaultValue, ::mlir::acc::ClauseDefaultValue> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::ClauseDefaultValue> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for DefaultValue Clause");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::ClauseDefaultValue> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::ClauseDefaultValue>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [present, none] for DefaultValue Clause, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::ClauseDefaultValue>, std::optional<::mlir::acc::ClauseDefaultValue>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::ClauseDefaultValue>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::ClauseDefaultValue>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::ClauseDefaultValue> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::ClauseDefaultValue>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [present, none] for DefaultValue Clause, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::ClauseDefaultValue value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ClauseDefaultValue> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::ClauseDefaultValue getEmptyKey() {
    return static_cast<::mlir::acc::ClauseDefaultValue>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ClauseDefaultValue getTombstoneKey() {
    return static_cast<::mlir::acc::ClauseDefaultValue>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ClauseDefaultValue &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::ClauseDefaultValue &lhs, const ::mlir::acc::ClauseDefaultValue &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// Differentiate between combined constructs
enum class CombinedConstructsType : uint32_t {
  KernelsLoop = 1,
  ParallelLoop = 2,
  SerialLoop = 3,
};

::std::optional<CombinedConstructsType> symbolizeCombinedConstructsType(uint32_t);
::llvm::StringRef stringifyCombinedConstructsType(CombinedConstructsType);
::std::optional<CombinedConstructsType> symbolizeCombinedConstructsType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForCombinedConstructsType() {
  return 3;
}


inline ::llvm::StringRef stringifyEnum(CombinedConstructsType enumValue) {
  return stringifyCombinedConstructsType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<CombinedConstructsType> symbolizeEnum<CombinedConstructsType>(::llvm::StringRef str) {
  return symbolizeCombinedConstructsType(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::CombinedConstructsType, ::mlir::acc::CombinedConstructsType> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::CombinedConstructsType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Differentiate between combined constructs");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::CombinedConstructsType> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::CombinedConstructsType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [kernels_loop, parallel_loop, serial_loop] for Differentiate between combined constructs, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::CombinedConstructsType>, std::optional<::mlir::acc::CombinedConstructsType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::CombinedConstructsType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::CombinedConstructsType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::CombinedConstructsType> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::CombinedConstructsType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [kernels_loop, parallel_loop, serial_loop] for Differentiate between combined constructs, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::CombinedConstructsType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::CombinedConstructsType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::CombinedConstructsType getEmptyKey() {
    return static_cast<::mlir::acc::CombinedConstructsType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::CombinedConstructsType getTombstoneKey() {
    return static_cast<::mlir::acc::CombinedConstructsType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::CombinedConstructsType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::CombinedConstructsType &lhs, const ::mlir::acc::CombinedConstructsType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// constructs supported by OpenACC
enum class Construct : uint64_t {
  acc_construct_parallel = 0,
  acc_construct_kernels = 1,
  acc_construct_loop = 2,
  acc_construct_data = 3,
  acc_construct_enter_data = 4,
  acc_construct_exit_data = 5,
  acc_construct_host_data = 6,
  acc_construct_atomic = 7,
  acc_construct_declare = 8,
  acc_construct_init = 9,
  acc_construct_shutdown = 10,
  acc_construct_set = 11,
  acc_construct_update = 12,
  acc_construct_routine = 13,
  acc_construct_wait = 14,
  acc_construct_runtime_api = 15,
  acc_construct_serial = 16,
};

::std::optional<Construct> symbolizeConstruct(uint64_t);
::llvm::StringRef stringifyConstruct(Construct);
::std::optional<Construct> symbolizeConstruct(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForConstruct() {
  return 16;
}


inline ::llvm::StringRef stringifyEnum(Construct enumValue) {
  return stringifyConstruct(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<Construct> symbolizeEnum<Construct>(::llvm::StringRef str) {
  return symbolizeConstruct(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::Construct, ::mlir::acc::Construct> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::Construct> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for constructs supported by OpenACC");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::Construct> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::Construct>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [acc_construct_parallel, acc_construct_kernels, acc_construct_loop, acc_construct_data, acc_construct_enter_data, acc_construct_exit_data, acc_construct_host_data, acc_construct_atomic, acc_construct_declare, acc_construct_init, acc_construct_shutdown, acc_construct_set, acc_construct_update, acc_construct_routine, acc_construct_wait, acc_construct_runtime_api, acc_construct_serial] for constructs supported by OpenACC, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::Construct>, std::optional<::mlir::acc::Construct>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::Construct>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::Construct>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::Construct> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::Construct>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [acc_construct_parallel, acc_construct_kernels, acc_construct_loop, acc_construct_data, acc_construct_enter_data, acc_construct_exit_data, acc_construct_host_data, acc_construct_atomic, acc_construct_declare, acc_construct_init, acc_construct_shutdown, acc_construct_set, acc_construct_update, acc_construct_routine, acc_construct_wait, acc_construct_runtime_api, acc_construct_serial] for constructs supported by OpenACC, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::Construct value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::Construct> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::acc::Construct getEmptyKey() {
    return static_cast<::mlir::acc::Construct>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::Construct getTombstoneKey() {
    return static_cast<::mlir::acc::Construct>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::Construct &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::acc::Construct &lhs, const ::mlir::acc::Construct &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// data clauses supported by OpenACC
enum class DataClause : uint64_t {
  acc_copyin = 1,
  acc_copyin_readonly = 2,
  acc_copy = 3,
  acc_copyout = 4,
  acc_copyout_zero = 5,
  acc_present = 6,
  acc_create = 7,
  acc_create_zero = 8,
  acc_delete = 9,
  acc_attach = 10,
  acc_detach = 11,
  acc_no_create = 12,
  acc_private = 13,
  acc_firstprivate = 14,
  acc_deviceptr = 15,
  acc_getdeviceptr = 16,
  acc_update_host = 17,
  acc_update_self = 18,
  acc_update_device = 19,
  acc_use_device = 20,
  acc_reduction = 21,
  acc_declare_device_resident = 22,
  acc_declare_link = 23,
  acc_cache = 24,
  acc_cache_readonly = 25,
};

::std::optional<DataClause> symbolizeDataClause(uint64_t);
::llvm::StringRef stringifyDataClause(DataClause);
::std::optional<DataClause> symbolizeDataClause(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDataClause() {
  return 25;
}


inline ::llvm::StringRef stringifyEnum(DataClause enumValue) {
  return stringifyDataClause(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DataClause> symbolizeEnum<DataClause>(::llvm::StringRef str) {
  return symbolizeDataClause(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::DataClause, ::mlir::acc::DataClause> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::DataClause> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for data clauses supported by OpenACC");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::DataClause> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::DataClause>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [acc_copyin, acc_copyin_readonly, acc_copy, acc_copyout, acc_copyout_zero, acc_present, acc_create, acc_create_zero, acc_delete, acc_attach, acc_detach, acc_no_create, acc_private, acc_firstprivate, acc_deviceptr, acc_getdeviceptr, acc_update_host, acc_update_self, acc_update_device, acc_use_device, acc_reduction, acc_declare_device_resident, acc_declare_link, acc_cache, acc_cache_readonly] for data clauses supported by OpenACC, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::DataClause>, std::optional<::mlir::acc::DataClause>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::DataClause>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::DataClause>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::DataClause> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::DataClause>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [acc_copyin, acc_copyin_readonly, acc_copy, acc_copyout, acc_copyout_zero, acc_present, acc_create, acc_create_zero, acc_delete, acc_attach, acc_detach, acc_no_create, acc_private, acc_firstprivate, acc_deviceptr, acc_getdeviceptr, acc_update_host, acc_update_self, acc_update_device, acc_use_device, acc_reduction, acc_declare_device_resident, acc_declare_link, acc_cache, acc_cache_readonly] for data clauses supported by OpenACC, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::DataClause value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::DataClause> {
  using StorageInfo = ::llvm::DenseMapInfo<uint64_t>;

  static inline ::mlir::acc::DataClause getEmptyKey() {
    return static_cast<::mlir::acc::DataClause>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::DataClause getTombstoneKey() {
    return static_cast<::mlir::acc::DataClause>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::DataClause &val) {
    return StorageInfo::getHashValue(static_cast<uint64_t>(val));
  }

  static bool isEqual(const ::mlir::acc::DataClause &lhs, const ::mlir::acc::DataClause &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// Captures data clause modifiers
enum class DataClauseModifier : uint32_t {
  none = 0,
  zero = 1,
  readonly = 2,
  alwaysin = 4,
  alwaysout = 8,
  always = 12,
  capture = 16,
};

::std::optional<DataClauseModifier> symbolizeDataClauseModifier(uint32_t);
std::string stringifyDataClauseModifier(DataClauseModifier);
::std::optional<DataClauseModifier> symbolizeDataClauseModifier(::llvm::StringRef);

inline constexpr DataClauseModifier operator|(DataClauseModifier a, DataClauseModifier b) {
  return static_cast<DataClauseModifier>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr DataClauseModifier operator&(DataClauseModifier a, DataClauseModifier b) {
  return static_cast<DataClauseModifier>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr DataClauseModifier operator^(DataClauseModifier a, DataClauseModifier b) {
  return static_cast<DataClauseModifier>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr DataClauseModifier &operator|=(DataClauseModifier &a, DataClauseModifier b) {
    return a = a | b;
}
inline constexpr DataClauseModifier &operator&=(DataClauseModifier &a, DataClauseModifier b) {
    return a = a & b;
}
inline constexpr DataClauseModifier &operator^=(DataClauseModifier &a, DataClauseModifier b) {
    return a = a ^ b;
}
inline constexpr DataClauseModifier operator~(DataClauseModifier bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<DataClauseModifier>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(31u));
}
inline constexpr bool bitEnumContainsAll(DataClauseModifier bits, DataClauseModifier bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(DataClauseModifier bits, DataClauseModifier bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr DataClauseModifier bitEnumClear(DataClauseModifier bits, DataClauseModifier bit) {
  return bits & ~bit;
}
inline constexpr DataClauseModifier bitEnumSet(DataClauseModifier bits, DataClauseModifier bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(DataClauseModifier enumValue) {
  return stringifyDataClauseModifier(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DataClauseModifier> symbolizeEnum<DataClauseModifier>(::llvm::StringRef str) {
  return symbolizeDataClauseModifier(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::DataClauseModifier, ::mlir::acc::DataClauseModifier> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::DataClauseModifier> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Captures data clause modifiers");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::DataClauseModifier> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::DataClauseModifier>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, zero, readonly, alwaysin, alwaysout, always, capture] for Captures data clause modifiers, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::DataClauseModifier>, std::optional<::mlir::acc::DataClauseModifier>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::DataClauseModifier>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::DataClauseModifier>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::DataClauseModifier> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::DataClauseModifier>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, zero, readonly, alwaysin, alwaysout, always, capture] for Captures data clause modifiers, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::DataClauseModifier value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::acc::DataClauseModifier::always:
    return p << valueStr;
  default:
    break;
  }
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::acc::DataClauseModifier>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::DataClauseModifier> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::DataClauseModifier getEmptyKey() {
    return static_cast<::mlir::acc::DataClauseModifier>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::DataClauseModifier getTombstoneKey() {
    return static_cast<::mlir::acc::DataClauseModifier>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::DataClauseModifier &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::DataClauseModifier &lhs, const ::mlir::acc::DataClauseModifier &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// built-in device type supported by OpenACC
enum class DeviceType : uint32_t {
  None = 0,
  Star = 1,
  Default = 2,
  Host = 3,
  Multicore = 4,
  Nvidia = 5,
  Radeon = 6,
};

::std::optional<DeviceType> symbolizeDeviceType(uint32_t);
::llvm::StringRef stringifyDeviceType(DeviceType);
::std::optional<DeviceType> symbolizeDeviceType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDeviceType() {
  return 6;
}


inline ::llvm::StringRef stringifyEnum(DeviceType enumValue) {
  return stringifyDeviceType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<DeviceType> symbolizeEnum<DeviceType>(::llvm::StringRef str) {
  return symbolizeDeviceType(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::DeviceType, ::mlir::acc::DeviceType> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::DeviceType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for built-in device type supported by OpenACC");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::DeviceType> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::DeviceType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, star, default, host, multicore, nvidia, radeon] for built-in device type supported by OpenACC, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::DeviceType>, std::optional<::mlir::acc::DeviceType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::DeviceType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::DeviceType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::DeviceType> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::DeviceType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, star, default, host, multicore, nvidia, radeon] for built-in device type supported by OpenACC, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::DeviceType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::DeviceType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::DeviceType getEmptyKey() {
    return static_cast<::mlir::acc::DeviceType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::DeviceType getTombstoneKey() {
    return static_cast<::mlir::acc::DeviceType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::DeviceType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::DeviceType &lhs, const ::mlir::acc::DeviceType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// Differentiate the different gang arg values
enum class GangArgType : uint32_t {
  Num = 0,
  Dim = 1,
  Static = 2,
};

::std::optional<GangArgType> symbolizeGangArgType(uint32_t);
::llvm::StringRef stringifyGangArgType(GangArgType);
::std::optional<GangArgType> symbolizeGangArgType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForGangArgType() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(GangArgType enumValue) {
  return stringifyGangArgType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<GangArgType> symbolizeEnum<GangArgType>(::llvm::StringRef str) {
  return symbolizeGangArgType(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::GangArgType, ::mlir::acc::GangArgType> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::GangArgType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Differentiate the different gang arg values");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::GangArgType> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::GangArgType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [Num, Dim, Static] for Differentiate the different gang arg values, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::GangArgType>, std::optional<::mlir::acc::GangArgType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::GangArgType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::GangArgType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::GangArgType> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::GangArgType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [Num, Dim, Static] for Differentiate the different gang arg values, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::GangArgType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::GangArgType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::GangArgType getEmptyKey() {
    return static_cast<::mlir::acc::GangArgType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::GangArgType getTombstoneKey() {
    return static_cast<::mlir::acc::GangArgType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::GangArgType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::GangArgType &lhs, const ::mlir::acc::GangArgType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// Encodes the options for loop parallelism determination mode
enum class LoopParMode : uint32_t {
  loop_auto = 1,
  loop_independent = 2,
  loop_seq = 0,
};

::std::optional<LoopParMode> symbolizeLoopParMode(uint32_t);
::llvm::StringRef stringifyLoopParMode(LoopParMode);
::std::optional<LoopParMode> symbolizeLoopParMode(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLoopParMode() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(LoopParMode enumValue) {
  return stringifyLoopParMode(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<LoopParMode> symbolizeEnum<LoopParMode>(::llvm::StringRef str) {
  return symbolizeLoopParMode(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::LoopParMode, ::mlir::acc::LoopParMode> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::LoopParMode> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Encodes the options for loop parallelism determination mode");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::LoopParMode> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::LoopParMode>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [loop_auto, loop_independent, loop_seq] for Encodes the options for loop parallelism determination mode, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::LoopParMode>, std::optional<::mlir::acc::LoopParMode>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::LoopParMode>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::LoopParMode>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::LoopParMode> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::LoopParMode>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [loop_auto, loop_independent, loop_seq] for Encodes the options for loop parallelism determination mode, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::LoopParMode value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::LoopParMode> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::LoopParMode getEmptyKey() {
    return static_cast<::mlir::acc::LoopParMode>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::LoopParMode getTombstoneKey() {
    return static_cast<::mlir::acc::LoopParMode>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::LoopParMode &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::LoopParMode &lhs, const ::mlir::acc::LoopParMode &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// Parallelism level (gang/worker/vector/seq)
enum class ParLevel : uint32_t {
  seq = 0,
  gang_dim1 = 1,
  gang_dim2 = 2,
  gang_dim3 = 3,
  worker = 4,
  vector = 5,
};

::std::optional<ParLevel> symbolizeParLevel(uint32_t);
::llvm::StringRef stringifyParLevel(ParLevel);
::std::optional<ParLevel> symbolizeParLevel(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForParLevel() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(ParLevel enumValue) {
  return stringifyParLevel(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ParLevel> symbolizeEnum<ParLevel>(::llvm::StringRef str) {
  return symbolizeParLevel(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::ParLevel, ::mlir::acc::ParLevel> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::ParLevel> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Parallelism level (gang/worker/vector/seq)");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::ParLevel> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::ParLevel>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [seq, gang_dim1, gang_dim2, gang_dim3, worker, vector] for Parallelism level (gang/worker/vector/seq), got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::ParLevel>, std::optional<::mlir::acc::ParLevel>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::ParLevel>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::ParLevel>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::ParLevel> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::ParLevel>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [seq, gang_dim1, gang_dim2, gang_dim3, worker, vector] for Parallelism level (gang/worker/vector/seq), got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::ParLevel value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ParLevel> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::ParLevel getEmptyKey() {
    return static_cast<::mlir::acc::ParLevel>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ParLevel getTombstoneKey() {
    return static_cast<::mlir::acc::ParLevel>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ParLevel &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::ParLevel &lhs, const ::mlir::acc::ParLevel &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// Encodes the options for kinds of recipes availabie in acc dialect
enum class RecipeKind : uint32_t {
  private_recipe = 0,
  firstprivate_recipe = 1,
  reduction_recipe = 2,
};

::std::optional<RecipeKind> symbolizeRecipeKind(uint32_t);
::llvm::StringRef stringifyRecipeKind(RecipeKind);
::std::optional<RecipeKind> symbolizeRecipeKind(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForRecipeKind() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(RecipeKind enumValue) {
  return stringifyRecipeKind(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<RecipeKind> symbolizeEnum<RecipeKind>(::llvm::StringRef str) {
  return symbolizeRecipeKind(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::RecipeKind, ::mlir::acc::RecipeKind> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::RecipeKind> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Encodes the options for kinds of recipes availabie in acc dialect");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::RecipeKind> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::RecipeKind>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [private_recipe, firstprivate_recipe, reduction_recipe] for Encodes the options for kinds of recipes availabie in acc dialect, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::RecipeKind>, std::optional<::mlir::acc::RecipeKind>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::RecipeKind>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::RecipeKind>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::RecipeKind> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::RecipeKind>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [private_recipe, firstprivate_recipe, reduction_recipe] for Encodes the options for kinds of recipes availabie in acc dialect, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::RecipeKind value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::RecipeKind> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::RecipeKind getEmptyKey() {
    return static_cast<::mlir::acc::RecipeKind>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::RecipeKind getTombstoneKey() {
    return static_cast<::mlir::acc::RecipeKind>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::RecipeKind &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::RecipeKind &lhs, const ::mlir::acc::RecipeKind &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// built-in reduction operations supported by OpenACC
enum class ReductionOperator : uint32_t {
  AccNone = 0,
  AccAdd = 1,
  AccMul = 2,
  AccMax = 3,
  AccMin = 4,
  AccIand = 5,
  AccIor = 6,
  AccXor = 7,
  AccEqv = 8,
  AccNeqv = 9,
  AccLand = 10,
  AccLor = 11,
};

::std::optional<ReductionOperator> symbolizeReductionOperator(uint32_t);
::llvm::StringRef stringifyReductionOperator(ReductionOperator);
::std::optional<ReductionOperator> symbolizeReductionOperator(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForReductionOperator() {
  return 11;
}


inline ::llvm::StringRef stringifyEnum(ReductionOperator enumValue) {
  return stringifyReductionOperator(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ReductionOperator> symbolizeEnum<ReductionOperator>(::llvm::StringRef str) {
  return symbolizeReductionOperator(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::ReductionOperator, ::mlir::acc::ReductionOperator> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::ReductionOperator> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for built-in reduction operations supported by OpenACC");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::ReductionOperator> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::ReductionOperator>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, add, mul, max, min, iand, ior, xor, eqv, neqv, land, lor] for built-in reduction operations supported by OpenACC, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::ReductionOperator>, std::optional<::mlir::acc::ReductionOperator>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::ReductionOperator>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::ReductionOperator>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::ReductionOperator> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::ReductionOperator>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, add, mul, max, min, iand, ior, xor, eqv, neqv, land, lor] for built-in reduction operations supported by OpenACC, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::ReductionOperator value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::ReductionOperator> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::ReductionOperator getEmptyKey() {
    return static_cast<::mlir::acc::ReductionOperator>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::ReductionOperator getTombstoneKey() {
    return static_cast<::mlir::acc::ReductionOperator>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::ReductionOperator &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::ReductionOperator &lhs, const ::mlir::acc::ReductionOperator &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::acc {

// Captures different type categories described in OpenACC spec
enum class VariableTypeCategory : uint32_t {
  uncategorized = 0,
  scalar = 1,
  array = 2,
  composite = 4,
  nonscalar = 8,
  aggregate = 14,
};

::std::optional<VariableTypeCategory> symbolizeVariableTypeCategory(uint32_t);
std::string stringifyVariableTypeCategory(VariableTypeCategory);
::std::optional<VariableTypeCategory> symbolizeVariableTypeCategory(::llvm::StringRef);

inline constexpr VariableTypeCategory operator|(VariableTypeCategory a, VariableTypeCategory b) {
  return static_cast<VariableTypeCategory>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr VariableTypeCategory operator&(VariableTypeCategory a, VariableTypeCategory b) {
  return static_cast<VariableTypeCategory>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr VariableTypeCategory operator^(VariableTypeCategory a, VariableTypeCategory b) {
  return static_cast<VariableTypeCategory>(static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b));
}
inline constexpr VariableTypeCategory &operator|=(VariableTypeCategory &a, VariableTypeCategory b) {
    return a = a | b;
}
inline constexpr VariableTypeCategory &operator&=(VariableTypeCategory &a, VariableTypeCategory b) {
    return a = a & b;
}
inline constexpr VariableTypeCategory &operator^=(VariableTypeCategory &a, VariableTypeCategory b) {
    return a = a ^ b;
}
inline constexpr VariableTypeCategory operator~(VariableTypeCategory bits) {
  // Ensure only bits that can be present in the enum are set
  return static_cast<VariableTypeCategory>(~static_cast<uint32_t>(bits) & static_cast<uint32_t>(15u));
}
inline constexpr bool bitEnumContainsAll(VariableTypeCategory bits, VariableTypeCategory bit) {
  return (bits & bit) == bit;
}
inline constexpr bool bitEnumContainsAny(VariableTypeCategory bits, VariableTypeCategory bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}
inline constexpr VariableTypeCategory bitEnumClear(VariableTypeCategory bits, VariableTypeCategory bit) {
  return bits & ~bit;
}
inline constexpr VariableTypeCategory bitEnumSet(VariableTypeCategory bits, VariableTypeCategory bit, /*optional*/bool value=true) {
  return value ? (bits | bit) : bitEnumClear(bits, bit);
}
  
inline std::string stringifyEnum(VariableTypeCategory enumValue) {
  return stringifyVariableTypeCategory(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<VariableTypeCategory> symbolizeEnum<VariableTypeCategory>(::llvm::StringRef str) {
  return symbolizeVariableTypeCategory(str);
}

} // namespace mlir::acc

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::acc::VariableTypeCategory, ::mlir::acc::VariableTypeCategory> {
  template <typename ParserT>
  static FailureOr<::mlir::acc::VariableTypeCategory> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Captures different type categories described in OpenACC spec");

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::VariableTypeCategory> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::VariableTypeCategory>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [uncategorized, scalar, array, composite, nonscalar, aggregate] for Captures different type categories described in OpenACC spec, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::acc::VariableTypeCategory>, std::optional<::mlir::acc::VariableTypeCategory>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::acc::VariableTypeCategory>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::acc::VariableTypeCategory>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::acc::VariableTypeCategory> attr = ::mlir::acc::symbolizeEnum<::mlir::acc::VariableTypeCategory>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [uncategorized, scalar, array, composite, nonscalar, aggregate] for Captures different type categories described in OpenACC spec, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::acc::VariableTypeCategory value) {
  auto valueStr = stringifyEnum(value);
  switch (value) {
  case ::mlir::acc::VariableTypeCategory::aggregate:
    return p << valueStr;
  default:
    break;
  }
  auto underlyingValue = static_cast<std::make_unsigned_t<::mlir::acc::VariableTypeCategory>>(value);
  if (underlyingValue && !llvm::has_single_bit(underlyingValue))
    return p << '"' << valueStr << '"';
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::acc::VariableTypeCategory> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::acc::VariableTypeCategory getEmptyKey() {
    return static_cast<::mlir::acc::VariableTypeCategory>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::acc::VariableTypeCategory getTombstoneKey() {
    return static_cast<::mlir::acc::VariableTypeCategory>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::acc::VariableTypeCategory &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::acc::VariableTypeCategory &lhs, const ::mlir::acc::VariableTypeCategory &rhs) {
    return lhs == rhs;
  }
};
}

