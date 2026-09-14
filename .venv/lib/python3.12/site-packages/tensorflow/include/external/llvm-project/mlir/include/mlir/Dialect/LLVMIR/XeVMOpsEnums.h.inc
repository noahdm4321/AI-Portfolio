/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: XeVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::xevm {

// Address spaces
enum class AddrSpace : uint32_t {
  PRIVATE = 0,
  GLOBAL = 1,
  CONSTANT = 2,
  SHARED = 3,
  GENERIC = 4,
};

::std::optional<AddrSpace> symbolizeAddrSpace(uint32_t);
::llvm::StringRef stringifyAddrSpace(AddrSpace);
::std::optional<AddrSpace> symbolizeAddrSpace(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForAddrSpace() {
  return 4;
}


inline ::llvm::StringRef stringifyEnum(AddrSpace enumValue) {
  return stringifyAddrSpace(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<AddrSpace> symbolizeEnum<AddrSpace>(::llvm::StringRef str) {
  return symbolizeAddrSpace(str);
}

} // namespace mlir::xevm

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<mlir::xevm::AddrSpace, mlir::xevm::AddrSpace> {
  template <typename ParserT>
  static FailureOr<mlir::xevm::AddrSpace> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Address spaces");

    // Symbolize the keyword.
    if (::std::optional<mlir::xevm::AddrSpace> attr = mlir::xevm::symbolizeEnum<mlir::xevm::AddrSpace>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [private, global, constant, shared, generic] for Address spaces, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<mlir::xevm::AddrSpace>, std::optional<mlir::xevm::AddrSpace>> {
  template <typename ParserT>
  static FailureOr<std::optional<mlir::xevm::AddrSpace>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<mlir::xevm::AddrSpace>{};

    // Symbolize the keyword.
    if (::std::optional<mlir::xevm::AddrSpace> attr = mlir::xevm::symbolizeEnum<mlir::xevm::AddrSpace>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [private, global, constant, shared, generic] for Address spaces, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, mlir::xevm::AddrSpace value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<mlir::xevm::AddrSpace> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline mlir::xevm::AddrSpace getEmptyKey() {
    return static_cast<mlir::xevm::AddrSpace>(StorageInfo::getEmptyKey());
  }

  static inline mlir::xevm::AddrSpace getTombstoneKey() {
    return static_cast<mlir::xevm::AddrSpace>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const mlir::xevm::AddrSpace &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const mlir::xevm::AddrSpace &lhs, const mlir::xevm::AddrSpace &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::xevm {

// XeVM element type
enum class ElemType : uint32_t {
  BF16 = 8,
  F16 = 9,
  S8 = 10,
  U8 = 11,
  S4 = 12,
  U4 = 13,
  TF32 = 14,
  F32 = 15,
  S32 = 16,
};

::std::optional<ElemType> symbolizeElemType(uint32_t);
::llvm::StringRef stringifyElemType(ElemType);
::std::optional<ElemType> symbolizeElemType(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForElemType() {
  return 16;
}


inline ::llvm::StringRef stringifyEnum(ElemType enumValue) {
  return stringifyElemType(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<ElemType> symbolizeEnum<ElemType>(::llvm::StringRef str) {
  return symbolizeElemType(str);
}

class ElemTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = ElemType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static ElemTypeAttr get(::mlir::MLIRContext *context, ElemType val);
  ElemType getValue() const;
};

} // namespace mlir::xevm

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::xevm::ElemType, ::mlir::xevm::ElemType> {
  template <typename ParserT>
  static FailureOr<::mlir::xevm::ElemType> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for XeVM element type");

    // Symbolize the keyword.
    if (::std::optional<::mlir::xevm::ElemType> attr = ::mlir::xevm::symbolizeEnum<::mlir::xevm::ElemType>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [bf16, f16, s8, u8, s4, u4, tf32, f32, s32] for XeVM element type, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::xevm::ElemType>, std::optional<::mlir::xevm::ElemType>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::xevm::ElemType>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::xevm::ElemType>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::xevm::ElemType> attr = ::mlir::xevm::symbolizeEnum<::mlir::xevm::ElemType>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [bf16, f16, s8, u8, s4, u4, tf32, f32, s32] for XeVM element type, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::xevm::ElemType value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::xevm::ElemType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::xevm::ElemType getEmptyKey() {
    return static_cast<::mlir::xevm::ElemType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::xevm::ElemType getTombstoneKey() {
    return static_cast<::mlir::xevm::ElemType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::xevm::ElemType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::xevm::ElemType &lhs, const ::mlir::xevm::ElemType &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::xevm {

// XeVM load ops cache control
enum class LoadCacheControl : uint32_t {
  L1UC_L2UC_L3UC = 1,
  L1UC_L2UC_L3C = 2,
  L1UC_L2C_L3UC = 3,
  L1UC_L2C_L3C = 4,
  L1C_L2UC_L3UC = 5,
  L1C_L2UC_L3C = 6,
  L1C_L2C_L3UC = 7,
  L1C_L2C_L3C = 8,
  L1S_L2UC_L3UC = 9,
  L1S_L2UC_L3C = 10,
  L1S_L2C_L3UC = 11,
  L1S_L2C_L3C = 12,
  INVALIDATE_READ = 13,
};

::std::optional<LoadCacheControl> symbolizeLoadCacheControl(uint32_t);
::llvm::StringRef stringifyLoadCacheControl(LoadCacheControl);
::std::optional<LoadCacheControl> symbolizeLoadCacheControl(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForLoadCacheControl() {
  return 13;
}


inline ::llvm::StringRef stringifyEnum(LoadCacheControl enumValue) {
  return stringifyLoadCacheControl(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<LoadCacheControl> symbolizeEnum<LoadCacheControl>(::llvm::StringRef str) {
  return symbolizeLoadCacheControl(str);
}

} // namespace mlir::xevm

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::xevm::LoadCacheControl, ::mlir::xevm::LoadCacheControl> {
  template <typename ParserT>
  static FailureOr<::mlir::xevm::LoadCacheControl> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for XeVM load ops cache control");

    // Symbolize the keyword.
    if (::std::optional<::mlir::xevm::LoadCacheControl> attr = ::mlir::xevm::symbolizeEnum<::mlir::xevm::LoadCacheControl>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [L1uc_L2uc_L3uc, L1uc_L2uc_L3c, L1uc_L2c_L3uc, L1uc_L2c_L3c, L1c_L2uc_L3uc, L1c_L2uc_L3c, L1c_L2c_L3uc, L1c_L2c_L3c, L1s_L2uc_L3uc, L1s_L2uc_L3c, L1s_L2c_L3uc, L1s_L2c_L3c, ir] for XeVM load ops cache control, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::xevm::LoadCacheControl>, std::optional<::mlir::xevm::LoadCacheControl>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::xevm::LoadCacheControl>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::xevm::LoadCacheControl>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::xevm::LoadCacheControl> attr = ::mlir::xevm::symbolizeEnum<::mlir::xevm::LoadCacheControl>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [L1uc_L2uc_L3uc, L1uc_L2uc_L3c, L1uc_L2c_L3uc, L1uc_L2c_L3c, L1c_L2uc_L3uc, L1c_L2uc_L3c, L1c_L2c_L3uc, L1c_L2c_L3c, L1s_L2uc_L3uc, L1s_L2uc_L3c, L1s_L2c_L3uc, L1s_L2c_L3c, ir] for XeVM load ops cache control, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::xevm::LoadCacheControl value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::xevm::LoadCacheControl> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::xevm::LoadCacheControl getEmptyKey() {
    return static_cast<::mlir::xevm::LoadCacheControl>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::xevm::LoadCacheControl getTombstoneKey() {
    return static_cast<::mlir::xevm::LoadCacheControl>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::xevm::LoadCacheControl &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::xevm::LoadCacheControl &lhs, const ::mlir::xevm::LoadCacheControl &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::xevm {

// XeVM memory scope
enum class MemScope : uint32_t {
  LANE = 0,
  SUBGROUP = 1,
  WORKGROUP = 2,
  CLUSTER = 3,
  DEVICE = 4,
  SYSTEM = 5,
};

::std::optional<MemScope> symbolizeMemScope(uint32_t);
::llvm::StringRef stringifyMemScope(MemScope);
::std::optional<MemScope> symbolizeMemScope(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMemScope() {
  return 5;
}


inline ::llvm::StringRef stringifyEnum(MemScope enumValue) {
  return stringifyMemScope(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<MemScope> symbolizeEnum<MemScope>(::llvm::StringRef str) {
  return symbolizeMemScope(str);
}

} // namespace mlir::xevm

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::xevm::MemScope, ::mlir::xevm::MemScope> {
  template <typename ParserT>
  static FailureOr<::mlir::xevm::MemScope> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for XeVM memory scope");

    // Symbolize the keyword.
    if (::std::optional<::mlir::xevm::MemScope> attr = ::mlir::xevm::symbolizeEnum<::mlir::xevm::MemScope>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [lane, subgroup, workgroup, cluster, device, system] for XeVM memory scope, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::xevm::MemScope>, std::optional<::mlir::xevm::MemScope>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::xevm::MemScope>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::xevm::MemScope>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::xevm::MemScope> attr = ::mlir::xevm::symbolizeEnum<::mlir::xevm::MemScope>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [lane, subgroup, workgroup, cluster, device, system] for XeVM memory scope, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::xevm::MemScope value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::xevm::MemScope> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::xevm::MemScope getEmptyKey() {
    return static_cast<::mlir::xevm::MemScope>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::xevm::MemScope getTombstoneKey() {
    return static_cast<::mlir::xevm::MemScope>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::xevm::MemScope &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::xevm::MemScope &lhs, const ::mlir::xevm::MemScope &rhs) {
    return lhs == rhs;
  }
};
}

namespace mlir::xevm {

// XeVM store ops cache control
enum class StoreCacheControl : uint32_t {
  L1UC_L2UC_L3UC = 1,
  L1UC_L2UC_L3WB = 2,
  L1UC_L2WB_L3UC = 3,
  L1UC_L2WB_L3WB = 4,
  L1WT_L2UC_L3UC = 5,
  L1WT_L2UC_L3WB = 6,
  L1WT_L2WB_L3UC = 7,
  L1WT_L2WB_L3WB = 8,
  L1S_L2UC_L3UC = 9,
  L1S_L2UC_L3WB = 10,
  L1S_L2WB_L3UC = 11,
  L1S_L2WB_L3WB = 12,
  L1WB_L2UC_L3UC = 13,
  L1WB_L2WB_L3UC = 14,
  L1WB_L2UC_L3WB = 15,
};

::std::optional<StoreCacheControl> symbolizeStoreCacheControl(uint32_t);
::llvm::StringRef stringifyStoreCacheControl(StoreCacheControl);
::std::optional<StoreCacheControl> symbolizeStoreCacheControl(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForStoreCacheControl() {
  return 15;
}


inline ::llvm::StringRef stringifyEnum(StoreCacheControl enumValue) {
  return stringifyStoreCacheControl(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<StoreCacheControl> symbolizeEnum<StoreCacheControl>(::llvm::StringRef str) {
  return symbolizeStoreCacheControl(str);
}

} // namespace mlir::xevm

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::xevm::StoreCacheControl, ::mlir::xevm::StoreCacheControl> {
  template <typename ParserT>
  static FailureOr<::mlir::xevm::StoreCacheControl> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for XeVM store ops cache control");

    // Symbolize the keyword.
    if (::std::optional<::mlir::xevm::StoreCacheControl> attr = ::mlir::xevm::symbolizeEnum<::mlir::xevm::StoreCacheControl>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [L1uc_L2uc_L3uc, L1uc_L2uc_L3wb, L1uc_L2wb_L3uc, L1uc_L2wb_L3wb, L1wt_L2uc_L3uc, L1wt_L2uc_L3wb, L1wt_L2wb_L3uc, L1wt_L2wb_L3wb, L1s_L2uc_L3uc, L1s_L2uc_L3wb, L1s_L2wb_L3uc, L1s_L2wb_L3wb, L1wb_L2uc_L3uc, L1wb_L2wb_L3uc, L1wb_L2uc_L3wb] for XeVM store ops cache control, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::xevm::StoreCacheControl>, std::optional<::mlir::xevm::StoreCacheControl>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::xevm::StoreCacheControl>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::xevm::StoreCacheControl>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::xevm::StoreCacheControl> attr = ::mlir::xevm::symbolizeEnum<::mlir::xevm::StoreCacheControl>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [L1uc_L2uc_L3uc, L1uc_L2uc_L3wb, L1uc_L2wb_L3uc, L1uc_L2wb_L3wb, L1wt_L2uc_L3uc, L1wt_L2uc_L3wb, L1wt_L2wb_L3uc, L1wt_L2wb_L3wb, L1s_L2uc_L3uc, L1s_L2uc_L3wb, L1s_L2wb_L3uc, L1s_L2wb_L3wb, L1wb_L2uc_L3uc, L1wb_L2wb_L3uc, L1wb_L2uc_L3wb] for XeVM store ops cache control, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::xevm::StoreCacheControl value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::xevm::StoreCacheControl> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::xevm::StoreCacheControl getEmptyKey() {
    return static_cast<::mlir::xevm::StoreCacheControl>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::xevm::StoreCacheControl getTombstoneKey() {
    return static_cast<::mlir::xevm::StoreCacheControl>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::xevm::StoreCacheControl &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::xevm::StoreCacheControl &lhs, const ::mlir::xevm::StoreCacheControl &rhs) {
    return lhs == rhs;
  }
};
}

