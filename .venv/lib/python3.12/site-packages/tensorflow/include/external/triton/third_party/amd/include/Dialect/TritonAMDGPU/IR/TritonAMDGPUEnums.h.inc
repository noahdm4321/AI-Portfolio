/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonAMDGPUAttrDefs.td                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::amdgpu {

// Instruction Scheduling Hints for AMD GPUs
enum class SchedHint : uint32_t {
  none = 0,
  attention = 2,
};

::std::optional<SchedHint> symbolizeSchedHint(uint32_t);
::llvm::StringRef stringifySchedHint(SchedHint);
::std::optional<SchedHint> symbolizeSchedHint(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForSchedHint() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(SchedHint enumValue) {
  return stringifySchedHint(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<SchedHint> symbolizeEnum<SchedHint>(::llvm::StringRef str) {
  return symbolizeSchedHint(str);
}

} // namespace mlir::triton::amdgpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::amdgpu::SchedHint, ::mlir::triton::amdgpu::SchedHint> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::amdgpu::SchedHint> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for Instruction Scheduling Hints for AMD GPUs");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::amdgpu::SchedHint> attr = ::mlir::triton::amdgpu::symbolizeEnum<::mlir::triton::amdgpu::SchedHint>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [none, attention] for Instruction Scheduling Hints for AMD GPUs, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::amdgpu::SchedHint>, std::optional<::mlir::triton::amdgpu::SchedHint>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::amdgpu::SchedHint>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::amdgpu::SchedHint>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::amdgpu::SchedHint> attr = ::mlir::triton::amdgpu::symbolizeEnum<::mlir::triton::amdgpu::SchedHint>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [none, attention] for Instruction Scheduling Hints for AMD GPUs, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::amdgpu::SchedHint value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::amdgpu::SchedHint> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::amdgpu::SchedHint getEmptyKey() {
    return static_cast<::mlir::triton::amdgpu::SchedHint>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::amdgpu::SchedHint getTombstoneKey() {
    return static_cast<::mlir::triton::amdgpu::SchedHint>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::amdgpu::SchedHint &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::amdgpu::SchedHint &lhs, const ::mlir::triton::amdgpu::SchedHint &rhs) {
    return lhs == rhs;
  }
};
}

