/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: ProtonAttrDefs.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::proton {

::llvm::StringRef stringifyMetricType(MetricType val) {
  switch (val) {
    case MetricType::CYCLE: return "cycle";
  }
  return "";
}

::std::optional<MetricType> symbolizeMetricType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MetricType>>(str)
      .Case("cycle", MetricType::CYCLE)
      .Default(::std::nullopt);
}
::std::optional<MetricType> symbolizeMetricType(uint32_t value) {
  switch (value) {
  case 0: return MetricType::CYCLE;
  default: return ::std::nullopt;
  }
}

bool MetricTypeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0))));
}
MetricTypeAttr MetricTypeAttr::get(::mlir::MLIRContext *context, MetricType val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<MetricTypeAttr>(baseAttr);
}
MetricType MetricTypeAttr::getValue() const {
  return static_cast<MetricType>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::triton::proton
namespace mlir::triton::proton {

::llvm::StringRef stringifyMode(Mode val) {
  switch (val) {
    case Mode::DEFAULT: return "default";
    case Mode::MMA: return "mma";
  }
  return "";
}

::std::optional<Mode> symbolizeMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Mode>>(str)
      .Case("default", Mode::DEFAULT)
      .Case("mma", Mode::MMA)
      .Default(::std::nullopt);
}
::std::optional<Mode> symbolizeMode(uint32_t value) {
  switch (value) {
  case 0: return Mode::DEFAULT;
  case 1: return Mode::MMA;
  default: return ::std::nullopt;
  }
}

bool ModeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
ModeAttr ModeAttr::get(::mlir::MLIRContext *context, Mode val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<ModeAttr>(baseAttr);
}
Mode ModeAttr::getValue() const {
  return static_cast<Mode>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::triton::proton
namespace mlir::triton::proton {

::llvm::StringRef stringifySamplingStrategy(SamplingStrategy val) {
  switch (val) {
    case SamplingStrategy::NONE: return "none";
    case SamplingStrategy::SELECTIVE: return "selective";
  }
  return "";
}

::std::optional<SamplingStrategy> symbolizeSamplingStrategy(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<SamplingStrategy>>(str)
      .Case("none", SamplingStrategy::NONE)
      .Case("selective", SamplingStrategy::SELECTIVE)
      .Default(::std::nullopt);
}
::std::optional<SamplingStrategy> symbolizeSamplingStrategy(uint32_t value) {
  switch (value) {
  case 0: return SamplingStrategy::NONE;
  case 1: return SamplingStrategy::SELECTIVE;
  default: return ::std::nullopt;
  }
}

bool SamplingStrategyAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
SamplingStrategyAttr SamplingStrategyAttr::get(::mlir::MLIRContext *context, SamplingStrategy val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<SamplingStrategyAttr>(baseAttr);
}
SamplingStrategy SamplingStrategyAttr::getValue() const {
  return static_cast<SamplingStrategy>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::triton::proton
