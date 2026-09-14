/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TosaOpBase.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::tosa {

::llvm::StringRef stringifyProfile(Profile val) {
  switch (val) {
    case Profile::pro_int: return "pro_int";
    case Profile::pro_fp: return "pro_fp";
    case Profile::none: return "none";
  }
  return "";
}

::std::optional<Profile> symbolizeProfile(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Profile>>(str)
      .Case("pro_int", Profile::pro_int)
      .Case("pro_fp", Profile::pro_fp)
      .Case("none", Profile::none)
      .Default(::std::nullopt);
}
::std::optional<Profile> symbolizeProfile(uint32_t value) {
  switch (value) {
  case 1: return Profile::pro_int;
  case 2: return Profile::pro_fp;
  case 0: return Profile::none;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::tosa
namespace mlir::tosa {

::llvm::StringRef stringifyExtension(Extension val) {
  switch (val) {
    case Extension::none: return "none";
    case Extension::int16: return "int16";
    case Extension::int4: return "int4";
    case Extension::bf16: return "bf16";
    case Extension::fp8e4m3: return "fp8e4m3";
    case Extension::fp8e5m2: return "fp8e5m2";
    case Extension::fft: return "fft";
    case Extension::variable: return "variable";
    case Extension::controlflow: return "controlflow";
    case Extension::doubleround: return "doubleround";
    case Extension::inexactround: return "inexactround";
    case Extension::dynamic: return "dynamic";
    case Extension::mxfp: return "mxfp";
    case Extension::int64: return "int64";
    case Extension::mxfp_conv: return "mxfp_conv";
    case Extension::shape: return "shape";
  }
  return "";
}

::std::optional<Extension> symbolizeExtension(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Extension>>(str)
      .Case("none", Extension::none)
      .Case("int16", Extension::int16)
      .Case("int4", Extension::int4)
      .Case("bf16", Extension::bf16)
      .Case("fp8e4m3", Extension::fp8e4m3)
      .Case("fp8e5m2", Extension::fp8e5m2)
      .Case("fft", Extension::fft)
      .Case("variable", Extension::variable)
      .Case("controlflow", Extension::controlflow)
      .Case("doubleround", Extension::doubleround)
      .Case("inexactround", Extension::inexactround)
      .Case("dynamic", Extension::dynamic)
      .Case("mxfp", Extension::mxfp)
      .Case("int64", Extension::int64)
      .Case("mxfp_conv", Extension::mxfp_conv)
      .Case("shape", Extension::shape)
      .Default(::std::nullopt);
}
::std::optional<Extension> symbolizeExtension(uint32_t value) {
  switch (value) {
  case 0: return Extension::none;
  case 1: return Extension::int16;
  case 2: return Extension::int4;
  case 3: return Extension::bf16;
  case 4: return Extension::fp8e4m3;
  case 5: return Extension::fp8e5m2;
  case 6: return Extension::fft;
  case 7: return Extension::variable;
  case 8: return Extension::controlflow;
  case 9: return Extension::doubleround;
  case 10: return Extension::inexactround;
  case 11: return Extension::dynamic;
  case 12: return Extension::mxfp;
  case 13: return Extension::int64;
  case 14: return Extension::mxfp_conv;
  case 15: return Extension::shape;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::tosa
namespace mlir::tosa {

::llvm::StringRef stringifyLevel(Level val) {
  switch (val) {
    case Level::none: return "none";
    case Level::eightK: return "8k";
  }
  return "";
}

::std::optional<Level> symbolizeLevel(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Level>>(str)
      .Case("none", Level::none)
      .Case("8k", Level::eightK)
      .Default(::std::nullopt);
}
::std::optional<Level> symbolizeLevel(uint32_t value) {
  switch (value) {
  case 0: return Level::none;
  case 1: return Level::eightK;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::tosa
namespace mlir::tosa {

::llvm::StringRef stringifySpecificationVersion(SpecificationVersion val) {
  switch (val) {
    case SpecificationVersion::V_1_0: return "1.0";
    case SpecificationVersion::V_1_1_DRAFT: return "1.1.draft";
  }
  return "";
}

::std::optional<SpecificationVersion> symbolizeSpecificationVersion(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<SpecificationVersion>>(str)
      .Case("1.0", SpecificationVersion::V_1_0)
      .Case("1.1.draft", SpecificationVersion::V_1_1_DRAFT)
      .Default(::std::nullopt);
}
::std::optional<SpecificationVersion> symbolizeSpecificationVersion(uint32_t value) {
  switch (value) {
  case 0: return SpecificationVersion::V_1_0;
  case 1: return SpecificationVersion::V_1_1_DRAFT;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::tosa
namespace mlir::tosa {

::llvm::StringRef stringifyResizeMode(ResizeMode val) {
  switch (val) {
    case ResizeMode::NEAREST_NEIGHBOR: return "NEAREST_NEIGHBOR";
    case ResizeMode::BILINEAR: return "BILINEAR";
  }
  return "";
}

::std::optional<ResizeMode> symbolizeResizeMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ResizeMode>>(str)
      .Case("NEAREST_NEIGHBOR", ResizeMode::NEAREST_NEIGHBOR)
      .Case("BILINEAR", ResizeMode::BILINEAR)
      .Default(::std::nullopt);
}
::std::optional<ResizeMode> symbolizeResizeMode(uint32_t value) {
  switch (value) {
  case 1: return ResizeMode::NEAREST_NEIGHBOR;
  case 2: return ResizeMode::BILINEAR;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::tosa
namespace mlir::tosa {

::llvm::StringRef stringifyNanPropagationMode(NanPropagationMode val) {
  switch (val) {
    case NanPropagationMode::PROPAGATE: return "PROPAGATE";
    case NanPropagationMode::IGNORE: return "IGNORE";
  }
  return "";
}

::std::optional<NanPropagationMode> symbolizeNanPropagationMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<NanPropagationMode>>(str)
      .Case("PROPAGATE", NanPropagationMode::PROPAGATE)
      .Case("IGNORE", NanPropagationMode::IGNORE)
      .Default(::std::nullopt);
}
::std::optional<NanPropagationMode> symbolizeNanPropagationMode(uint32_t value) {
  switch (value) {
  case 1: return NanPropagationMode::PROPAGATE;
  case 2: return NanPropagationMode::IGNORE;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::tosa
namespace mlir::tosa {

::llvm::StringRef stringifyRoundingMode(RoundingMode val) {
  switch (val) {
    case RoundingMode::SINGLE_ROUND: return "SINGLE_ROUND";
    case RoundingMode::INEXACT_ROUND: return "INEXACT_ROUND";
    case RoundingMode::DOUBLE_ROUND: return "DOUBLE_ROUND";
  }
  return "";
}

::std::optional<RoundingMode> symbolizeRoundingMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RoundingMode>>(str)
      .Case("SINGLE_ROUND", RoundingMode::SINGLE_ROUND)
      .Case("INEXACT_ROUND", RoundingMode::INEXACT_ROUND)
      .Case("DOUBLE_ROUND", RoundingMode::DOUBLE_ROUND)
      .Default(::std::nullopt);
}
::std::optional<RoundingMode> symbolizeRoundingMode(uint32_t value) {
  switch (value) {
  case 1: return RoundingMode::SINGLE_ROUND;
  case 2: return RoundingMode::INEXACT_ROUND;
  case 3: return RoundingMode::DOUBLE_ROUND;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::tosa
namespace mlir::tosa {

::llvm::StringRef stringifyBlockSize(BlockSize val) {
  switch (val) {
    case BlockSize::BLOCK_SIZE_32: return "BLOCK_SIZE_32";
  }
  return "";
}

::std::optional<BlockSize> symbolizeBlockSize(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<BlockSize>>(str)
      .Case("BLOCK_SIZE_32", BlockSize::BLOCK_SIZE_32)
      .Default(::std::nullopt);
}
::std::optional<BlockSize> symbolizeBlockSize(uint32_t value) {
  switch (value) {
  case 32: return BlockSize::BLOCK_SIZE_32;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::tosa
