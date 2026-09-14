/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonAMDGPUAttrDefs.td                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::amdgpu {

::llvm::StringRef stringifySchedHint(SchedHint val) {
  switch (val) {
    case SchedHint::none: return "none";
    case SchedHint::attention: return "attention";
  }
  return "";
}

::std::optional<SchedHint> symbolizeSchedHint(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<SchedHint>>(str)
      .Case("none", SchedHint::none)
      .Case("attention", SchedHint::attention)
      .Default(::std::nullopt);
}
::std::optional<SchedHint> symbolizeSchedHint(uint32_t value) {
  switch (value) {
  case 0: return SchedHint::none;
  case 2: return SchedHint::attention;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::triton::amdgpu
