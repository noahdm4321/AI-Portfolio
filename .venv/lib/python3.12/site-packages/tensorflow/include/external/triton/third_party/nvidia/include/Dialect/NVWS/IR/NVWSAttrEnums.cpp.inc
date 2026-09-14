/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: NVWSAttrDefs.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvws {

::llvm::StringRef stringifyAsyncOp(AsyncOp val) {
  switch (val) {
    case AsyncOp::NONE: return "none";
    case AsyncOp::TMALoad: return "tma_load";
    case AsyncOp::TC5MMA: return "tc5mma";
    case AsyncOp::TMEMCopy: return "tmem_copy";
    case AsyncOp::CpAsync: return "cp_async";
    case AsyncOp::WGMMA: return "wgmma";
  }
  return "";
}

::std::optional<AsyncOp> symbolizeAsyncOp(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AsyncOp>>(str)
      .Case("none", AsyncOp::NONE)
      .Case("tma_load", AsyncOp::TMALoad)
      .Case("tc5mma", AsyncOp::TC5MMA)
      .Case("tmem_copy", AsyncOp::TMEMCopy)
      .Case("cp_async", AsyncOp::CpAsync)
      .Case("wgmma", AsyncOp::WGMMA)
      .Default(::std::nullopt);
}
::std::optional<AsyncOp> symbolizeAsyncOp(uint32_t value) {
  switch (value) {
  case 0: return AsyncOp::NONE;
  case 1: return AsyncOp::TMALoad;
  case 2: return AsyncOp::TC5MMA;
  case 3: return AsyncOp::TMEMCopy;
  case 4: return AsyncOp::CpAsync;
  case 5: return AsyncOp::WGMMA;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

::llvm::StringRef stringifyTokenLoadType(TokenLoadType val) {
  switch (val) {
    case TokenLoadType::None: return "none";
    case TokenLoadType::AsyncLoadOp: return "asyncLoadOp";
    case TokenLoadType::TMALoadOp: return "tmaLoadOp";
    case TokenLoadType::LocalStoreOp: return "localStoreOp";
    case TokenLoadType::TmemLoadOp: return "TmemLoadOp";
  }
  return "";
}

::std::optional<TokenLoadType> symbolizeTokenLoadType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<TokenLoadType>>(str)
      .Case("none", TokenLoadType::None)
      .Case("asyncLoadOp", TokenLoadType::AsyncLoadOp)
      .Case("tmaLoadOp", TokenLoadType::TMALoadOp)
      .Case("localStoreOp", TokenLoadType::LocalStoreOp)
      .Case("TmemLoadOp", TokenLoadType::TmemLoadOp)
      .Default(::std::nullopt);
}
::std::optional<TokenLoadType> symbolizeTokenLoadType(uint32_t value) {
  switch (value) {
  case 0: return TokenLoadType::None;
  case 1: return TokenLoadType::AsyncLoadOp;
  case 2: return TokenLoadType::TMALoadOp;
  case 3: return TokenLoadType::LocalStoreOp;
  case 4: return TokenLoadType::TmemLoadOp;
  default: return ::std::nullopt;
  }
}

bool TokenLoadTypeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 4)))));
}
TokenLoadTypeAttr TokenLoadTypeAttr::get(::mlir::MLIRContext *context, TokenLoadType val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<TokenLoadTypeAttr>(baseAttr);
}
TokenLoadType TokenLoadTypeAttr::getValue() const {
  return static_cast<TokenLoadType>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::triton::nvws
