/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonInstrumentOps.td                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::instrument {

::llvm::StringRef stringifyMemType(MemType val) {
  switch (val) {
    case MemType::SHARED_MEM: return "shared_mem";
    case MemType::TENSOR_MEM: return "tensor_mem";
  }
  return "";
}

::std::optional<MemType> symbolizeMemType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MemType>>(str)
      .Case("shared_mem", MemType::SHARED_MEM)
      .Case("tensor_mem", MemType::TENSOR_MEM)
      .Default(::std::nullopt);
}
::std::optional<MemType> symbolizeMemType(uint32_t value) {
  switch (value) {
  case 0: return MemType::SHARED_MEM;
  case 1: return MemType::TENSOR_MEM;
  default: return ::std::nullopt;
  }
}

bool MemTypeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 1)))));
}
MemTypeAttr MemTypeAttr::get(::mlir::MLIRContext *context, MemType val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<MemTypeAttr>(baseAttr);
}
MemType MemTypeAttr::getValue() const {
  return static_cast<MemType>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::triton::instrument
