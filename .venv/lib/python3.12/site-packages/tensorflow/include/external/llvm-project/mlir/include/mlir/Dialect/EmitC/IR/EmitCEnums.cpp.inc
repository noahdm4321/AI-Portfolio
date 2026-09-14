/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: EmitC.td                                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::emitc {

::llvm::StringRef stringifyCmpPredicate(CmpPredicate val) {
  switch (val) {
    case CmpPredicate::eq: return "eq";
    case CmpPredicate::ne: return "ne";
    case CmpPredicate::lt: return "lt";
    case CmpPredicate::le: return "le";
    case CmpPredicate::gt: return "gt";
    case CmpPredicate::ge: return "ge";
    case CmpPredicate::three_way: return "three_way";
  }
  return "";
}

::std::optional<CmpPredicate> symbolizeCmpPredicate(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CmpPredicate>>(str)
      .Case("eq", CmpPredicate::eq)
      .Case("ne", CmpPredicate::ne)
      .Case("lt", CmpPredicate::lt)
      .Case("le", CmpPredicate::le)
      .Case("gt", CmpPredicate::gt)
      .Case("ge", CmpPredicate::ge)
      .Case("three_way", CmpPredicate::three_way)
      .Default(::std::nullopt);
}
::std::optional<CmpPredicate> symbolizeCmpPredicate(uint64_t value) {
  switch (value) {
  case 0: return CmpPredicate::eq;
  case 1: return CmpPredicate::ne;
  case 2: return CmpPredicate::lt;
  case 3: return CmpPredicate::le;
  case 4: return CmpPredicate::gt;
  case 5: return CmpPredicate::ge;
  case 6: return CmpPredicate::three_way;
  default: return ::std::nullopt;
  }
}

bool CmpPredicateAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 0)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 1)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 2)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 3)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 4)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 5)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(64, 6)))));
}
CmpPredicateAttr CmpPredicateAttr::get(::mlir::MLIRContext *context, CmpPredicate val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 64);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint64_t>(val));
  return ::llvm::cast<CmpPredicateAttr>(baseAttr);
}
CmpPredicate CmpPredicateAttr::getValue() const {
  return static_cast<CmpPredicate>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::emitc
