/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: XeVMOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::xevm {

::llvm::StringRef stringifyAddrSpace(AddrSpace val) {
  switch (val) {
    case AddrSpace::PRIVATE: return "private";
    case AddrSpace::GLOBAL: return "global";
    case AddrSpace::CONSTANT: return "constant";
    case AddrSpace::SHARED: return "shared";
    case AddrSpace::GENERIC: return "generic";
  }
  return "";
}

::std::optional<AddrSpace> symbolizeAddrSpace(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<AddrSpace>>(str)
      .Case("private", AddrSpace::PRIVATE)
      .Case("global", AddrSpace::GLOBAL)
      .Case("constant", AddrSpace::CONSTANT)
      .Case("shared", AddrSpace::SHARED)
      .Case("generic", AddrSpace::GENERIC)
      .Default(::std::nullopt);
}
::std::optional<AddrSpace> symbolizeAddrSpace(uint32_t value) {
  switch (value) {
  case 0: return AddrSpace::PRIVATE;
  case 1: return AddrSpace::GLOBAL;
  case 2: return AddrSpace::CONSTANT;
  case 3: return AddrSpace::SHARED;
  case 4: return AddrSpace::GENERIC;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::xevm
namespace mlir::xevm {

::llvm::StringRef stringifyElemType(ElemType val) {
  switch (val) {
    case ElemType::BF16: return "bf16";
    case ElemType::F16: return "f16";
    case ElemType::S8: return "s8";
    case ElemType::U8: return "u8";
    case ElemType::S4: return "s4";
    case ElemType::U4: return "u4";
    case ElemType::TF32: return "tf32";
    case ElemType::F32: return "f32";
    case ElemType::S32: return "s32";
  }
  return "";
}

::std::optional<ElemType> symbolizeElemType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ElemType>>(str)
      .Case("bf16", ElemType::BF16)
      .Case("f16", ElemType::F16)
      .Case("s8", ElemType::S8)
      .Case("u8", ElemType::U8)
      .Case("s4", ElemType::S4)
      .Case("u4", ElemType::U4)
      .Case("tf32", ElemType::TF32)
      .Case("f32", ElemType::F32)
      .Case("s32", ElemType::S32)
      .Default(::std::nullopt);
}
::std::optional<ElemType> symbolizeElemType(uint32_t value) {
  switch (value) {
  case 8: return ElemType::BF16;
  case 9: return ElemType::F16;
  case 10: return ElemType::S8;
  case 11: return ElemType::U8;
  case 12: return ElemType::S4;
  case 13: return ElemType::U4;
  case 14: return ElemType::TF32;
  case 15: return ElemType::F32;
  case 16: return ElemType::S32;
  default: return ::std::nullopt;
  }
}

bool ElemTypeAttr::classof(::mlir::Attribute attr) {
  return (((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && (((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 8)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 9)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 10)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 11)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 12)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 13)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 14)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 15)))) || ((
    ::llvm::cast<::mlir::IntegerAttr>(attr).getValue().eq(::llvm::APInt(32, 16)))));
}
ElemTypeAttr ElemTypeAttr::get(::mlir::MLIRContext *context, ElemType val) {
  ::mlir::IntegerType intType = ::mlir::IntegerType::get(context, 32);
  ::mlir::IntegerAttr baseAttr = ::mlir::IntegerAttr::get(intType, static_cast<uint32_t>(val));
  return ::llvm::cast<ElemTypeAttr>(baseAttr);
}
ElemType ElemTypeAttr::getValue() const {
  return static_cast<ElemType>(::mlir::IntegerAttr::getValue().getZExtValue());
}

} // namespace mlir::xevm
namespace mlir::xevm {

::llvm::StringRef stringifyLoadCacheControl(LoadCacheControl val) {
  switch (val) {
    case LoadCacheControl::L1UC_L2UC_L3UC: return "L1uc_L2uc_L3uc";
    case LoadCacheControl::L1UC_L2UC_L3C: return "L1uc_L2uc_L3c";
    case LoadCacheControl::L1UC_L2C_L3UC: return "L1uc_L2c_L3uc";
    case LoadCacheControl::L1UC_L2C_L3C: return "L1uc_L2c_L3c";
    case LoadCacheControl::L1C_L2UC_L3UC: return "L1c_L2uc_L3uc";
    case LoadCacheControl::L1C_L2UC_L3C: return "L1c_L2uc_L3c";
    case LoadCacheControl::L1C_L2C_L3UC: return "L1c_L2c_L3uc";
    case LoadCacheControl::L1C_L2C_L3C: return "L1c_L2c_L3c";
    case LoadCacheControl::L1S_L2UC_L3UC: return "L1s_L2uc_L3uc";
    case LoadCacheControl::L1S_L2UC_L3C: return "L1s_L2uc_L3c";
    case LoadCacheControl::L1S_L2C_L3UC: return "L1s_L2c_L3uc";
    case LoadCacheControl::L1S_L2C_L3C: return "L1s_L2c_L3c";
    case LoadCacheControl::INVALIDATE_READ: return "ir";
  }
  return "";
}

::std::optional<LoadCacheControl> symbolizeLoadCacheControl(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<LoadCacheControl>>(str)
      .Case("L1uc_L2uc_L3uc", LoadCacheControl::L1UC_L2UC_L3UC)
      .Case("L1uc_L2uc_L3c", LoadCacheControl::L1UC_L2UC_L3C)
      .Case("L1uc_L2c_L3uc", LoadCacheControl::L1UC_L2C_L3UC)
      .Case("L1uc_L2c_L3c", LoadCacheControl::L1UC_L2C_L3C)
      .Case("L1c_L2uc_L3uc", LoadCacheControl::L1C_L2UC_L3UC)
      .Case("L1c_L2uc_L3c", LoadCacheControl::L1C_L2UC_L3C)
      .Case("L1c_L2c_L3uc", LoadCacheControl::L1C_L2C_L3UC)
      .Case("L1c_L2c_L3c", LoadCacheControl::L1C_L2C_L3C)
      .Case("L1s_L2uc_L3uc", LoadCacheControl::L1S_L2UC_L3UC)
      .Case("L1s_L2uc_L3c", LoadCacheControl::L1S_L2UC_L3C)
      .Case("L1s_L2c_L3uc", LoadCacheControl::L1S_L2C_L3UC)
      .Case("L1s_L2c_L3c", LoadCacheControl::L1S_L2C_L3C)
      .Case("ir", LoadCacheControl::INVALIDATE_READ)
      .Default(::std::nullopt);
}
::std::optional<LoadCacheControl> symbolizeLoadCacheControl(uint32_t value) {
  switch (value) {
  case 1: return LoadCacheControl::L1UC_L2UC_L3UC;
  case 2: return LoadCacheControl::L1UC_L2UC_L3C;
  case 3: return LoadCacheControl::L1UC_L2C_L3UC;
  case 4: return LoadCacheControl::L1UC_L2C_L3C;
  case 5: return LoadCacheControl::L1C_L2UC_L3UC;
  case 6: return LoadCacheControl::L1C_L2UC_L3C;
  case 7: return LoadCacheControl::L1C_L2C_L3UC;
  case 8: return LoadCacheControl::L1C_L2C_L3C;
  case 9: return LoadCacheControl::L1S_L2UC_L3UC;
  case 10: return LoadCacheControl::L1S_L2UC_L3C;
  case 11: return LoadCacheControl::L1S_L2C_L3UC;
  case 12: return LoadCacheControl::L1S_L2C_L3C;
  case 13: return LoadCacheControl::INVALIDATE_READ;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::xevm
namespace mlir::xevm {

::llvm::StringRef stringifyMemScope(MemScope val) {
  switch (val) {
    case MemScope::LANE: return "lane";
    case MemScope::SUBGROUP: return "subgroup";
    case MemScope::WORKGROUP: return "workgroup";
    case MemScope::CLUSTER: return "cluster";
    case MemScope::DEVICE: return "device";
    case MemScope::SYSTEM: return "system";
  }
  return "";
}

::std::optional<MemScope> symbolizeMemScope(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<MemScope>>(str)
      .Case("lane", MemScope::LANE)
      .Case("subgroup", MemScope::SUBGROUP)
      .Case("workgroup", MemScope::WORKGROUP)
      .Case("cluster", MemScope::CLUSTER)
      .Case("device", MemScope::DEVICE)
      .Case("system", MemScope::SYSTEM)
      .Default(::std::nullopt);
}
::std::optional<MemScope> symbolizeMemScope(uint32_t value) {
  switch (value) {
  case 0: return MemScope::LANE;
  case 1: return MemScope::SUBGROUP;
  case 2: return MemScope::WORKGROUP;
  case 3: return MemScope::CLUSTER;
  case 4: return MemScope::DEVICE;
  case 5: return MemScope::SYSTEM;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::xevm
namespace mlir::xevm {

::llvm::StringRef stringifyStoreCacheControl(StoreCacheControl val) {
  switch (val) {
    case StoreCacheControl::L1UC_L2UC_L3UC: return "L1uc_L2uc_L3uc";
    case StoreCacheControl::L1UC_L2UC_L3WB: return "L1uc_L2uc_L3wb";
    case StoreCacheControl::L1UC_L2WB_L3UC: return "L1uc_L2wb_L3uc";
    case StoreCacheControl::L1UC_L2WB_L3WB: return "L1uc_L2wb_L3wb";
    case StoreCacheControl::L1WT_L2UC_L3UC: return "L1wt_L2uc_L3uc";
    case StoreCacheControl::L1WT_L2UC_L3WB: return "L1wt_L2uc_L3wb";
    case StoreCacheControl::L1WT_L2WB_L3UC: return "L1wt_L2wb_L3uc";
    case StoreCacheControl::L1WT_L2WB_L3WB: return "L1wt_L2wb_L3wb";
    case StoreCacheControl::L1S_L2UC_L3UC: return "L1s_L2uc_L3uc";
    case StoreCacheControl::L1S_L2UC_L3WB: return "L1s_L2uc_L3wb";
    case StoreCacheControl::L1S_L2WB_L3UC: return "L1s_L2wb_L3uc";
    case StoreCacheControl::L1S_L2WB_L3WB: return "L1s_L2wb_L3wb";
    case StoreCacheControl::L1WB_L2UC_L3UC: return "L1wb_L2uc_L3uc";
    case StoreCacheControl::L1WB_L2WB_L3UC: return "L1wb_L2wb_L3uc";
    case StoreCacheControl::L1WB_L2UC_L3WB: return "L1wb_L2uc_L3wb";
  }
  return "";
}

::std::optional<StoreCacheControl> symbolizeStoreCacheControl(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<StoreCacheControl>>(str)
      .Case("L1uc_L2uc_L3uc", StoreCacheControl::L1UC_L2UC_L3UC)
      .Case("L1uc_L2uc_L3wb", StoreCacheControl::L1UC_L2UC_L3WB)
      .Case("L1uc_L2wb_L3uc", StoreCacheControl::L1UC_L2WB_L3UC)
      .Case("L1uc_L2wb_L3wb", StoreCacheControl::L1UC_L2WB_L3WB)
      .Case("L1wt_L2uc_L3uc", StoreCacheControl::L1WT_L2UC_L3UC)
      .Case("L1wt_L2uc_L3wb", StoreCacheControl::L1WT_L2UC_L3WB)
      .Case("L1wt_L2wb_L3uc", StoreCacheControl::L1WT_L2WB_L3UC)
      .Case("L1wt_L2wb_L3wb", StoreCacheControl::L1WT_L2WB_L3WB)
      .Case("L1s_L2uc_L3uc", StoreCacheControl::L1S_L2UC_L3UC)
      .Case("L1s_L2uc_L3wb", StoreCacheControl::L1S_L2UC_L3WB)
      .Case("L1s_L2wb_L3uc", StoreCacheControl::L1S_L2WB_L3UC)
      .Case("L1s_L2wb_L3wb", StoreCacheControl::L1S_L2WB_L3WB)
      .Case("L1wb_L2uc_L3uc", StoreCacheControl::L1WB_L2UC_L3UC)
      .Case("L1wb_L2wb_L3uc", StoreCacheControl::L1WB_L2WB_L3UC)
      .Case("L1wb_L2uc_L3wb", StoreCacheControl::L1WB_L2UC_L3WB)
      .Default(::std::nullopt);
}
::std::optional<StoreCacheControl> symbolizeStoreCacheControl(uint32_t value) {
  switch (value) {
  case 1: return StoreCacheControl::L1UC_L2UC_L3UC;
  case 2: return StoreCacheControl::L1UC_L2UC_L3WB;
  case 3: return StoreCacheControl::L1UC_L2WB_L3UC;
  case 4: return StoreCacheControl::L1UC_L2WB_L3WB;
  case 5: return StoreCacheControl::L1WT_L2UC_L3UC;
  case 6: return StoreCacheControl::L1WT_L2UC_L3WB;
  case 7: return StoreCacheControl::L1WT_L2WB_L3UC;
  case 8: return StoreCacheControl::L1WT_L2WB_L3WB;
  case 9: return StoreCacheControl::L1S_L2UC_L3UC;
  case 10: return StoreCacheControl::L1S_L2UC_L3WB;
  case 11: return StoreCacheControl::L1S_L2WB_L3UC;
  case 12: return StoreCacheControl::L1S_L2WB_L3WB;
  case 13: return StoreCacheControl::L1WB_L2UC_L3UC;
  case 14: return StoreCacheControl::L1WB_L2WB_L3UC;
  case 15: return StoreCacheControl::L1WB_L2UC_L3WB;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::xevm
