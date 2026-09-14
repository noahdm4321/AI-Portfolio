/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OpenACCOps.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::acc {

::llvm::StringRef stringifyClauseDefaultValue(ClauseDefaultValue val) {
  switch (val) {
    case ClauseDefaultValue::Present: return "present";
    case ClauseDefaultValue::None: return "none";
  }
  return "";
}

::std::optional<ClauseDefaultValue> symbolizeClauseDefaultValue(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ClauseDefaultValue>>(str)
      .Case("present", ClauseDefaultValue::Present)
      .Case("none", ClauseDefaultValue::None)
      .Default(::std::nullopt);
}
::std::optional<ClauseDefaultValue> symbolizeClauseDefaultValue(uint32_t value) {
  switch (value) {
  case 0: return ClauseDefaultValue::Present;
  case 1: return ClauseDefaultValue::None;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyCombinedConstructsType(CombinedConstructsType val) {
  switch (val) {
    case CombinedConstructsType::KernelsLoop: return "kernels_loop";
    case CombinedConstructsType::ParallelLoop: return "parallel_loop";
    case CombinedConstructsType::SerialLoop: return "serial_loop";
  }
  return "";
}

::std::optional<CombinedConstructsType> symbolizeCombinedConstructsType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<CombinedConstructsType>>(str)
      .Case("kernels_loop", CombinedConstructsType::KernelsLoop)
      .Case("parallel_loop", CombinedConstructsType::ParallelLoop)
      .Case("serial_loop", CombinedConstructsType::SerialLoop)
      .Default(::std::nullopt);
}
::std::optional<CombinedConstructsType> symbolizeCombinedConstructsType(uint32_t value) {
  switch (value) {
  case 1: return CombinedConstructsType::KernelsLoop;
  case 2: return CombinedConstructsType::ParallelLoop;
  case 3: return CombinedConstructsType::SerialLoop;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyConstruct(Construct val) {
  switch (val) {
    case Construct::acc_construct_parallel: return "acc_construct_parallel";
    case Construct::acc_construct_kernels: return "acc_construct_kernels";
    case Construct::acc_construct_loop: return "acc_construct_loop";
    case Construct::acc_construct_data: return "acc_construct_data";
    case Construct::acc_construct_enter_data: return "acc_construct_enter_data";
    case Construct::acc_construct_exit_data: return "acc_construct_exit_data";
    case Construct::acc_construct_host_data: return "acc_construct_host_data";
    case Construct::acc_construct_atomic: return "acc_construct_atomic";
    case Construct::acc_construct_declare: return "acc_construct_declare";
    case Construct::acc_construct_init: return "acc_construct_init";
    case Construct::acc_construct_shutdown: return "acc_construct_shutdown";
    case Construct::acc_construct_set: return "acc_construct_set";
    case Construct::acc_construct_update: return "acc_construct_update";
    case Construct::acc_construct_routine: return "acc_construct_routine";
    case Construct::acc_construct_wait: return "acc_construct_wait";
    case Construct::acc_construct_runtime_api: return "acc_construct_runtime_api";
    case Construct::acc_construct_serial: return "acc_construct_serial";
  }
  return "";
}

::std::optional<Construct> symbolizeConstruct(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<Construct>>(str)
      .Case("acc_construct_parallel", Construct::acc_construct_parallel)
      .Case("acc_construct_kernels", Construct::acc_construct_kernels)
      .Case("acc_construct_loop", Construct::acc_construct_loop)
      .Case("acc_construct_data", Construct::acc_construct_data)
      .Case("acc_construct_enter_data", Construct::acc_construct_enter_data)
      .Case("acc_construct_exit_data", Construct::acc_construct_exit_data)
      .Case("acc_construct_host_data", Construct::acc_construct_host_data)
      .Case("acc_construct_atomic", Construct::acc_construct_atomic)
      .Case("acc_construct_declare", Construct::acc_construct_declare)
      .Case("acc_construct_init", Construct::acc_construct_init)
      .Case("acc_construct_shutdown", Construct::acc_construct_shutdown)
      .Case("acc_construct_set", Construct::acc_construct_set)
      .Case("acc_construct_update", Construct::acc_construct_update)
      .Case("acc_construct_routine", Construct::acc_construct_routine)
      .Case("acc_construct_wait", Construct::acc_construct_wait)
      .Case("acc_construct_runtime_api", Construct::acc_construct_runtime_api)
      .Case("acc_construct_serial", Construct::acc_construct_serial)
      .Default(::std::nullopt);
}
::std::optional<Construct> symbolizeConstruct(uint64_t value) {
  switch (value) {
  case 0: return Construct::acc_construct_parallel;
  case 1: return Construct::acc_construct_kernels;
  case 2: return Construct::acc_construct_loop;
  case 3: return Construct::acc_construct_data;
  case 4: return Construct::acc_construct_enter_data;
  case 5: return Construct::acc_construct_exit_data;
  case 6: return Construct::acc_construct_host_data;
  case 7: return Construct::acc_construct_atomic;
  case 8: return Construct::acc_construct_declare;
  case 9: return Construct::acc_construct_init;
  case 10: return Construct::acc_construct_shutdown;
  case 11: return Construct::acc_construct_set;
  case 12: return Construct::acc_construct_update;
  case 13: return Construct::acc_construct_routine;
  case 14: return Construct::acc_construct_wait;
  case 15: return Construct::acc_construct_runtime_api;
  case 16: return Construct::acc_construct_serial;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyDataClause(DataClause val) {
  switch (val) {
    case DataClause::acc_copyin: return "acc_copyin";
    case DataClause::acc_copyin_readonly: return "acc_copyin_readonly";
    case DataClause::acc_copy: return "acc_copy";
    case DataClause::acc_copyout: return "acc_copyout";
    case DataClause::acc_copyout_zero: return "acc_copyout_zero";
    case DataClause::acc_present: return "acc_present";
    case DataClause::acc_create: return "acc_create";
    case DataClause::acc_create_zero: return "acc_create_zero";
    case DataClause::acc_delete: return "acc_delete";
    case DataClause::acc_attach: return "acc_attach";
    case DataClause::acc_detach: return "acc_detach";
    case DataClause::acc_no_create: return "acc_no_create";
    case DataClause::acc_private: return "acc_private";
    case DataClause::acc_firstprivate: return "acc_firstprivate";
    case DataClause::acc_deviceptr: return "acc_deviceptr";
    case DataClause::acc_getdeviceptr: return "acc_getdeviceptr";
    case DataClause::acc_update_host: return "acc_update_host";
    case DataClause::acc_update_self: return "acc_update_self";
    case DataClause::acc_update_device: return "acc_update_device";
    case DataClause::acc_use_device: return "acc_use_device";
    case DataClause::acc_reduction: return "acc_reduction";
    case DataClause::acc_declare_device_resident: return "acc_declare_device_resident";
    case DataClause::acc_declare_link: return "acc_declare_link";
    case DataClause::acc_cache: return "acc_cache";
    case DataClause::acc_cache_readonly: return "acc_cache_readonly";
  }
  return "";
}

::std::optional<DataClause> symbolizeDataClause(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DataClause>>(str)
      .Case("acc_copyin", DataClause::acc_copyin)
      .Case("acc_copyin_readonly", DataClause::acc_copyin_readonly)
      .Case("acc_copy", DataClause::acc_copy)
      .Case("acc_copyout", DataClause::acc_copyout)
      .Case("acc_copyout_zero", DataClause::acc_copyout_zero)
      .Case("acc_present", DataClause::acc_present)
      .Case("acc_create", DataClause::acc_create)
      .Case("acc_create_zero", DataClause::acc_create_zero)
      .Case("acc_delete", DataClause::acc_delete)
      .Case("acc_attach", DataClause::acc_attach)
      .Case("acc_detach", DataClause::acc_detach)
      .Case("acc_no_create", DataClause::acc_no_create)
      .Case("acc_private", DataClause::acc_private)
      .Case("acc_firstprivate", DataClause::acc_firstprivate)
      .Case("acc_deviceptr", DataClause::acc_deviceptr)
      .Case("acc_getdeviceptr", DataClause::acc_getdeviceptr)
      .Case("acc_update_host", DataClause::acc_update_host)
      .Case("acc_update_self", DataClause::acc_update_self)
      .Case("acc_update_device", DataClause::acc_update_device)
      .Case("acc_use_device", DataClause::acc_use_device)
      .Case("acc_reduction", DataClause::acc_reduction)
      .Case("acc_declare_device_resident", DataClause::acc_declare_device_resident)
      .Case("acc_declare_link", DataClause::acc_declare_link)
      .Case("acc_cache", DataClause::acc_cache)
      .Case("acc_cache_readonly", DataClause::acc_cache_readonly)
      .Default(::std::nullopt);
}
::std::optional<DataClause> symbolizeDataClause(uint64_t value) {
  switch (value) {
  case 1: return DataClause::acc_copyin;
  case 2: return DataClause::acc_copyin_readonly;
  case 3: return DataClause::acc_copy;
  case 4: return DataClause::acc_copyout;
  case 5: return DataClause::acc_copyout_zero;
  case 6: return DataClause::acc_present;
  case 7: return DataClause::acc_create;
  case 8: return DataClause::acc_create_zero;
  case 9: return DataClause::acc_delete;
  case 10: return DataClause::acc_attach;
  case 11: return DataClause::acc_detach;
  case 12: return DataClause::acc_no_create;
  case 13: return DataClause::acc_private;
  case 14: return DataClause::acc_firstprivate;
  case 15: return DataClause::acc_deviceptr;
  case 16: return DataClause::acc_getdeviceptr;
  case 17: return DataClause::acc_update_host;
  case 18: return DataClause::acc_update_self;
  case 19: return DataClause::acc_update_device;
  case 20: return DataClause::acc_use_device;
  case 21: return DataClause::acc_reduction;
  case 22: return DataClause::acc_declare_device_resident;
  case 23: return DataClause::acc_declare_link;
  case 24: return DataClause::acc_cache;
  case 25: return DataClause::acc_cache_readonly;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

std::string stringifyDataClauseModifier(DataClauseModifier symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(31u == (31u | val) && "invalid bits set in bit enum");
  // Special case for all bits unset.
  if (val == 0) return "none";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (12u == (12u & val)) {
    strs.push_back("always");
    val &= ~static_cast<uint32_t>(12);
  }

  if (1u == (1u & val))
    strs.push_back("zero");

  if (2u == (2u & val))
    strs.push_back("readonly");

  if (4u == (4u & val))
    strs.push_back("alwaysin");

  if (8u == (8u & val))
    strs.push_back("alwaysout");

  if (16u == (16u & val))
    strs.push_back("capture");
  return ::llvm::join(strs, ",");
}

::std::optional<DataClauseModifier> symbolizeDataClauseModifier(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "none") return DataClauseModifier::none;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, ",");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("zero", 1)
      .Case("readonly", 2)
      .Case("alwaysin", 4)
      .Case("alwaysout", 8)
      .Case("always", 12)
      .Case("capture", 16)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<DataClauseModifier>(val);
}

::std::optional<DataClauseModifier> symbolizeDataClauseModifier(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return DataClauseModifier::none;

  if (value & ~static_cast<uint32_t>(31u)) return std::nullopt;
  return static_cast<DataClauseModifier>(value);
}

} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyDeviceType(DeviceType val) {
  switch (val) {
    case DeviceType::None: return "none";
    case DeviceType::Star: return "star";
    case DeviceType::Default: return "default";
    case DeviceType::Host: return "host";
    case DeviceType::Multicore: return "multicore";
    case DeviceType::Nvidia: return "nvidia";
    case DeviceType::Radeon: return "radeon";
  }
  return "";
}

::std::optional<DeviceType> symbolizeDeviceType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<DeviceType>>(str)
      .Case("none", DeviceType::None)
      .Case("star", DeviceType::Star)
      .Case("default", DeviceType::Default)
      .Case("host", DeviceType::Host)
      .Case("multicore", DeviceType::Multicore)
      .Case("nvidia", DeviceType::Nvidia)
      .Case("radeon", DeviceType::Radeon)
      .Default(::std::nullopt);
}
::std::optional<DeviceType> symbolizeDeviceType(uint32_t value) {
  switch (value) {
  case 0: return DeviceType::None;
  case 1: return DeviceType::Star;
  case 2: return DeviceType::Default;
  case 3: return DeviceType::Host;
  case 4: return DeviceType::Multicore;
  case 5: return DeviceType::Nvidia;
  case 6: return DeviceType::Radeon;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyGangArgType(GangArgType val) {
  switch (val) {
    case GangArgType::Num: return "Num";
    case GangArgType::Dim: return "Dim";
    case GangArgType::Static: return "Static";
  }
  return "";
}

::std::optional<GangArgType> symbolizeGangArgType(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<GangArgType>>(str)
      .Case("Num", GangArgType::Num)
      .Case("Dim", GangArgType::Dim)
      .Case("Static", GangArgType::Static)
      .Default(::std::nullopt);
}
::std::optional<GangArgType> symbolizeGangArgType(uint32_t value) {
  switch (value) {
  case 0: return GangArgType::Num;
  case 1: return GangArgType::Dim;
  case 2: return GangArgType::Static;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyLoopParMode(LoopParMode val) {
  switch (val) {
    case LoopParMode::loop_auto: return "loop_auto";
    case LoopParMode::loop_independent: return "loop_independent";
    case LoopParMode::loop_seq: return "loop_seq";
  }
  return "";
}

::std::optional<LoopParMode> symbolizeLoopParMode(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<LoopParMode>>(str)
      .Case("loop_auto", LoopParMode::loop_auto)
      .Case("loop_independent", LoopParMode::loop_independent)
      .Case("loop_seq", LoopParMode::loop_seq)
      .Default(::std::nullopt);
}
::std::optional<LoopParMode> symbolizeLoopParMode(uint32_t value) {
  switch (value) {
  case 1: return LoopParMode::loop_auto;
  case 2: return LoopParMode::loop_independent;
  case 0: return LoopParMode::loop_seq;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyParLevel(ParLevel val) {
  switch (val) {
    case ParLevel::seq: return "seq";
    case ParLevel::gang_dim1: return "gang_dim1";
    case ParLevel::gang_dim2: return "gang_dim2";
    case ParLevel::gang_dim3: return "gang_dim3";
    case ParLevel::worker: return "worker";
    case ParLevel::vector: return "vector";
  }
  return "";
}

::std::optional<ParLevel> symbolizeParLevel(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ParLevel>>(str)
      .Case("seq", ParLevel::seq)
      .Case("gang_dim1", ParLevel::gang_dim1)
      .Case("gang_dim2", ParLevel::gang_dim2)
      .Case("gang_dim3", ParLevel::gang_dim3)
      .Case("worker", ParLevel::worker)
      .Case("vector", ParLevel::vector)
      .Default(::std::nullopt);
}
::std::optional<ParLevel> symbolizeParLevel(uint32_t value) {
  switch (value) {
  case 0: return ParLevel::seq;
  case 1: return ParLevel::gang_dim1;
  case 2: return ParLevel::gang_dim2;
  case 3: return ParLevel::gang_dim3;
  case 4: return ParLevel::worker;
  case 5: return ParLevel::vector;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyRecipeKind(RecipeKind val) {
  switch (val) {
    case RecipeKind::private_recipe: return "private_recipe";
    case RecipeKind::firstprivate_recipe: return "firstprivate_recipe";
    case RecipeKind::reduction_recipe: return "reduction_recipe";
  }
  return "";
}

::std::optional<RecipeKind> symbolizeRecipeKind(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<RecipeKind>>(str)
      .Case("private_recipe", RecipeKind::private_recipe)
      .Case("firstprivate_recipe", RecipeKind::firstprivate_recipe)
      .Case("reduction_recipe", RecipeKind::reduction_recipe)
      .Default(::std::nullopt);
}
::std::optional<RecipeKind> symbolizeRecipeKind(uint32_t value) {
  switch (value) {
  case 0: return RecipeKind::private_recipe;
  case 1: return RecipeKind::firstprivate_recipe;
  case 2: return RecipeKind::reduction_recipe;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

::llvm::StringRef stringifyReductionOperator(ReductionOperator val) {
  switch (val) {
    case ReductionOperator::AccNone: return "none";
    case ReductionOperator::AccAdd: return "add";
    case ReductionOperator::AccMul: return "mul";
    case ReductionOperator::AccMax: return "max";
    case ReductionOperator::AccMin: return "min";
    case ReductionOperator::AccIand: return "iand";
    case ReductionOperator::AccIor: return "ior";
    case ReductionOperator::AccXor: return "xor";
    case ReductionOperator::AccEqv: return "eqv";
    case ReductionOperator::AccNeqv: return "neqv";
    case ReductionOperator::AccLand: return "land";
    case ReductionOperator::AccLor: return "lor";
  }
  return "";
}

::std::optional<ReductionOperator> symbolizeReductionOperator(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::std::optional<ReductionOperator>>(str)
      .Case("none", ReductionOperator::AccNone)
      .Case("add", ReductionOperator::AccAdd)
      .Case("mul", ReductionOperator::AccMul)
      .Case("max", ReductionOperator::AccMax)
      .Case("min", ReductionOperator::AccMin)
      .Case("iand", ReductionOperator::AccIand)
      .Case("ior", ReductionOperator::AccIor)
      .Case("xor", ReductionOperator::AccXor)
      .Case("eqv", ReductionOperator::AccEqv)
      .Case("neqv", ReductionOperator::AccNeqv)
      .Case("land", ReductionOperator::AccLand)
      .Case("lor", ReductionOperator::AccLor)
      .Default(::std::nullopt);
}
::std::optional<ReductionOperator> symbolizeReductionOperator(uint32_t value) {
  switch (value) {
  case 0: return ReductionOperator::AccNone;
  case 1: return ReductionOperator::AccAdd;
  case 2: return ReductionOperator::AccMul;
  case 3: return ReductionOperator::AccMax;
  case 4: return ReductionOperator::AccMin;
  case 5: return ReductionOperator::AccIand;
  case 6: return ReductionOperator::AccIor;
  case 7: return ReductionOperator::AccXor;
  case 8: return ReductionOperator::AccEqv;
  case 9: return ReductionOperator::AccNeqv;
  case 10: return ReductionOperator::AccLand;
  case 11: return ReductionOperator::AccLor;
  default: return ::std::nullopt;
  }
}


} // namespace mlir::acc
namespace mlir::acc {

std::string stringifyVariableTypeCategory(VariableTypeCategory symbol) {
  auto val = static_cast<uint32_t>(symbol);
  assert(15u == (15u | val) && "invalid bits set in bit enum");
  // Special case for all bits unset.
  if (val == 0) return "uncategorized";

  ::llvm::SmallVector<::llvm::StringRef, 2> strs;
  // Print bit enum groups before individual bits

  if (14u == (14u & val)) {
    strs.push_back("aggregate");
    val &= ~static_cast<uint32_t>(14);
  }

  if (1u == (1u & val))
    strs.push_back("scalar");

  if (2u == (2u & val))
    strs.push_back("array");

  if (4u == (4u & val))
    strs.push_back("composite");

  if (8u == (8u & val))
    strs.push_back("nonscalar");
  return ::llvm::join(strs, ",");
}

::std::optional<VariableTypeCategory> symbolizeVariableTypeCategory(::llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "uncategorized") return VariableTypeCategory::uncategorized;

  ::llvm::SmallVector<::llvm::StringRef, 2> symbols;
  str.split(symbols, ",");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<::std::optional<uint32_t>>(symbol.trim())
      .Case("scalar", 1)
      .Case("array", 2)
      .Case("composite", 4)
      .Case("nonscalar", 8)
      .Case("aggregate", 14)
      .Default(::std::nullopt);
    if (bit) { val |= *bit; } else { return ::std::nullopt; }
  }
  return static_cast<VariableTypeCategory>(val);
}

::std::optional<VariableTypeCategory> symbolizeVariableTypeCategory(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return VariableTypeCategory::uncategorized;

  if (value & ~static_cast<uint32_t>(15u)) return std::nullopt;
  return static_cast<VariableTypeCategory>(value);
}

} // namespace mlir::acc
