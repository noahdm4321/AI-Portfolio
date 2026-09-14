/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::acc {

/// DefaultValue Clause
class ClauseDefaultValueAttr;
/// built-in reduction operations supported by OpenACC
class ReductionOperatorAttr;
/// Parallelism level (gang/worker/vector/seq)
class ParLevelAttr;
/// Encodes the options for kinds of recipes availabie in acc dialect
class RecipeKindAttr;
/// data clauses supported by OpenACC
class DataClauseAttr;
/// Captures data clause modifiers
class DataClauseModifierAttr;
class DeclareAttr;
class DeclareActionAttr;
/// built-in device type supported by OpenACC
class DeviceTypeAttr;
/// Differentiate the different gang arg values
class GangArgTypeAttr;
/// Differentiate between combined constructs
class CombinedConstructsTypeAttr;
/// constructs supported by OpenACC
class ConstructAttr;
class VarNameAttr;
/// Keeps track of associated acc routine information
/// This attribute is used to create the association between a function and
///     its `acc.routine` operation. A `func.func` uses this if its name
///     was referenced in an `acc routine` directive.
class RoutineInfoAttr;
/// Marks a specialized device version of an acc routine
/// This attribute is attached to a function that was specialized from a host
///     function marked with `acc.routine_info`. It captures the parallelism level,
///     a reference to the original `acc.routine` operation, and the original
///     function name (since the specialized function may be renamed).
/// 
///     Example - before specialization:
///     ```mlir
///     acc.routine @routine_gang func(@foo) gang
///     acc.routine @routine_vector func(@foo) vector
/// 
///     func.func @foo() attributes {
///       acc.routine_info = #acc.routine_info<[@routine_gang, @routine_vector]>
///     } { ... }
///     ```
/// 
///     After specialization, there are three functions: the original function and
///     two specialized versions (one per parallelism level):
///     ```mlir
///     acc.routine @routine_gang func(@foo) gang
///     acc.routine @routine_vector func(@foo) vector
/// 
///     // Original function (unchanged)
///     func.func @foo() attributes {
///       acc.routine_info = #acc.routine_info<[@routine_gang, @routine_vector]>
///     } { ... }
/// 
///     // Specialized for gang parallelism
///     func.func @foo_gang() attributes {
///       acc.specialized_routine = #acc.specialized_routine<@routine_gang, <gang_dim1>, "foo">
///     } { ... }
/// 
///     // Specialized for vector parallelism
///     func.func @foo_vector() attributes {
///       acc.specialized_routine = #acc.specialized_routine<@routine_vector, <vector>, "foo">
///     } { ... }
///     ```
class SpecializedRoutineAttr;
namespace detail {

struct ClauseDefaultValueAttrStorage;

} // namespace detail
class ClauseDefaultValueAttr : public ::mlir::Attribute::AttrBase<ClauseDefaultValueAttr, ::mlir::Attribute, detail::ClauseDefaultValueAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.defaultvalue";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static ClauseDefaultValueAttr get(::mlir::MLIRContext *context, ::mlir::acc::ClauseDefaultValue value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"defaultvalue"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::ClauseDefaultValue getValue() const;
};
namespace detail {

struct ReductionOperatorAttrStorage;

} // namespace detail
class ReductionOperatorAttr : public ::mlir::Attribute::AttrBase<ReductionOperatorAttr, ::mlir::Attribute, detail::ReductionOperatorAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.reduction_operator";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static ReductionOperatorAttr get(::mlir::MLIRContext *context, ::mlir::acc::ReductionOperator value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"reduction_operator"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::ReductionOperator getValue() const;
};
namespace detail {

struct ParLevelAttrStorage;

} // namespace detail
class ParLevelAttr : public ::mlir::Attribute::AttrBase<ParLevelAttr, ::mlir::Attribute, detail::ParLevelAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.par_level";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static ParLevelAttr get(::mlir::MLIRContext *context, ::mlir::acc::ParLevel value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"par_level"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::ParLevel getValue() const;
};
namespace detail {

struct RecipeKindAttrStorage;

} // namespace detail
class RecipeKindAttr : public ::mlir::Attribute::AttrBase<RecipeKindAttr, ::mlir::Attribute, detail::RecipeKindAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.recipe_kind";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static RecipeKindAttr get(::mlir::MLIRContext *context, ::mlir::acc::RecipeKind value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"recipe_kind"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::RecipeKind getValue() const;
};
namespace detail {

struct DataClauseAttrStorage;

} // namespace detail
class DataClauseAttr : public ::mlir::Attribute::AttrBase<DataClauseAttr, ::mlir::Attribute, detail::DataClauseAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.data_clause";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static DataClauseAttr get(::mlir::MLIRContext *context, ::mlir::acc::DataClause value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_clause"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::DataClause getValue() const;
};
namespace detail {

struct DataClauseModifierAttrStorage;

} // namespace detail
class DataClauseModifierAttr : public ::mlir::Attribute::AttrBase<DataClauseModifierAttr, ::mlir::Attribute, detail::DataClauseModifierAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.data_clause_modifier";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static DataClauseModifierAttr get(::mlir::MLIRContext *context, ::mlir::acc::DataClauseModifier value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"data_clause_modifier"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::DataClauseModifier getValue() const;
};
namespace detail {

struct DeclareAttrStorage;

} // namespace detail
class DeclareAttr : public ::mlir::Attribute::AttrBase<DeclareAttr, ::mlir::Attribute, detail::DeclareAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.declare";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static DeclareAttr get(::mlir::MLIRContext *context, DataClauseAttr dataClause, bool implicit);
  static DeclareAttr get(::mlir::MLIRContext *context, DataClauseAttr dataClause);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"declare"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  DataClauseAttr getDataClause() const;
  bool getImplicit() const;
};
namespace detail {

struct DeclareActionAttrStorage;

} // namespace detail
class DeclareActionAttr : public ::mlir::Attribute::AttrBase<DeclareActionAttr, ::mlir::Attribute, detail::DeclareActionAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.declare_action";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static DeclareActionAttr get(::mlir::MLIRContext *context, SymbolRefAttr preAlloc, SymbolRefAttr postAlloc, SymbolRefAttr preDealloc, SymbolRefAttr postDealloc);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"declare_action"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  SymbolRefAttr getPreAlloc() const;
  SymbolRefAttr getPostAlloc() const;
  SymbolRefAttr getPreDealloc() const;
  SymbolRefAttr getPostDealloc() const;
};
namespace detail {

struct DeviceTypeAttrStorage;

} // namespace detail
class DeviceTypeAttr : public ::mlir::Attribute::AttrBase<DeviceTypeAttr, ::mlir::Attribute, detail::DeviceTypeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.device_type";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static DeviceTypeAttr get(::mlir::MLIRContext *context, ::mlir::acc::DeviceType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"device_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::DeviceType getValue() const;
};
namespace detail {

struct GangArgTypeAttrStorage;

} // namespace detail
class GangArgTypeAttr : public ::mlir::Attribute::AttrBase<GangArgTypeAttr, ::mlir::Attribute, detail::GangArgTypeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.gang_arg_type";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static GangArgTypeAttr get(::mlir::MLIRContext *context, ::mlir::acc::GangArgType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"gang_arg_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::GangArgType getValue() const;
};
namespace detail {

struct CombinedConstructsTypeAttrStorage;

} // namespace detail
class CombinedConstructsTypeAttr : public ::mlir::Attribute::AttrBase<CombinedConstructsTypeAttr, ::mlir::Attribute, detail::CombinedConstructsTypeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.combined_constructs";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static CombinedConstructsTypeAttr get(::mlir::MLIRContext *context, ::mlir::acc::CombinedConstructsType value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"combined_constructs"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::CombinedConstructsType getValue() const;
};
namespace detail {

struct ConstructAttrStorage;

} // namespace detail
class ConstructAttr : public ::mlir::Attribute::AttrBase<ConstructAttr, ::mlir::Attribute, detail::ConstructAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.construct";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static ConstructAttr get(::mlir::MLIRContext *context, ::mlir::acc::Construct value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"construct"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::acc::Construct getValue() const;
};
namespace detail {

struct VarNameAttrStorage;

} // namespace detail
class VarNameAttr : public ::mlir::Attribute::AttrBase<VarNameAttr, ::mlir::Attribute, detail::VarNameAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.var_name";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static VarNameAttr get(::mlir::MLIRContext *context, ::llvm::StringRef name);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"var_name"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getName() const;
};
namespace detail {

struct RoutineInfoAttrStorage;

} // namespace detail
class RoutineInfoAttr : public ::mlir::Attribute::AttrBase<RoutineInfoAttr, ::mlir::Attribute, detail::RoutineInfoAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.routine_info";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static RoutineInfoAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<SymbolRefAttr> accRoutines);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"routine_info"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<SymbolRefAttr> getAccRoutines() const;
};
namespace detail {

struct SpecializedRoutineAttrStorage;

} // namespace detail
class SpecializedRoutineAttr : public ::mlir::Attribute::AttrBase<SpecializedRoutineAttr, ::mlir::Attribute, detail::SpecializedRoutineAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "acc.specialized_routine";
  static constexpr ::llvm::StringLiteral dialectName = "acc";
  static SpecializedRoutineAttr get(::mlir::MLIRContext *context, SymbolRefAttr routine, ParLevelAttr level, StringAttr funcName);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"specialized_routine"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  SymbolRefAttr getRoutine() const;
  ParLevelAttr getLevel() const;
  StringAttr getFuncName() const;
};

} // namespace mlir::acc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ClauseDefaultValueAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ReductionOperatorAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ParLevelAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::RecipeKindAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DataClauseAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DataClauseModifierAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DeclareAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DeclareActionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DeviceTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::GangArgTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::CombinedConstructsTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ConstructAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::VarNameAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::RoutineInfoAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::SpecializedRoutineAttr)

#endif // GET_ATTRDEF_CLASSES

