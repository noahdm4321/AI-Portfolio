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
namespace mlir::xevm {

/// Describe the cache settings for load operators
class LoadCacheControlAttr;
/// Describe the cache settings for store operators
class StoreCacheControlAttr;
/// Describe memory scopes
class MemScopeAttr;
/// Describe address spaces
class AddrSpaceAttr;
/// MMA operation is represented as D=AxB+C, where
///       - A has the shape MxK.
///       - B has the shape KxN.
///       - D and C have the shape MxN.
///     This attribute encodes the shape of all matrices that participate in MMA.
class MMAShapeAttr;
class MMATypesAttr;
/// GPU target attribute for controlling compilation of Intel GPU targets. All
///     parameters decay into default values if not present.
/// 
///     Examples:
/// 
///     1. Target with default values.
///     ```
///       gpu.module @mymodule [#xevm.target] attributes {...} {
///         ...
///       }
///     ```
class XeVMTargetAttr;
namespace detail {

struct LoadCacheControlAttrStorage;

} // namespace detail
class LoadCacheControlAttr : public ::mlir::Attribute::AttrBase<LoadCacheControlAttr, ::mlir::Attribute, detail::LoadCacheControlAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xevm.load_cache_control";
  static constexpr ::llvm::StringLiteral dialectName = "xevm";
  static LoadCacheControlAttr get(::mlir::MLIRContext *context, ::mlir::xevm::LoadCacheControl value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"load_cache_control"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xevm::LoadCacheControl getValue() const;
};
namespace detail {

struct StoreCacheControlAttrStorage;

} // namespace detail
class StoreCacheControlAttr : public ::mlir::Attribute::AttrBase<StoreCacheControlAttr, ::mlir::Attribute, detail::StoreCacheControlAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xevm.store_cache_control";
  static constexpr ::llvm::StringLiteral dialectName = "xevm";
  static StoreCacheControlAttr get(::mlir::MLIRContext *context, ::mlir::xevm::StoreCacheControl value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"store_cache_control"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xevm::StoreCacheControl getValue() const;
};
namespace detail {

struct MemScopeAttrStorage;

} // namespace detail
class MemScopeAttr : public ::mlir::Attribute::AttrBase<MemScopeAttr, ::mlir::Attribute, detail::MemScopeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xevm.mem_scope";
  static constexpr ::llvm::StringLiteral dialectName = "xevm";
  static MemScopeAttr get(::mlir::MLIRContext *context, ::mlir::xevm::MemScope value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mem_scope"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::xevm::MemScope getValue() const;
};
namespace detail {

struct AddrSpaceAttrStorage;

} // namespace detail
class AddrSpaceAttr : public ::mlir::Attribute::AttrBase<AddrSpaceAttr, ::mlir::Attribute, detail::AddrSpaceAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xevm.addr_space";
  static constexpr ::llvm::StringLiteral dialectName = "xevm";
  static AddrSpaceAttr get(::mlir::MLIRContext *context, mlir::xevm::AddrSpace value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"addr_space"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  mlir::xevm::AddrSpace getValue() const;
};
namespace detail {

struct MMAShapeAttrStorage;

} // namespace detail
class MMAShapeAttr : public ::mlir::Attribute::AttrBase<MMAShapeAttr, ::mlir::Attribute, detail::MMAShapeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xevm.mma_shape";
  static constexpr ::llvm::StringLiteral dialectName = "xevm";
  static MMAShapeAttr get(::mlir::MLIRContext *context, int m, int n, int k);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_shape"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getM() const;
  int getN() const;
  int getK() const;
};
namespace detail {

struct MMATypesAttrStorage;

} // namespace detail
class MMATypesAttr : public ::mlir::Attribute::AttrBase<MMATypesAttr, ::mlir::Attribute, detail::MMATypesAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xevm.mma_types";
  static constexpr ::llvm::StringLiteral dialectName = "xevm";
  static MMATypesAttr get(::mlir::MLIRContext *context, xevm::ElemType d, xevm::ElemType a, xevm::ElemType b, xevm::ElemType c);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"mma_types"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  xevm::ElemType getD() const;
  xevm::ElemType getA() const;
  xevm::ElemType getB() const;
  xevm::ElemType getC() const;
};
namespace detail {

struct XeVMTargetAttrStorage;

} // namespace detail
class XeVMTargetAttr : public ::mlir::Attribute::AttrBase<XeVMTargetAttr, ::mlir::Attribute, detail::XeVMTargetAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "xevm.target";
  static constexpr ::llvm::StringLiteral dialectName = "xevm";
  using Base::getChecked;
  static XeVMTargetAttr get(::mlir::MLIRContext *context, int optLevel = 2, ::llvm::StringRef triple = "spirv64-unknown-unknown", ::llvm::StringRef chip = "bmg", ::mlir::DictionaryAttr targetFlags = nullptr, ::mlir::ArrayAttr linkFiles = nullptr);
  static XeVMTargetAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int optLevel = 2, ::llvm::StringRef triple = "spirv64-unknown-unknown", ::llvm::StringRef chip = "bmg", ::mlir::DictionaryAttr targetFlags = nullptr, ::mlir::ArrayAttr linkFiles = nullptr);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int O, ::llvm::StringRef triple, ::llvm::StringRef chip, ::mlir::DictionaryAttr flags, ::mlir::ArrayAttr linkFiles);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, int O, ::llvm::StringRef triple, ::llvm::StringRef chip, ::mlir::DictionaryAttr flags, ::mlir::ArrayAttr linkFiles);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"target"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getO() const;
  ::llvm::StringRef getTriple() const;
  ::llvm::StringRef getChip() const;
  ::mlir::DictionaryAttr getFlags() const;
  ::mlir::ArrayAttr getLinkFiles() const;
};

} // namespace mlir::xevm
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xevm::LoadCacheControlAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xevm::StoreCacheControlAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xevm::MemScopeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xevm::AddrSpaceAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xevm::MMAShapeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xevm::MMATypesAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::xevm::XeVMTargetAttr)

#endif // GET_ATTRDEF_CLASSES

