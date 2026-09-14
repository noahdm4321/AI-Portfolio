/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::amdgpu {

/// Pair of base addresses that move data between LDS and global storage.
/// This type is opaque and it is used to represent a struct of two addresses.
///     One address is in LDS while the other is in global memory.
/// 
///     The value defined by this operation is only intended to be used by
///     amdgpu.tdm_make_descriptor.
class TDMBaseType;
/// Pair of base addresses that move data between LDS and global storage.
/// This type is opaque and it is used to represent a struct of two addresses.
///     One address is in LDS while the other is in global memory.
/// 
///     This operation is similar to amdgpu.tdm_make_base but intended to be
///     used in gather mode.
/// 
///     The value defined by this operation is only intended to be used by
///     amdgpu.tdm_make_gather_descriptor.
class TDMGatherBaseType;
/// Descriptors used in tensor store/load operations.
/// This type is opaque and corresponds to the two or four descriptor groups
///     used in tensor_load_to_lds or tensor_store_from_lds.
class TDMDescriptorType;
namespace detail {

struct TDMBaseTypeStorage;

} // namespace detail
class TDMBaseType : public ::mlir::Type::TypeBase<TDMBaseType, ::mlir::Type, detail::TDMBaseTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "amdgpu.tdm_base";
  static constexpr ::llvm::StringLiteral dialectName = "amdgpu";
  static TDMBaseType get(::mlir::MLIRContext *context, Type elementType);
  static TDMBaseType get(Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tdm_base"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getElementType() const;
};
namespace detail {

struct TDMGatherBaseTypeStorage;

} // namespace detail
class TDMGatherBaseType : public ::mlir::Type::TypeBase<TDMGatherBaseType, ::mlir::Type, detail::TDMGatherBaseTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "amdgpu.tdm_gather_base";
  static constexpr ::llvm::StringLiteral dialectName = "amdgpu";
  using Base::getChecked;
  static TDMGatherBaseType get(::mlir::MLIRContext *context, Type elementType, Type indexType);
  static TDMGatherBaseType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type elementType, Type indexType);
  static TDMGatherBaseType get(Type elementType, Type indexType);
  static TDMGatherBaseType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Type indexType);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Type indexType);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type elementType, Type indexType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tdm_gather_base"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getElementType() const;
  Type getIndexType() const;
};
class TDMDescriptorType : public ::mlir::Type::TypeBase<TDMDescriptorType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "amdgpu.tdm_descriptor";
  static constexpr ::llvm::StringLiteral dialectName = "amdgpu";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tdm_descriptor"};
  }

};

} // namespace mlir::amdgpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::TDMBaseType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::TDMGatherBaseType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::TDMDescriptorType)

#endif // GET_TYPEDEF_CLASSES

