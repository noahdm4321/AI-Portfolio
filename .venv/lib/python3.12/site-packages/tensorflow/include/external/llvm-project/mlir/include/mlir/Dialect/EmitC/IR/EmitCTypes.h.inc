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
namespace mlir::emitc {

/// EmitC array type
/// An array data type.
/// 
///     Example:
/// 
///     ```mlir
///     // Array emitted as `int32_t[10]`
///     !emitc.array<10xi32>
///     // Array emitted as `float[10][20]`
///     !emitc.array<10x20xf32>
///     ```
class ArrayType;
/// EmitC lvalue type
/// Values of this type can be assigned to and their address can be taken.
class LValueType;
/// EmitC opaque type
/// An opaque data type of which the value gets emitted as is.
/// 
///     Example:
/// 
///     ```mlir
///     !emitc.opaque<"int">
///     !emitc.opaque<"mytype">
///     !emitc.opaque<"std::vector<std::string>">
///     ```
class OpaqueType;
/// EmitC pointer type
/// A pointer data type.
/// 
///     Example:
/// 
///     ```mlir
///     // Pointer emitted as `int32_t*`
///     !emitc.ptr<i32>
///     // Pointer emitted as `float*`
///     !emitc.ptr<f32>
///     // Pointer emitted as `int*`
///     !emitc.ptr<!emitc.opaque<"int">>
///     ```
class PointerType;
/// EmitC signed size type
/// Data type representing all values of `emitc.size_t`, plus -1.
///     It corresponds to `ssize_t` found in `<sys/types.h>`.
///     
///     Use of this type causes the code to be non-C99 compliant.
class SignedSizeTType;
/// EmitC signed pointer diff type
/// Signed data type as wide as platform-specific pointer types.
///     In particular, it is as wide as `emitc.size_t`.
///     It corresponds to `ptrdiff_t` found in `<stddef.h>`.
class PtrDiffTType;
/// EmitC unsigned size type
/// Unsigned data type as wide as platform-specific pointer types.
///     It corresponds to `size_t` found in `<stddef.h>`.
class SizeTType;
namespace detail {

struct ArrayTypeStorage;

} // namespace detail
class ArrayType : public ::mlir::Type::TypeBase<ArrayType, ::mlir::Type, detail::ArrayTypeStorage, ::mlir::ShapedType::Trait> {
public:
  using Base::Base;
  /// Returns if this type is ranked (always true).
  bool hasRank() const { return true; }

  /// Clone this array type with the given shape and element type. If the
  /// provided shape is `std::nullopt`, the current shape of the type is used.
  ArrayType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                      Type elementType) const;

  static bool isValidElementType(Type type) {
    return emitc::isSupportedFloatType(type) ||
       emitc::isIntegerIndexOrOpaqueType(type) ||
       llvm::isa<PointerType>(type);
  }
  static constexpr ::llvm::StringLiteral name = "emitc.array";
  static constexpr ::llvm::StringLiteral dialectName = "emitc";
  using Base::getChecked;
  static ArrayType get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType);
  static ArrayType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, Type elementType);
  static ArrayType get(ArrayRef<int64_t> shape, Type elementType);
  static ArrayType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape, Type elementType);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<int64_t> shape, Type elementType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"array"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  Type getElementType() const;
};
namespace detail {

struct LValueTypeStorage;

} // namespace detail
class LValueType : public ::mlir::Type::TypeBase<LValueType, ::mlir::Type, detail::LValueTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "emitc.lvalue";
  static constexpr ::llvm::StringLiteral dialectName = "emitc";
  using Base::getChecked;
  static LValueType get(::mlir::MLIRContext *context, Type valueType);
  static LValueType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type valueType);
  static LValueType get(Type valueType);
  static LValueType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type valueType);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type valueType);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type valueType);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"lvalue"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getValueType() const;
};
namespace detail {

struct OpaqueTypeStorage;

} // namespace detail
class OpaqueType : public ::mlir::Type::TypeBase<OpaqueType, ::mlir::Type, detail::OpaqueTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "emitc.opaque";
  static constexpr ::llvm::StringLiteral dialectName = "emitc";
  using Base::getChecked;
  static OpaqueType get(::mlir::MLIRContext *context, ::llvm::StringRef value);
  static OpaqueType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::StringRef value);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::StringRef value);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::StringRef value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"opaque"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getValue() const;
};
namespace detail {

struct PointerTypeStorage;

} // namespace detail
class PointerType : public ::mlir::Type::TypeBase<PointerType, ::mlir::Type, detail::PointerTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "emitc.ptr";
  static constexpr ::llvm::StringLiteral dialectName = "emitc";
  using Base::getChecked;
  static PointerType get(::mlir::MLIRContext *context, Type pointee);
  static PointerType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, Type pointee);
  static PointerType get(Type pointee);
  static PointerType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type pointee);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type pointee);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type pointee);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ptr"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getPointee() const;
};
class SignedSizeTType : public ::mlir::Type::TypeBase<SignedSizeTType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "emitc.ssize_t";
  static constexpr ::llvm::StringLiteral dialectName = "emitc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ssize_t"};
  }

};
class PtrDiffTType : public ::mlir::Type::TypeBase<PtrDiffTType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "emitc.ptrdiff_t";
  static constexpr ::llvm::StringLiteral dialectName = "emitc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ptrdiff_t"};
  }

};
class SizeTType : public ::mlir::Type::TypeBase<SizeTType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "emitc.size_t";
  static constexpr ::llvm::StringLiteral dialectName = "emitc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"size_t"};
  }

};

} // namespace mlir::emitc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::ArrayType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::LValueType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::OpaqueType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::PointerType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::SignedSizeTType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::PtrDiffTType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::emitc::SizeTType)

#endif // GET_TYPEDEF_CLASSES

