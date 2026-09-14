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
namespace mlir::triton::nvws {

class TypeArrayAttr;
class IntArrayAttr;
/// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 5
class AsyncOpAttr;
namespace detail {

struct TypeArrayAttrStorage;

} // namespace detail
class TypeArrayAttr : public ::mlir::Attribute::AttrBase<TypeArrayAttr, ::mlir::Attribute, detail::TypeArrayAttrStorage> {
public:
  using Base::Base;
  auto begin() const { return getValue().begin(); }
  auto end() const { return getValue().end(); }
  bool empty() const { return getValue().empty(); }
  size_t size() const { return getValue().size(); }
  auto &front() const { return getValue().front(); }
  auto &back() const { return getValue().back(); }
  auto &operator[](size_t index) { return getValue()[index]; }
  operator ::llvm::ArrayRef<Type>() const { return getValue(); }
  static constexpr ::llvm::StringLiteral name = "nvws.type_array";
  static constexpr ::llvm::StringLiteral dialectName = "nvws";
  static TypeArrayAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<Type> value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"type_array"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<Type> getValue() const;
};
namespace detail {

struct IntArrayAttrStorage;

} // namespace detail
class IntArrayAttr : public ::mlir::Attribute::AttrBase<IntArrayAttr, ::mlir::Attribute, detail::IntArrayAttrStorage> {
public:
  using Base::Base;
  auto begin() const { return getValue().begin(); }
  auto end() const { return getValue().end(); }
  bool empty() const { return getValue().empty(); }
  size_t size() const { return getValue().size(); }
  auto &front() const { return getValue().front(); }
  auto &back() const { return getValue().back(); }
  auto &operator[](size_t index) { return getValue()[index]; }
  operator ::llvm::ArrayRef<int>() const { return getValue(); }
  static constexpr ::llvm::StringLiteral name = "nvws.int_array";
  static constexpr ::llvm::StringLiteral dialectName = "nvws";
  static IntArrayAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int> value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"int_array"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int> getValue() const;
};
namespace detail {

struct AsyncOpAttrStorage;

} // namespace detail
class AsyncOpAttr : public ::mlir::Attribute::AttrBase<AsyncOpAttr, ::mlir::Attribute, detail::AsyncOpAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "nvws.async_op";
  static constexpr ::llvm::StringLiteral dialectName = "nvws";
  static AsyncOpAttr get(::mlir::MLIRContext *context, ::mlir::triton::nvws::AsyncOp value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"async_op"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::triton::nvws::AsyncOp getValue() const;
};

} // namespace mlir::triton::nvws
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::TypeArrayAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::IntArrayAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::nvws::AsyncOpAttr)

#endif // GET_ATTRDEF_CLASSES

