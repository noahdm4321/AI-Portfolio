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
namespace mlir::triton::gluon {

/// An encoding that is inferred from neighboring ops in the graph.
class AutoEncodingAttr;
/// An encoding that is optimized for load/store performance.
class CoalescedEncodingAttr;
class AutoEncodingAttr : public ::mlir::Attribute::AttrBase<AutoEncodingAttr, ::mlir::Attribute, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gluon.auto_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "gluon";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"auto_encoding"};
  }

};
class CoalescedEncodingAttr : public ::mlir::Attribute::AttrBase<CoalescedEncodingAttr, ::mlir::Attribute, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "gluon.coalesced_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "gluon";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"coalesced_encoding"};
  }

};

} // namespace mlir::triton::gluon
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gluon::AutoEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gluon::CoalescedEncodingAttr)

#endif // GET_ATTRDEF_CLASSES

