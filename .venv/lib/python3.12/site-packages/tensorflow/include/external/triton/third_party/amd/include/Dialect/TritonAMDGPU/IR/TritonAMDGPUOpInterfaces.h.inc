/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::amdgpu {

/// This interface is implemented by buffer load/store operations.
///     It provides methods to access common properties such base pointer, offset, mask and others.
class BufferOpInterface;

} // namespace mlir::triton::amdgpu
namespace mlir::triton::amdgpu {

/// This interface is implemented by buffer load/store operations.
///     It provides methods to access common properties such base pointer, offset, mask and others.
namespace detail {
struct BufferOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::TypedValue<::mlir::triton::PointerType> (*getPtr)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperand &(*getPtrMutable)(const Concept *impl, ::mlir::Operation *);
    ::mlir::TypedValue<::mlir::TensorType> (*getOffsets)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperand &(*getOffsetsMutable)(const Concept *impl, ::mlir::Operation *);
    ::mlir::TypedValue<::mlir::IntegerType> (*getStride)(const Concept *impl, ::mlir::Operation *);
    ::mlir::MutableOperandRange (*getStrideMutable)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::amdgpu::BufferOpInterface;
    Model() : Concept{getPtr, getPtrMutable, getOffsets, getOffsetsMutable, getStride, getStrideMutable} {}

    static inline ::mlir::TypedValue<::mlir::triton::PointerType> getPtr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand &getPtrMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::TensorType> getOffsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand &getOffsetsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::IntegerType> getStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getStrideMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::amdgpu::BufferOpInterface;
    FallbackModel() : Concept{getPtr, getPtrMutable, getOffsets, getOffsetsMutable, getStride, getStrideMutable} {}

    static inline ::mlir::TypedValue<::mlir::triton::PointerType> getPtr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand &getPtrMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::TensorType> getOffsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand &getOffsetsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::IntegerType> getStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getStrideMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct BufferOpInterfaceTrait;

} // namespace detail
class BufferOpInterface : public ::mlir::OpInterface<BufferOpInterface, detail::BufferOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<BufferOpInterface, detail::BufferOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::BufferOpInterfaceTrait<ConcreteOp> {};

  /// Get operation base ptr.
  ::mlir::TypedValue<::mlir::triton::PointerType> getPtr();

  /// Get mutable operation base ptr.
  ::mlir::OpOperand &getPtrMutable();

  /// Get operation offset tensor.
  ::mlir::TypedValue<::mlir::TensorType> getOffsets();

  /// Get mutable operation offset tensor.
  ::mlir::OpOperand &getOffsetsMutable();

  /// Get operation stride.
  ::mlir::TypedValue<::mlir::IntegerType> getStride();

  /// Get mutable operation stride.
  ::mlir::MutableOperandRange getStrideMutable();
};

} // namespace mlir::triton::amdgpu
namespace mlir::triton::amdgpu::detail {

  template <typename ConcreteOp>
  struct BufferOpInterfaceTrait : public ::mlir::OpInterface<BufferOpInterface, detail::BufferOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::amdgpu::detail
namespace mlir::triton::amdgpu {

template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::PointerType> detail::BufferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPtr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPtr();
}
template<typename ConcreteOp>
::mlir::OpOperand &detail::BufferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getPtrMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPtrMutable();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::TensorType> detail::BufferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOffsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOffsets();
}
template<typename ConcreteOp>
::mlir::OpOperand &detail::BufferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOffsetsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOffsetsMutable();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::IntegerType> detail::BufferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStride();
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::BufferOpInterfaceInterfaceTraits::Model<ConcreteOp>::getStrideMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStrideMutable();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::PointerType> detail::BufferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPtr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPtr(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperand &detail::BufferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPtrMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPtrMutable(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::TensorType> detail::BufferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOffsets(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOffsets(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperand &detail::BufferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOffsetsMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOffsetsMutable(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::IntegerType> detail::BufferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStride(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getStride(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::BufferOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStrideMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getStrideMutable(tablegen_opaque_val);
}

} // namespace mlir::triton::amdgpu
