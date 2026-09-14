/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::emitc {

/// Interface to mark operations that can be part of the CExpression.
class CExpressionInterface;

} // namespace mlir::emitc
namespace mlir::emitc {

/// Interface to mark operations that can be part of the CExpression.
namespace detail {
struct CExpressionInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*hasSideEffects)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::emitc::CExpressionInterface;
    Model() : Concept{hasSideEffects} {}

    static inline bool hasSideEffects(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::emitc::CExpressionInterface;
    FallbackModel() : Concept{hasSideEffects} {}

    static inline bool hasSideEffects(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    bool hasSideEffects(::mlir::Operation *tablegen_opaque_val) const;
  };
};
template <typename ConcreteOp>
struct CExpressionInterfaceTrait;

} // namespace detail
class CExpressionInterface : public ::mlir::OpInterface<CExpressionInterface, detail::CExpressionInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CExpressionInterface, detail::CExpressionInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::CExpressionInterfaceTrait<ConcreteOp> {};

  /// Check whether operation has side effects that may affect the expression
  /// evaluation.
  /// 
  /// By default operation is marked as having side effects.
  /// 
  /// ```c++
  /// class ConcreteOp ... {
  /// public:
  ///   bool hasSideEffects() {
  ///     // That way we can override the default implementation.
  ///     return false;
  ///   }
  /// };
  /// ```
  bool hasSideEffects();
};

} // namespace mlir::emitc
namespace mlir::emitc::detail {

  template <typename ConcreteOp>
  struct CExpressionInterfaceTrait : public ::mlir::OpInterface<CExpressionInterface, detail::CExpressionInterfaceInterfaceTraits>::Trait<ConcreteOp> {

    /// Check whether operation has side effects that may affect the expression
    /// evaluation.
    /// 
    /// By default operation is marked as having side effects.
    /// 
    /// ```c++
    /// class ConcreteOp ... {
    /// public:
    ///   bool hasSideEffects() {
    ///     // That way we can override the default implementation.
    ///     return false;
    ///   }
    /// };
    /// ```
    bool hasSideEffects() {
      return true;
    }
  };

} // namespace mlir::emitc::detail
namespace mlir::emitc {

template<typename ConcreteOp>
bool detail::CExpressionInterfaceInterfaceTraits::Model<ConcreteOp>::hasSideEffects(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasSideEffects();
}
template<typename ConcreteOp>
bool detail::CExpressionInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasSideEffects(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasSideEffects(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::CExpressionInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasSideEffects(::mlir::Operation *tablegen_opaque_val) const {
return true;
}

} // namespace mlir::emitc
