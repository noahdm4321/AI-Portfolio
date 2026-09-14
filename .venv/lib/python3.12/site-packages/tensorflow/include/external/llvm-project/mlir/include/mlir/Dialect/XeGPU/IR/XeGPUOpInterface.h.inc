/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::xegpu {

/// An attribute interface for accessing anchor layout information.
///     This interface provides a method to set and retrieve the anchor layout
///     from operations that implement it.
class AnchorLayoutInterface;

} // namespace mlir::xegpu
namespace mlir::xegpu {

/// An attribute interface for accessing anchor layout information.
///     This interface provides a method to set and retrieve the anchor layout
///     from operations that implement it.
namespace detail {
struct AnchorLayoutInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    xegpu::DistributeLayoutAttr (*getAnchorLayout)(const Concept *impl, ::mlir::Operation *);
    void (*setAnchorLayout)(const Concept *impl, ::mlir::Operation *, xegpu::DistributeLayoutAttr);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::xegpu::AnchorLayoutInterface;
    Model() : Concept{getAnchorLayout, setAnchorLayout} {}

    static inline xegpu::DistributeLayoutAttr getAnchorLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAnchorLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, xegpu::DistributeLayoutAttr layout);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::xegpu::AnchorLayoutInterface;
    FallbackModel() : Concept{getAnchorLayout, setAnchorLayout} {}

    static inline xegpu::DistributeLayoutAttr getAnchorLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAnchorLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, xegpu::DistributeLayoutAttr layout);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct AnchorLayoutInterfaceTrait;

} // namespace detail
class AnchorLayoutInterface : public ::mlir::OpInterface<AnchorLayoutInterface, detail::AnchorLayoutInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AnchorLayoutInterface, detail::AnchorLayoutInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AnchorLayoutInterfaceTrait<ConcreteOp> {};

  /// Get the anchor layout attribute.
  xegpu::DistributeLayoutAttr getAnchorLayout();

  /// Set the anchor layout attribute.
  void setAnchorLayout(xegpu::DistributeLayoutAttr layout);
};

} // namespace mlir::xegpu
namespace mlir::xegpu::detail {

  template <typename ConcreteOp>
  struct AnchorLayoutInterfaceTrait : public ::mlir::OpInterface<AnchorLayoutInterface, detail::AnchorLayoutInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::xegpu::detail
namespace mlir::xegpu {

template<typename ConcreteOp>
xegpu::DistributeLayoutAttr detail::AnchorLayoutInterfaceInterfaceTraits::Model<ConcreteOp>::getAnchorLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAnchorLayout();
}
template<typename ConcreteOp>
void detail::AnchorLayoutInterfaceInterfaceTraits::Model<ConcreteOp>::setAnchorLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, xegpu::DistributeLayoutAttr layout) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setAnchorLayout(layout);
}
template<typename ConcreteOp>
xegpu::DistributeLayoutAttr detail::AnchorLayoutInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAnchorLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAnchorLayout(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::AnchorLayoutInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setAnchorLayout(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, xegpu::DistributeLayoutAttr layout) {
  return static_cast<const ConcreteOp *>(impl)->setAnchorLayout(tablegen_opaque_val, layout);
}

} // namespace mlir::xegpu
