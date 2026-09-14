/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvws {

class DescriptorLoadOpInterface;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// This interface implements setStage/getStage for aref ops
class ArefStageInterface;

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

namespace detail {
struct DescriptorLoadOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    int (*getTxCount)(const Concept *impl, ::mlir::Operation *);
    /// The base classes of this interface.
    const ::mlir::triton::DescriptorOpInterface::Concept *implDescriptorOpInterface = nullptr;

    void initializeInterfaceConcept(::mlir::detail::InterfaceMap &interfaceMap) {
      implDescriptorOpInterface = interfaceMap.lookup<::mlir::triton::DescriptorOpInterface>();
      assert(implDescriptorOpInterface && "`::mlir::triton::nvws::DescriptorLoadOpInterface` expected its base interface `::mlir::triton::DescriptorOpInterface` to be registered");
    }
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::nvws::DescriptorLoadOpInterface;
    Model() : Concept{getTxCount} {}

    static inline int getTxCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::nvws::DescriptorLoadOpInterface;
    FallbackModel() : Concept{getTxCount} {}

    static inline int getTxCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct DescriptorLoadOpInterfaceTrait;

} // namespace detail
class DescriptorLoadOpInterface : public ::mlir::OpInterface<DescriptorLoadOpInterface, detail::DescriptorLoadOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DescriptorLoadOpInterface, detail::DescriptorLoadOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DescriptorLoadOpInterfaceTrait<ConcreteOp> {};

  /// Get the transaction counts
  int getTxCount();
  //===----------------------------------------------------------------===//
  // Inherited from ::mlir::triton::DescriptorOpInterface
  //===----------------------------------------------------------------===//

  operator ::mlir::triton::DescriptorOpInterface () const {
    if (!*this) return nullptr;
    return ::mlir::triton::DescriptorOpInterface(*this, getImpl()->implDescriptorOpInterface);
  }


  /// Get the descriptor
  ::mlir::TypedValue<mlir::triton::TensorDescType> getDesc();
};

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

/// This interface implements setStage/getStage for aref ops
namespace detail {
struct ArefStageInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getStage)(const Concept *impl, ::mlir::Operation *);
    void (*setStage)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::nvws::ArefStageInterface;
    Model() : Concept{getStage, setStage} {}

    static inline ::mlir::Value getStage(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setStage(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value stage);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::nvws::ArefStageInterface;
    FallbackModel() : Concept{getStage, setStage} {}

    static inline ::mlir::Value getStage(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setStage(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value stage);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct ArefStageInterfaceTrait;

} // namespace detail
class ArefStageInterface : public ::mlir::OpInterface<ArefStageInterface, detail::ArefStageInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ArefStageInterface, detail::ArefStageInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ArefStageInterfaceTrait<ConcreteOp> {};

  /// Return aref stage
  ::mlir::Value getStage();

  /// Set aref stage
  void setStage(::mlir::Value stage);
};

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws::detail {

  template <typename ConcreteOp>
  struct DescriptorLoadOpInterfaceTrait : public ::mlir::OpInterface<DescriptorLoadOpInterface, detail::DescriptorLoadOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::nvws::detail
namespace mlir::triton::nvws::detail {

  template <typename ConcreteOp>
  struct ArefStageInterfaceTrait : public ::mlir::OpInterface<ArefStageInterface, detail::ArefStageInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::triton::nvws::detail
namespace mlir::triton::nvws {

template<typename ConcreteOp>
int detail::DescriptorLoadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTxCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTxCount();
}
template<typename ConcreteOp>
int detail::DescriptorLoadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTxCount(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getTxCount(tablegen_opaque_val);
}

} // namespace mlir::triton::nvws
namespace mlir::triton::nvws {

template<typename ConcreteOp>
::mlir::Value detail::ArefStageInterfaceInterfaceTraits::Model<ConcreteOp>::getStage(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStage();
}
template<typename ConcreteOp>
void detail::ArefStageInterfaceInterfaceTraits::Model<ConcreteOp>::setStage(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value stage) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setStage(stage);
}
template<typename ConcreteOp>
::mlir::Value detail::ArefStageInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getStage(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getStage(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::ArefStageInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setStage(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value stage) {
  return static_cast<const ConcreteOp *>(impl)->setStage(tablegen_opaque_val, stage);
}

} // namespace mlir::triton::nvws
