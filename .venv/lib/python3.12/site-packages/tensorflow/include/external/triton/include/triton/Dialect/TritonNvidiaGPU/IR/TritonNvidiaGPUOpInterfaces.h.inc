/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvidia_gpu {

/// This interface is implemented by MMAv5 dot and dot scaled ops.
class MMAv5OpInterface;

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu {

/// This interface is implemented by MMAv5 dot and dot scaled ops.
namespace detail {
struct MMAv5OpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> (*getA)(const Concept *impl, ::mlir::Operation *);
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> (*getB)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*useAccumulator)(const Concept *impl, ::mlir::Operation *);
    void (*setUseAccumulator)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
    ::mlir::ValueRange (*getCompletionBarriers)(const Concept *impl, ::mlir::Operation *);
    ::mlir::ValueRange (*getCompletionBarrierPreds)(const Concept *impl, ::mlir::Operation *);
    void (*addCompletionBarrier)(const Concept *impl, ::mlir::Operation *, ::mlir::Value, ::mlir::Value);
    ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> (*getAccumulator)(const Concept *impl, ::mlir::Operation *);
    void (*setAccumulator)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
    ::mlir::Value (*getPredicate)(const Concept *impl, ::mlir::Operation *);
    void (*setPredicate)(const Concept *impl, ::mlir::Operation *, ::mlir::Value);
    ::mlir::Value (*getAccDep)(const Concept *impl, ::mlir::Operation *);
    ::mlir::MutableOperandRange (*getAccDepMutable)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Value (*getToken)(const Concept *impl, ::mlir::Operation *);
    void (*setIsAsync)(const Concept *impl, ::mlir::Operation *, bool);
    bool (*isAsync)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::MMAv5OpInterface;
    Model() : Concept{getA, getB, useAccumulator, setUseAccumulator, getCompletionBarriers, getCompletionBarrierPreds, addCompletionBarrier, getAccumulator, setAccumulator, getPredicate, setPredicate, getAccDep, getAccDepMutable, getToken, setIsAsync, isAsync} {}

    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value useAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setUseAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value flag);
    static inline ::mlir::ValueRange getCompletionBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ValueRange getCompletionBarrierPreds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void addCompletionBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value barrier, ::mlir::Value pred);
    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value accum);
    static inline ::mlir::Value getPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred);
    static inline ::mlir::Value getAccDep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getAccDepMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setIsAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isAsync);
    static inline bool isAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::nvidia_gpu::MMAv5OpInterface;
    FallbackModel() : Concept{getA, getB, useAccumulator, setUseAccumulator, getCompletionBarriers, getCompletionBarrierPreds, addCompletionBarrier, getAccumulator, setAccumulator, getPredicate, setPredicate, getAccDep, getAccDepMutable, getToken, setIsAsync, isAsync} {}

    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value useAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setUseAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value flag);
    static inline ::mlir::ValueRange getCompletionBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::ValueRange getCompletionBarrierPreds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void addCompletionBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value barrier, ::mlir::Value pred);
    static inline ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value accum);
    static inline ::mlir::Value getPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred);
    static inline ::mlir::Value getAccDep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::MutableOperandRange getAccDepMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Value getToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void setIsAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isAsync);
    static inline bool isAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct MMAv5OpInterfaceTrait;

} // namespace detail
class MMAv5OpInterface : public ::mlir::OpInterface<MMAv5OpInterface, detail::MMAv5OpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<MMAv5OpInterface, detail::MMAv5OpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::MMAv5OpInterfaceTrait<ConcreteOp> {};

  /// Return the A operand.
  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getA();

  /// Return the B operand.
  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getB();

  /// Return the accumulator init flag.
  ::mlir::Value useAccumulator();

  /// Set the accumulator init flag.
  void setUseAccumulator(::mlir::Value flag);

  /// Return the completion barriers of this MMAv5 op.
  ::mlir::ValueRange getCompletionBarriers();

  /// Return the completion barrier predicates of this MMAv5 op.
  ::mlir::ValueRange getCompletionBarrierPreds();

  /// Associate a new completion barrier to this MMAv5 op.
  void addCompletionBarrier(::mlir::Value barrier, ::mlir::Value pred);

  /// Return the accumulator.
  ::mlir::TypedValue<::mlir::triton::gpu::MemDescType> getAccumulator();

  /// Set the accumulator.
  void setAccumulator(::mlir::Value accum);

  /// Return the predicate of this op.
  ::mlir::Value getPredicate();

  /// Set the predicate of this op.
  void setPredicate(::mlir::Value pred);

  /// Get the memory dependencies of the accumulator.
  ::mlir::Value getAccDep();

  /// Get the mutable memory dependencies of the accumulator.
  ::mlir::MutableOperandRange getAccDepMutable();

  /// Get the produced write dependency of the accumulator.
  ::mlir::Value getToken();

  /// Indicate that this MMA op executes asynchronously.
  void setIsAsync(bool isAsync);

  /// Return true if this MMA op executes asynchronously.
  bool isAsync();
};

} // namespace mlir::triton::nvidia_gpu
namespace mlir::triton::nvidia_gpu::detail {

  template <typename ConcreteOp>
  struct MMAv5OpInterfaceTrait : public ::mlir::OpInterface<MMAv5OpInterface, detail::MMAv5OpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      return ::mlir::triton::nvidia_gpu::impl::verifyMMAv5Op(op);
    }
  };

} // namespace mlir::triton::nvidia_gpu::detail
namespace mlir::triton::nvidia_gpu {

template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getA();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getB();
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::useAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).useAccumulator();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::setUseAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value flag) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setUseAccumulator(flag);
}
template<typename ConcreteOp>
::mlir::ValueRange detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getCompletionBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCompletionBarriers();
}
template<typename ConcreteOp>
::mlir::ValueRange detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getCompletionBarrierPreds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getCompletionBarrierPreds();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::addCompletionBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value barrier, ::mlir::Value pred) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).addCompletionBarrier(barrier, pred);
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccumulator();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::setAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value accum) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setAccumulator(accum);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getPredicate();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::setPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setPredicate(pred);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccDep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccDep();
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getAccDepMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAccDepMutable();
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::getToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getToken();
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::setIsAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isAsync) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setIsAsync(isAsync);
}
template<typename ConcreteOp>
bool detail::MMAv5OpInterfaceInterfaceTraits::Model<ConcreteOp>::isAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isAsync();
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getA(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getA(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getB(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getB(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::useAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->useAccumulator(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setUseAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value flag) {
  return static_cast<const ConcreteOp *>(impl)->setUseAccumulator(tablegen_opaque_val, flag);
}
template<typename ConcreteOp>
::mlir::ValueRange detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getCompletionBarriers(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCompletionBarriers(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::ValueRange detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getCompletionBarrierPreds(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getCompletionBarrierPreds(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::addCompletionBarrier(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value barrier, ::mlir::Value pred) {
  return static_cast<const ConcreteOp *>(impl)->addCompletionBarrier(tablegen_opaque_val, barrier, pred);
}
template<typename ConcreteOp>
::mlir::TypedValue<::mlir::triton::gpu::MemDescType> detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccumulator(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setAccumulator(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value accum) {
  return static_cast<const ConcreteOp *>(impl)->setAccumulator(tablegen_opaque_val, accum);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getPredicate(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setPredicate(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Value pred) {
  return static_cast<const ConcreteOp *>(impl)->setPredicate(tablegen_opaque_val, pred);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccDep(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccDep(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::MutableOperandRange detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAccDepMutable(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAccDepMutable(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Value detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getToken(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getToken(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setIsAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, bool isAsync) {
  return static_cast<const ConcreteOp *>(impl)->setIsAsync(tablegen_opaque_val, isAsync);
}
template<typename ConcreteOp>
bool detail::MMAv5OpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isAsync(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isAsync(tablegen_opaque_val);
}

} // namespace mlir::triton::nvidia_gpu
