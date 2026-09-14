/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::gpu {

/// Common trait for all TTGIR layouts.
class LayoutEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Common trait describing shared memory.
class SharedEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// The Distributed encoding describes the layout L with the 4-level compute hierarchy on GPU.
/// It is abstracted from the top to the bottom as CTAs Per CGA->Warps Per CTA->Threads Per Warp->Values Per Thread.
/// 
/// For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.
/// For example, for a shape/order pair defines a distribution layout
/// shape = [4, 4]
/// order = [0, 1] // The fastest-changing axis first
/// ->
/// layout = [0  4  8  12]
///          [1  5  9  13]
///          [2  6  10 14]
///          [3  7  11 15]
/// 
/// For the Threads Per Warp and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.
/// 
/// If the layout does not completely cover the tensor, we tile it until we cover the entire tensor.
/// We call each individual tile "rep".
class DistributedEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

class MmaEncodingTrait;

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Common trait for all TTGIR layouts.
namespace detail {
struct LayoutEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    CGAEncodingAttr (*getCGALayout)(const Concept *impl, ::mlir::Attribute );
    unsigned (*getRank)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::LayoutEncodingTrait;
    Model() : Concept{getCGALayout, getRank} {}

    static inline CGAEncodingAttr getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::LayoutEncodingTrait;
    FallbackModel() : Concept{getCGALayout, getRank} {}

    static inline CGAEncodingAttr getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    unsigned getRank(::mlir::Attribute tablegen_opaque_val) const;
  };
};
template <typename ConcreteAttr>
struct LayoutEncodingTraitTrait;

} // namespace detail
class LayoutEncodingTrait : public ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::LayoutEncodingTraitTrait<ConcreteAttr> {};

  /// Get the CGA layout backing this encoding.
  CGAEncodingAttr getCGALayout() const;

  /// Get the rank of the layout.
  unsigned getRank() const;
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// Common trait describing shared memory.
namespace detail {
struct SharedEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    int32_t (*getAlignment)(const Concept *impl, ::mlir::Attribute );
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::SharedEncodingTrait;
    Model() : Concept{getAlignment} {}

    static inline int32_t getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::SharedEncodingTrait;
    FallbackModel() : Concept{getAlignment} {}

    static inline int32_t getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
    int32_t getAlignment(::mlir::Attribute tablegen_opaque_val) const;
  };
};
template <typename ConcreteAttr>
struct SharedEncodingTraitTrait;

} // namespace detail
class SharedEncodingTrait : public ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::SharedEncodingTraitTrait<ConcreteAttr> {};

  /// Return the default alignment for the layout.
  int32_t getAlignment() const;
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

/// The Distributed encoding describes the layout L with the 4-level compute hierarchy on GPU.
/// It is abstracted from the top to the bottom as CTAs Per CGA->Warps Per CTA->Threads Per Warp->Values Per Thread.
/// 
/// For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.
/// For example, for a shape/order pair defines a distribution layout
/// shape = [4, 4]
/// order = [0, 1] // The fastest-changing axis first
/// ->
/// layout = [0  4  8  12]
///          [1  5  9  13]
///          [2  6  10 14]
///          [3  7  11 15]
/// 
/// For the Threads Per Warp and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.
/// 
/// If the layout does not completely cover the tensor, we tile it until we cover the entire tensor.
/// We call each individual tile "rep".
namespace detail {
struct DistributedEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<unsigned> (*getRepOrder)(const Concept *impl, ::mlir::Attribute );
    unsigned (*getTotalElemsPerThread)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
    SmallVector<unsigned> (*getElemsPerThread)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
    LinearLayout (*toLinearLayout)(const Concept *impl, ::mlir::Attribute , ArrayRef<int64_t>);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::DistributedEncodingTrait;
    Model() : Concept{getRepOrder, getTotalElemsPerThread, getElemsPerThread, toLinearLayout} {}

    static inline SmallVector<unsigned> getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline SmallVector<unsigned> getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::DistributedEncodingTrait;
    FallbackModel() : Concept{getRepOrder, getTotalElemsPerThread, getElemsPerThread, toLinearLayout} {}

    static inline SmallVector<unsigned> getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val);
    static inline unsigned getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline SmallVector<unsigned> getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
    static inline LinearLayout toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct DistributedEncodingTraitTrait;

} // namespace detail
class DistributedEncodingTrait : public ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::DistributedEncodingTraitTrait<ConcreteAttr> {};

  /// Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first
  SmallVector<unsigned> getRepOrder() const;

  /// Return total element size per thread.
  unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;

  /// Return element size per thread in each dimension.
  SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const;

  /// Convert to LinearLayout.
  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

namespace detail {
struct MmaEncodingTraitInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<unsigned> (*getRepOrderForOperand)(const Concept *impl, ::mlir::Attribute , int);
  };
  template<typename ConcreteAttr>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::MmaEncodingTrait;
    Model() : Concept{getRepOrderForOperand} {}

    static inline SmallVector<unsigned> getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx);
  };
  template<typename ConcreteAttr>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::gpu::MmaEncodingTrait;
    FallbackModel() : Concept{getRepOrderForOperand} {}

    static inline SmallVector<unsigned> getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx);
  };
  template<typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteAttr;
  };
};
template <typename ConcreteAttr>
struct MmaEncodingTraitTrait;

} // namespace detail
class MmaEncodingTrait : public ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits> {
public:
  using ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits>::AttributeInterface;
  template <typename ConcreteAttr>
  struct Trait : public detail::MmaEncodingTraitTrait<ConcreteAttr> {};

  /// Get the order of reps (tiles of this layout that tile the whole tensor). The fastest-changing axis first
  SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
};

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct LayoutEncodingTraitTrait : public ::mlir::AttributeInterface<LayoutEncodingTrait, detail::LayoutEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {

    /// Get the rank of the layout.
    unsigned getRank() const {
      return (*static_cast<const ConcreteAttr *>(this)).getCGALayout().getRank();
    }
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct SharedEncodingTraitTrait : public ::mlir::AttributeInterface<SharedEncodingTrait, detail::SharedEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {

    /// Return the default alignment for the layout.
    int32_t getAlignment() const {
      return 16;
    }
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct DistributedEncodingTraitTrait : public ::mlir::AttributeInterface<DistributedEncodingTrait, detail::DistributedEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu::detail {

  template <typename ConcreteAttr>
  struct MmaEncodingTraitTrait : public ::mlir::AttributeInterface<MmaEncodingTrait, detail::MmaEncodingTraitInterfaceTraits>::Trait<ConcreteAttr> {
  };

} // namespace mlir::triton::gpu::detail
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
CGAEncodingAttr detail::LayoutEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCGALayout();
}
template<typename ConcreteAttr>
unsigned detail::LayoutEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRank();
}
template<typename ConcreteAttr>
CGAEncodingAttr detail::LayoutEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getCGALayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getCGALayout(tablegen_opaque_val);
}
template<typename ConcreteAttr>
unsigned detail::LayoutEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRank(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRank(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteAttr>
unsigned detail::LayoutEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getRank(::mlir::Attribute tablegen_opaque_val) const {
return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getCGALayout().getRank();
}

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
int32_t detail::SharedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getAlignment();
}
template<typename ConcreteAttr>
int32_t detail::SharedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getAlignment(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getAlignment(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteAttr>
int32_t detail::SharedEncodingTraitInterfaceTraits::ExternalModel<ConcreteModel, ConcreteAttr>::getAlignment(::mlir::Attribute tablegen_opaque_val) const {
return 16;
}

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRepOrder();
}
template<typename ConcreteAttr>
unsigned detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return toLinearEncoding((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)), shape).getTotalElemsPerThread(shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return toLinearEncoding((::llvm::cast<ConcreteAttr>(tablegen_opaque_val)), shape).getElemsPerThread(shape);
}
template<typename ConcreteAttr>
LinearLayout detail::DistributedEncodingTraitInterfaceTraits::Model<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).toLinearLayout(shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRepOrder(const Concept *impl, ::mlir::Attribute tablegen_opaque_val) {
  return static_cast<const ConcreteAttr *>(impl)->getRepOrder(tablegen_opaque_val);
}
template<typename ConcreteAttr>
unsigned detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getTotalElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->getTotalElemsPerThread(tablegen_opaque_val, shape);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getElemsPerThread(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->getElemsPerThread(tablegen_opaque_val, shape);
}
template<typename ConcreteAttr>
LinearLayout detail::DistributedEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::toLinearLayout(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, ArrayRef<int64_t> shape) {
  return static_cast<const ConcreteAttr *>(impl)->toLinearLayout(tablegen_opaque_val, shape);
}

} // namespace mlir::triton::gpu
namespace mlir::triton::gpu {

template<typename ConcreteAttr>
SmallVector<unsigned> detail::MmaEncodingTraitInterfaceTraits::Model<ConcreteAttr>::getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx) {
  return (::llvm::cast<ConcreteAttr>(tablegen_opaque_val)).getRepOrderForOperand(opIdx);
}
template<typename ConcreteAttr>
SmallVector<unsigned> detail::MmaEncodingTraitInterfaceTraits::FallbackModel<ConcreteAttr>::getRepOrderForOperand(const Concept *impl, ::mlir::Attribute tablegen_opaque_val, int opIdx) {
  return static_cast<const ConcreteAttr *>(impl)->getRepOrderForOperand(tablegen_opaque_val, opIdx);
}

} // namespace mlir::triton::gpu
