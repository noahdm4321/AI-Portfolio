/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::acc {

/// An interface for compute and loop construct operations.
class ComputeRegionOpInterface;

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that access a partial entity such as
///     field or array element access.
class PartialEntityAccessOpInterface;

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that compute the address of a global variable
///     or symbol.
class AddressOfGlobalOpInterface;

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that define global variables. This interface
///     provides a uniform way to query properties of global variables across
///     different dialects.
class GlobalVariableOpInterface;

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that indirectly access global symbols.
///     This interface provides a way to query which global symbols are referenced
///     by an operation, which is useful for tracking dependencies and performing
///     analysis on global variable usage.
/// 
///     The symbolTable parameter is optional. If null, implementations will look up
///     their own symbol table. This allows callers to pass a pre-existing symbol
///     table for efficiency when querying multiple operations.
class IndirectGlobalAccessOpInterface;

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that are candidates for rematerialization
///     during outlining. These operations produce synthetic types or values
///     that cannot be passed as arguments to outlined regions and must be
///     rematerialized inside the region instead.
/// 
///     Operations implementing this interface are expected to be memory effect
///     free. Their results are typically used for type construction or providing
///     metadata (such as shape information for arrays).
class OutlineRematerializationOpInterface;

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations whose regions are targets for offloading
///     and outlining. Operations implementing this interface indicate that
///     their regions will be extracted and compiled separately (e.g., as
///     device kernels or outlined functions).
class OffloadRegionOpInterface;

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for compute and loop construct operations.
namespace detail {
struct ComputeRegionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Block*(*getAllocaBlock)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::ComputeRegionOpInterface;
    Model() : Concept{getAllocaBlock} {}

    static inline ::mlir::Block*getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::ComputeRegionOpInterface;
    FallbackModel() : Concept{getAllocaBlock} {}

    static inline ::mlir::Block*getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct ComputeRegionOpInterfaceTrait;

} // namespace detail
class ComputeRegionOpInterface : public ::mlir::OpInterface<ComputeRegionOpInterface, detail::ComputeRegionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ComputeRegionOpInterface, detail::ComputeRegionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ComputeRegionOpInterfaceTrait<ConcreteOp> {};

  /// Get alloca block
  ::mlir::Block*getAllocaBlock();
};

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that access a partial entity such as
///     field or array element access.
namespace detail {
struct PartialEntityAccessOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Value (*getBaseEntity)(const Concept *impl, ::mlir::Operation *);
    bool (*isCompleteView)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::PartialEntityAccessOpInterface;
    Model() : Concept{getBaseEntity, isCompleteView} {}

    static inline ::mlir::Value getBaseEntity(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isCompleteView(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::PartialEntityAccessOpInterface;
    FallbackModel() : Concept{getBaseEntity, isCompleteView} {}

    static inline ::mlir::Value getBaseEntity(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isCompleteView(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct PartialEntityAccessOpInterfaceTrait;

} // namespace detail
class PartialEntityAccessOpInterface : public ::mlir::OpInterface<PartialEntityAccessOpInterface, detail::PartialEntityAccessOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PartialEntityAccessOpInterface, detail::PartialEntityAccessOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PartialEntityAccessOpInterfaceTrait<ConcreteOp> {};

  /// Get the base entity being accessed
  ::mlir::Value getBaseEntity();

  /// Check if this is a complete view of the entity
  bool isCompleteView();
};

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that compute the address of a global variable
///     or symbol.
namespace detail {
struct AddressOfGlobalOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::SymbolRefAttr (*getSymbol)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::AddressOfGlobalOpInterface;
    Model() : Concept{getSymbol} {}

    static inline ::mlir::SymbolRefAttr getSymbol(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::AddressOfGlobalOpInterface;
    FallbackModel() : Concept{getSymbol} {}

    static inline ::mlir::SymbolRefAttr getSymbol(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct AddressOfGlobalOpInterfaceTrait;

} // namespace detail
class AddressOfGlobalOpInterface : public ::mlir::OpInterface<AddressOfGlobalOpInterface, detail::AddressOfGlobalOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AddressOfGlobalOpInterface, detail::AddressOfGlobalOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AddressOfGlobalOpInterfaceTrait<ConcreteOp> {};

  /// Get the symbol reference to the global
  ::mlir::SymbolRefAttr getSymbol();
};

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that define global variables. This interface
///     provides a uniform way to query properties of global variables across
///     different dialects.
namespace detail {
struct GlobalVariableOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    bool (*isConstant)(const Concept *impl, ::mlir::Operation *);
    ::mlir::Region*(*getInitRegion)(const Concept *impl, ::mlir::Operation *);
    bool (*isDeviceData)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::GlobalVariableOpInterface;
    Model() : Concept{isConstant, getInitRegion, isDeviceData} {}

    static inline bool isConstant(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Region*getInitRegion(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isDeviceData(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::GlobalVariableOpInterface;
    FallbackModel() : Concept{isConstant, getInitRegion, isDeviceData} {}

    static inline bool isConstant(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::Region*getInitRegion(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isDeviceData(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct GlobalVariableOpInterfaceTrait;

} // namespace detail
class GlobalVariableOpInterface : public ::mlir::OpInterface<GlobalVariableOpInterface, detail::GlobalVariableOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<GlobalVariableOpInterface, detail::GlobalVariableOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::GlobalVariableOpInterfaceTrait<ConcreteOp> {};

  /// Check if the global variable is constant
  bool isConstant();

  /// Get the initialization region (returns nullptr if none)
  ::mlir::Region*getInitRegion();

  /// Check if the global variable is device data
  bool isDeviceData();
};

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that indirectly access global symbols.
///     This interface provides a way to query which global symbols are referenced
///     by an operation, which is useful for tracking dependencies and performing
///     analysis on global variable usage.
/// 
///     The symbolTable parameter is optional. If null, implementations will look up
///     their own symbol table. This allows callers to pass a pre-existing symbol
///     table for efficiency when querying multiple operations.
namespace detail {
struct IndirectGlobalAccessOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    void (*getReferencedSymbols)(const Concept *impl, ::mlir::Operation *, ::llvm::SmallVectorImpl<::mlir::SymbolRefAttr>&, ::mlir::SymbolTable *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::IndirectGlobalAccessOpInterface;
    Model() : Concept{getReferencedSymbols} {}

    static inline void getReferencedSymbols(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::SmallVectorImpl<::mlir::SymbolRefAttr>& symbols, ::mlir::SymbolTable * symbolTable);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::IndirectGlobalAccessOpInterface;
    FallbackModel() : Concept{getReferencedSymbols} {}

    static inline void getReferencedSymbols(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::SmallVectorImpl<::mlir::SymbolRefAttr>& symbols, ::mlir::SymbolTable * symbolTable);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct IndirectGlobalAccessOpInterfaceTrait;

} // namespace detail
class IndirectGlobalAccessOpInterface : public ::mlir::OpInterface<IndirectGlobalAccessOpInterface, detail::IndirectGlobalAccessOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<IndirectGlobalAccessOpInterface, detail::IndirectGlobalAccessOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::IndirectGlobalAccessOpInterfaceTrait<ConcreteOp> {};

  /// Get the symbols referenced by this operation
  void getReferencedSymbols(::llvm::SmallVectorImpl<::mlir::SymbolRefAttr>& symbols, ::mlir::SymbolTable * symbolTable);
};

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations that are candidates for rematerialization
///     during outlining. These operations produce synthetic types or values
///     that cannot be passed as arguments to outlined regions and must be
///     rematerialized inside the region instead.
/// 
///     Operations implementing this interface are expected to be memory effect
///     free. Their results are typically used for type construction or providing
///     metadata (such as shape information for arrays).
namespace detail {
struct OutlineRematerializationOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::OutlineRematerializationOpInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::OutlineRematerializationOpInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct OutlineRematerializationOpInterfaceTrait;

} // namespace detail
class OutlineRematerializationOpInterface : public ::mlir::OpInterface<OutlineRematerializationOpInterface, detail::OutlineRematerializationOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OutlineRematerializationOpInterface, detail::OutlineRematerializationOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OutlineRematerializationOpInterfaceTrait<ConcreteOp> {};
};

} // namespace mlir::acc
namespace mlir::acc {

/// An interface for operations whose regions are targets for offloading
///     and outlining. Operations implementing this interface indicate that
///     their regions will be extracted and compiled separately (e.g., as
///     device kernels or outlined functions).
namespace detail {
struct OffloadRegionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::OffloadRegionOpInterface;
    Model() : Concept{} {}

  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::OffloadRegionOpInterface;
    FallbackModel() : Concept{} {}

  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};
template <typename ConcreteOp>
struct OffloadRegionOpInterfaceTrait;

} // namespace detail
class OffloadRegionOpInterface : public ::mlir::OpInterface<OffloadRegionOpInterface, detail::OffloadRegionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OffloadRegionOpInterface, detail::OffloadRegionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OffloadRegionOpInterfaceTrait<ConcreteOp> {};
};

} // namespace mlir::acc
namespace mlir::acc::detail {

  template <typename ConcreteOp>
  struct ComputeRegionOpInterfaceTrait : public ::mlir::OpInterface<ComputeRegionOpInterface, detail::ComputeRegionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::acc::detail
namespace mlir::acc::detail {

  template <typename ConcreteOp>
  struct PartialEntityAccessOpInterfaceTrait : public ::mlir::OpInterface<PartialEntityAccessOpInterface, detail::PartialEntityAccessOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::acc::detail
namespace mlir::acc::detail {

  template <typename ConcreteOp>
  struct AddressOfGlobalOpInterfaceTrait : public ::mlir::OpInterface<AddressOfGlobalOpInterface, detail::AddressOfGlobalOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::acc::detail
namespace mlir::acc::detail {

  template <typename ConcreteOp>
  struct GlobalVariableOpInterfaceTrait : public ::mlir::OpInterface<GlobalVariableOpInterface, detail::GlobalVariableOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::acc::detail
namespace mlir::acc::detail {

  template <typename ConcreteOp>
  struct IndirectGlobalAccessOpInterfaceTrait : public ::mlir::OpInterface<IndirectGlobalAccessOpInterface, detail::IndirectGlobalAccessOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::acc::detail
namespace mlir::acc::detail {

  template <typename ConcreteOp>
  struct OutlineRematerializationOpInterfaceTrait : public ::mlir::OpInterface<OutlineRematerializationOpInterface, detail::OutlineRematerializationOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::llvm::LogicalResult verifyTrait(::mlir::Operation *op) {
      if (!::mlir::isMemoryEffectFree(op))
      return op->emitOpError("must be memory effect free");
    return ::mlir::success();
    }
  };

} // namespace mlir::acc::detail
namespace mlir::acc::detail {

  template <typename ConcreteOp>
  struct OffloadRegionOpInterfaceTrait : public ::mlir::OpInterface<OffloadRegionOpInterface, detail::OffloadRegionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };

} // namespace mlir::acc::detail
namespace mlir::acc {

template<typename ConcreteOp>
::mlir::Block*detail::ComputeRegionOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return &(llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegion().front();
}
template<typename ConcreteOp>
::mlir::Block*detail::ComputeRegionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAllocaBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAllocaBlock(tablegen_opaque_val);
}

} // namespace mlir::acc
namespace mlir::acc {

template<typename ConcreteOp>
::mlir::Value detail::PartialEntityAccessOpInterfaceInterfaceTraits::Model<ConcreteOp>::getBaseEntity(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBaseEntity();
}
template<typename ConcreteOp>
bool detail::PartialEntityAccessOpInterfaceInterfaceTraits::Model<ConcreteOp>::isCompleteView(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return false;
}
template<typename ConcreteOp>
::mlir::Value detail::PartialEntityAccessOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getBaseEntity(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBaseEntity(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::PartialEntityAccessOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isCompleteView(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isCompleteView(tablegen_opaque_val);
}

} // namespace mlir::acc
namespace mlir::acc {

template<typename ConcreteOp>
::mlir::SymbolRefAttr detail::AddressOfGlobalOpInterfaceInterfaceTraits::Model<ConcreteOp>::getSymbol(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSymbol();
}
template<typename ConcreteOp>
::mlir::SymbolRefAttr detail::AddressOfGlobalOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getSymbol(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getSymbol(tablegen_opaque_val);
}

} // namespace mlir::acc
namespace mlir::acc {

template<typename ConcreteOp>
bool detail::GlobalVariableOpInterfaceInterfaceTraits::Model<ConcreteOp>::isConstant(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return false;
}
template<typename ConcreteOp>
::mlir::Region*detail::GlobalVariableOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInitRegion(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInitRegion();
}
template<typename ConcreteOp>
bool detail::GlobalVariableOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDeviceData(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return false;
}
template<typename ConcreteOp>
bool detail::GlobalVariableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isConstant(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isConstant(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::Region*detail::GlobalVariableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInitRegion(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInitRegion(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::GlobalVariableOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDeviceData(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isDeviceData(tablegen_opaque_val);
}

} // namespace mlir::acc
namespace mlir::acc {

template<typename ConcreteOp>
void detail::IndirectGlobalAccessOpInterfaceInterfaceTraits::Model<ConcreteOp>::getReferencedSymbols(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::SmallVectorImpl<::mlir::SymbolRefAttr>& symbols, ::mlir::SymbolTable * symbolTable) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getReferencedSymbols(symbols, symbolTable);
}
template<typename ConcreteOp>
void detail::IndirectGlobalAccessOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getReferencedSymbols(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::llvm::SmallVectorImpl<::mlir::SymbolRefAttr>& symbols, ::mlir::SymbolTable * symbolTable) {
  return static_cast<const ConcreteOp *>(impl)->getReferencedSymbols(tablegen_opaque_val, symbols, symbolTable);
}

} // namespace mlir::acc
namespace mlir::acc {


} // namespace mlir::acc
namespace mlir::acc {


} // namespace mlir::acc
