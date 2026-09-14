/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::acc {

/// This is a minimal interface to interact with acc dialect data clause
///     operations to represent an OpenACC variable. The reason for its existence
///     is to create a minimal contract between a source dialect and the acc
///     dialect - and it works on the basis that this is a pointer that can
///     be mapped to device memory. This exists for the following reasons:
///     - Many dialects choose to represent their variables as pointers.
///       Specifically locals are created with some form of `alloca` and globals
///       are referred through by their address.
///     - Eventually all such pointers end up as LLVM pointers along with LLVM
///       types whose size is computable. This is the minimal information needed
///       to map a variable.
///     - The OpenACC spec describes reference counters in terms of memory
///       addressability. In 2.6.7. It says: "A structured reference counter
///       is incremented when entering each data or compute region that contain
///       an explicit data clause or implicitly-determined data attributes for
///       that section of memory". This implies addressability of memory.
///     - Attach semantics (2.6.8 attachment counter) are specified using
///       "address" terminology: "The attachment counter for a pointer is set to
///       one whenever the pointer is attached to new target address, and
///       incremented whenever an attach action for that pointer is performed for
///       the same target address."
class PointerLikeType;

} // namespace mlir::acc
namespace mlir::acc {

/// This interface is a richer contract than being a pointer-like type
///     and can be used in conjunction with it.
///     It should be attached to types that a source dialect considers to
///     be variables. And unlike pointer-like type, it can be attached to variables
///     which the source dialect does not represent through the use of memory.
///     The richer API allows for post-frontend type-based semantics to be
///     applied such as generating recipes or extracting array bounds.
class MappableType;

} // namespace mlir::acc
namespace mlir::acc {

/// This is a minimal interface to interact with acc dialect data clause
///     operations to represent an OpenACC variable. The reason for its existence
///     is to create a minimal contract between a source dialect and the acc
///     dialect - and it works on the basis that this is a pointer that can
///     be mapped to device memory. This exists for the following reasons:
///     - Many dialects choose to represent their variables as pointers.
///       Specifically locals are created with some form of `alloca` and globals
///       are referred through by their address.
///     - Eventually all such pointers end up as LLVM pointers along with LLVM
///       types whose size is computable. This is the minimal information needed
///       to map a variable.
///     - The OpenACC spec describes reference counters in terms of memory
///       addressability. In 2.6.7. It says: "A structured reference counter
///       is incremented when entering each data or compute region that contain
///       an explicit data clause or implicitly-determined data attributes for
///       that section of memory". This implies addressability of memory.
///     - Attach semantics (2.6.8 attachment counter) are specified using
///       "address" terminology: "The attachment counter for a pointer is set to
///       one whenever the pointer is attached to new target address, and
///       incremented whenever an attach action for that pointer is performed for
///       the same target address."
namespace detail {
struct PointerLikeTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::Type (*getElementType)(const Concept *impl, ::mlir::Type );
    ::mlir::acc::VariableTypeCategory (*getPointeeTypeCategory)(const Concept *impl, ::mlir::Type , ::mlir::TypedValue<::mlir::acc::PointerLikeType>, ::mlir::Type);
    ::mlir::Value (*genAllocate)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::llvm::StringRef, ::mlir::Type, ::mlir::Value, bool &);
    bool (*genFree)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypedValue<::mlir::acc::PointerLikeType>, ::mlir::Value, ::mlir::Type);
    bool (*genCopy)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypedValue<::mlir::acc::PointerLikeType>, ::mlir::TypedValue<::mlir::acc::PointerLikeType>, ::mlir::Type);
    ::mlir::Value (*genLoad)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypedValue<::mlir::acc::PointerLikeType>, ::mlir::Type);
    bool (*genStore)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::mlir::Value, ::mlir::TypedValue<::mlir::acc::PointerLikeType>);
    bool (*isDeviceData)(const Concept *impl, ::mlir::Type , ::mlir::Value);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::PointerLikeType;
    Model() : Concept{getElementType, getPointeeTypeCategory, genAllocate, genFree, genCopy, genLoad, genStore, isDeviceData} {}

    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType);
    static inline ::mlir::Value genAllocate(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool & needsFree);
    static inline bool genFree(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType);
    static inline bool genCopy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType);
    static inline ::mlir::Value genLoad(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType);
    static inline bool genStore(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr);
    static inline bool isDeviceData(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::PointerLikeType;
    FallbackModel() : Concept{getElementType, getPointeeTypeCategory, genAllocate, genFree, genCopy, genLoad, genStore, isDeviceData} {}

    static inline ::mlir::Type getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType);
    static inline ::mlir::Value genAllocate(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool & needsFree);
    static inline bool genFree(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType);
    static inline bool genCopy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType);
    static inline ::mlir::Value genLoad(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType);
    static inline bool genStore(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr);
    static inline bool isDeviceData(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
    ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const;
    ::mlir::Value genAllocate(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool &needsFree) const;
    bool genFree(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType) const;
    bool genCopy(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType) const;
    ::mlir::Value genLoad(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType) const;
    bool genStore(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr) const;
    bool isDeviceData(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const;
  };
};
template <typename ConcreteType>
struct PointerLikeTypeTrait;

} // namespace detail
class PointerLikeType : public ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::PointerLikeTypeTrait<ConcreteType> {};

  /// Returns the pointee type or null if the pointer has no pointee type
  ::mlir::Type getElementType() const;

  /// Returns the type category of the pointee. The `var` is provided because
  /// a dialect's type system may be incomplete. For example, consider a
  /// dialect which computes interior pointers - so a float array element
  /// may be represented as `ptr<f32>`. The type system says the pointee
  /// is `f32` but this is not a scalar from the point-of-view of OpenACC.
  /// It is an array element and thus the appropriate type category is
  /// "array" - therefore being able to look up how a variable is computed
  /// is important for a complete type determination.
  /// The `varType` is provided in cases where a dialect's type system
  /// erased the target type.
  ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const;

  /// Generates allocation operations for the pointer-like type. It will create
  /// an allocate operation that produces memory space for an instance of the
  /// current type.
  /// 
  /// The `varName` parameter is optional and can be used to provide a name
  /// for the allocated variable. When provided, it must be used by the
  /// implementation; and if the implementing dialect does not have its own
  /// way to save it, the discardable `acc.var_name` attribute from the acc
  /// dialect will be used.
  /// 
  /// If the current type is represented in a way that it does not capture
  /// the pointee type, `varType` must be passed in to provide the necessary
  /// type information.
  /// 
  /// The `originalVar` parameter is optional but enables support for dynamic
  /// types (e.g., dynamic memrefs). When provided, implementations can extract
  /// runtime dimension information from the original variable to create
  /// allocations with matching dynamic sizes. When generating recipe bodies,
  /// `originalVar` should be the block argument representing the original
  /// variable in the recipe region.
  /// 
  /// The `needsFree` output parameter indicates whether the allocated memory
  /// requires explicit deallocation. Implementations should set this to true
  /// for heap allocations that need a matching deallocation operation (e.g.,
  /// alloc) and false for stack-based allocations (e.g., alloca). During
  /// recipe generation, this determines whether a destroy region is created.
  /// 
  /// Returns a Value representing the result of the allocation. If no value
  /// is returned, it means the allocation was not successfully generated.
  ::mlir::Value genAllocate(::mlir::OpBuilder & builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool & needsFree) const;

  /// Generates deallocation operations for the pointer-like type.
  /// 
  /// The `varToFree` parameter is required and must represent an instance
  /// that was previously allocated. When generating recipe bodies, this
  /// should be the block argument representing the private variable in the
  /// destroy region.
  /// 
  /// The `allocRes` parameter is optional and provides the result of the
  /// corresponding allocation from the init region. This allows implementations
  /// to inspect the allocation operation to determine the appropriate
  /// deallocation strategy. This is necessary because in recipe generation,
  /// the allocation and deallocation occur in separate regions. Dialects that
  /// use only one allocation type or can determine deallocation from type
  /// information alone may ignore this parameter.
  /// 
  /// The `varType` parameter must be provided if the current type does not
  /// capture the pointee type information. No deallocation is generated for
  /// stack-based allocations (e.g., alloca).
  /// 
  /// Returns true if deallocation was successfully generated or determined to
  /// be unnecessary, false otherwise.
  bool genFree(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType) const;

  /// Generates copy operations for the pointer-like type. It copies the memory
  /// from the source to the destination. Typically used to initialize one
  /// variable of this type from another.
  /// 
  /// The `destination` and `source` parameters represent the target and source
  /// instances respectively. If the current type is represented in a way that it
  /// does not capture the pointee type, `varType` must be passed in to provide
  /// the necessary type information.
  /// 
  /// Returns true if copy was successfully generated, false otherwise.
  bool genCopy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType) const;

  /// Generates a load operation from the pointer-like type. This dereferences
  /// the pointer and returns the loaded value.
  /// 
  /// The `srcPtr` parameter is the pointer to load from. If the current type is
  /// represented in a way that it does not capture the pointee type, `valueType`
  /// must be passed in to provide the necessary type information.
  /// 
  /// Returns the loaded value, or an empty Value if load generation failed.
  ::mlir::Value genLoad(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType) const;

  /// Generates a store operation to the pointer-like type. This stores a value
  /// to the memory location pointed to by the pointer.
  /// 
  /// The `destPtr` parameter is the pointer to store to. The `valueToStore`
  /// parameter is the value to be stored. The type information is derived from
  /// the valueToStore parameter itself.
  /// 
  /// Returns true if store was successfully generated, false otherwise.
  bool genStore(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr) const;

  /// Returns true if the pointer points to device data.
  bool isDeviceData(::mlir::Value var) const;
};

} // namespace mlir::acc
namespace mlir::acc {

/// This interface is a richer contract than being a pointer-like type
///     and can be used in conjunction with it.
///     It should be attached to types that a source dialect considers to
///     be variables. And unlike pointer-like type, it can be attached to variables
///     which the source dialect does not represent through the use of memory.
///     The richer API allows for post-frontend type-based semantics to be
///     applied such as generating recipes or extracting array bounds.
namespace detail {
struct MappableTypeInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    ::mlir::TypedValue<::mlir::acc::PointerLikeType> (*getVarPtr)(const Concept *impl, ::mlir::Type , ::mlir::Value);
    ::std::optional<::llvm::TypeSize> (*getSizeInBytes)(const Concept *impl, ::mlir::Type , ::mlir::Value, ::mlir::ValueRange, const ::mlir::DataLayout &);
    ::std::optional<::int64_t> (*getOffsetInBytes)(const Concept *impl, ::mlir::Type , ::mlir::Value, ::mlir::ValueRange, const ::mlir::DataLayout &);
    bool (*hasUnknownDimensions)(const Concept *impl, ::mlir::Type );
    ::llvm::SmallVector<::mlir::Value> (*generateAccBounds)(const Concept *impl, ::mlir::Type , ::mlir::Value, ::mlir::OpBuilder &);
    ::mlir::acc::VariableTypeCategory (*getTypeCategory)(const Concept *impl, ::mlir::Type , ::mlir::Value);
    ::mlir::Value (*generatePrivateInit)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypedValue<::mlir::acc::MappableType>, ::llvm::StringRef, ::mlir::ValueRange, ::mlir::Value, bool &);
    bool (*generateCopy)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypedValue<::mlir::acc::MappableType>, ::mlir::TypedValue<::mlir::acc::MappableType>, ::mlir::ValueRange);
    bool (*generateCombiner)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypedValue<::mlir::acc::MappableType>, ::mlir::TypedValue<::mlir::acc::MappableType>, ::mlir::ValueRange, ::mlir::acc::ReductionOperator, ::mlir::Attribute);
    bool (*generatePrivateDestroy)(const Concept *impl, ::mlir::Type , ::mlir::OpBuilder &, ::mlir::Location, ::mlir::Value, ::mlir::ValueRange);
    bool (*isDeviceData)(const Concept *impl, ::mlir::Type , ::mlir::Value);
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::acc::MappableType;
    Model() : Concept{getVarPtr, getSizeInBytes, getOffsetInBytes, hasUnknownDimensions, generateAccBounds, getTypeCategory, generatePrivateInit, generateCopy, generateCombiner, generatePrivateDestroy, isDeviceData} {}

    static inline ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
    static inline ::std::optional<::llvm::TypeSize> getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout);
    static inline ::std::optional<::int64_t> getOffsetInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout);
    static inline bool hasUnknownDimensions(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::Value> generateAccBounds(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder & builder);
    static inline ::mlir::acc::VariableTypeCategory getTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
    static inline ::mlir::Value generatePrivateInit(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool & needsDestroy);
    static inline bool generateCopy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds);
    static inline bool generateCombiner(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags);
    static inline bool generatePrivateDestroy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents);
    static inline bool isDeviceData(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::acc::MappableType;
    FallbackModel() : Concept{getVarPtr, getSizeInBytes, getOffsetInBytes, hasUnknownDimensions, generateAccBounds, getTypeCategory, generatePrivateInit, generateCopy, generateCombiner, generatePrivateDestroy, isDeviceData} {}

    static inline ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
    static inline ::std::optional<::llvm::TypeSize> getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout);
    static inline ::std::optional<::int64_t> getOffsetInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout);
    static inline bool hasUnknownDimensions(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline ::llvm::SmallVector<::mlir::Value> generateAccBounds(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder & builder);
    static inline ::mlir::acc::VariableTypeCategory getTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
    static inline ::mlir::Value generatePrivateInit(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool & needsDestroy);
    static inline bool generateCopy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds);
    static inline bool generateCombiner(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags);
    static inline bool generatePrivateDestroy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents);
    static inline bool isDeviceData(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
    ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const;
    ::std::optional<::llvm::TypeSize> getSizeInBytes(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout &dataLayout) const;
    ::std::optional<::int64_t> getOffsetInBytes(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout &dataLayout) const;
    ::llvm::SmallVector<::mlir::Value> generateAccBounds(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder &builder) const;
    ::mlir::acc::VariableTypeCategory getTypeCategory(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const;
    ::mlir::Value generatePrivateInit(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool &needsDestroy) const;
    bool generateCopy(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds) const;
    bool generateCombiner(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags) const;
    bool generatePrivateDestroy(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents) const;
    bool isDeviceData(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const;
  };
};
template <typename ConcreteType>
struct MappableTypeTrait;

} // namespace detail
class MappableType : public ::mlir::TypeInterface<MappableType, detail::MappableTypeInterfaceTraits> {
public:
  using ::mlir::TypeInterface<MappableType, detail::MappableTypeInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::MappableTypeTrait<ConcreteType> {};

  /// Returns the pointer to the `var` if recoverable (such as in cases
  /// where the current operation is a load from a memory slot).
  ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(::mlir::Value var) const;

  /// Returns the size in bytes when computable. If this is an array-like
  /// type, avoiding passing `accBounds` ensures a computation of the size
  /// of whole type.
  ::std::optional<::llvm::TypeSize> getSizeInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const;

  /// Returns the offset in bytes when computable.
  ::std::optional<::int64_t> getOffsetInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const;

  /// Returns true if the dimensions of this type are not known. This occurs
  /// when the MLIR type does not encode dimensional information and there is
  /// no associated descriptor or metadata in the current entity that would
  /// make this information extractable. For example, an opaque pointer type
  /// pointing to an array without dimension information would have unknown
  /// dimensions.
  bool hasUnknownDimensions() const;

  /// Returns explicit `acc.bounds` operations that envelop the whole
  /// data structure. These operations are inserted using the provided builder
  /// at the location set before calling this API.
  ::llvm::SmallVector<::mlir::Value> generateAccBounds(::mlir::Value var, ::mlir::OpBuilder & builder) const;

  /// Returns the OpenACC type category.
  ::mlir::acc::VariableTypeCategory getTypeCategory(::mlir::Value var) const;

  /// Generates the operations that would be normally placed in a recipe's
  /// init region. It inserts at the builder's current location.
  /// It can be used either to directly "inline" the init region
  /// or if the caller sets the insertion point to inside a recipe body,
  /// it fills it in. This does not generate the `acc.yield` that normally
  /// would terminate a recipe.
  /// 
  /// The `extents` are optional and can be empty - it is only when a
  /// slice of the private variable needs allocation.
  /// The `initVal` can be empty - it is primarily needed for reductions
  /// to ensure the variable is also initialized with appropriate value.
  /// 
  /// The `needsDestroy` out-parameter is set by implementations to indicate
  /// that destruction code must be generated after the returned private
  /// variable usages, typically in the destroy region of recipe operations
  /// (for example, when heap allocations or temporaries requiring cleanup
  /// are created during initialization). When `needsDestroy` is set, callers
  /// should invoke `generatePrivateDestroy` in the recipe's destroy region
  /// with the privatized value returned by this method.
  /// 
  /// If the return value is empty, it means that recipe body was not
  /// successfully generated.
  ::mlir::Value generatePrivateInit(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool & needsDestroy) const;

  bool generateCopy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds) const;

  bool generateCombiner(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags) const;

  /// Generates destruction operations for a privatized value previously
  /// produced by `generatePrivateInit`. This is typically inserted in a
  /// recipe's destroy region, after all uses of the privatized value.
  /// 
  /// The `privatized` value is the SSA value yielded by the init region
  /// (and passed as the privatized argument to the destroy region).
  /// Implementations should free heap-allocated storage or perform any
  /// cleanup required for the given type. If no destruction is required,
  /// this function should be a no-op and return `true`.
  /// 
  /// The `bounds` must be passed when only a section of the variable was
  /// privatized. 
  /// 
  /// Returns true if destruction was successfully generated or deemed not
  /// necessary, false otherwise.
  bool generatePrivateDestroy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents) const;

  /// Returns true if the variable represents device data.
  bool isDeviceData(::mlir::Value var) const;
};

} // namespace mlir::acc
namespace mlir::acc::detail {

  template <typename ConcreteType>
  struct PointerLikeTypeTrait : public ::mlir::TypeInterface<PointerLikeType, detail::PointerLikeTypeInterfaceTraits>::Trait<ConcreteType> {

    /// Returns the type category of the pointee. The `var` is provided because
    /// a dialect's type system may be incomplete. For example, consider a
    /// dialect which computes interior pointers - so a float array element
    /// may be represented as `ptr<f32>`. The type system says the pointee
    /// is `f32` but this is not a scalar from the point-of-view of OpenACC.
    /// It is an array element and thus the appropriate type category is
    /// "array" - therefore being able to look up how a variable is computed
    /// is important for a complete type determination.
    /// The `varType` is provided in cases where a dialect's type system
    /// erased the target type.
    ::mlir::acc::VariableTypeCategory getPointeeTypeCategory(::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const {
      return ::mlir::acc::VariableTypeCategory::uncategorized;
    }

    /// Generates allocation operations for the pointer-like type. It will create
    /// an allocate operation that produces memory space for an instance of the
    /// current type.
    /// 
    /// The `varName` parameter is optional and can be used to provide a name
    /// for the allocated variable. When provided, it must be used by the
    /// implementation; and if the implementing dialect does not have its own
    /// way to save it, the discardable `acc.var_name` attribute from the acc
    /// dialect will be used.
    /// 
    /// If the current type is represented in a way that it does not capture
    /// the pointee type, `varType` must be passed in to provide the necessary
    /// type information.
    /// 
    /// The `originalVar` parameter is optional but enables support for dynamic
    /// types (e.g., dynamic memrefs). When provided, implementations can extract
    /// runtime dimension information from the original variable to create
    /// allocations with matching dynamic sizes. When generating recipe bodies,
    /// `originalVar` should be the block argument representing the original
    /// variable in the recipe region.
    /// 
    /// The `needsFree` output parameter indicates whether the allocated memory
    /// requires explicit deallocation. Implementations should set this to true
    /// for heap allocations that need a matching deallocation operation (e.g.,
    /// alloc) and false for stack-based allocations (e.g., alloca). During
    /// recipe generation, this determines whether a destroy region is created.
    /// 
    /// Returns a Value representing the result of the allocation. If no value
    /// is returned, it means the allocation was not successfully generated.
    ::mlir::Value genAllocate(::mlir::OpBuilder & builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool & needsFree) const {
      return {};
    }

    /// Generates deallocation operations for the pointer-like type.
    /// 
    /// The `varToFree` parameter is required and must represent an instance
    /// that was previously allocated. When generating recipe bodies, this
    /// should be the block argument representing the private variable in the
    /// destroy region.
    /// 
    /// The `allocRes` parameter is optional and provides the result of the
    /// corresponding allocation from the init region. This allows implementations
    /// to inspect the allocation operation to determine the appropriate
    /// deallocation strategy. This is necessary because in recipe generation,
    /// the allocation and deallocation occur in separate regions. Dialects that
    /// use only one allocation type or can determine deallocation from type
    /// information alone may ignore this parameter.
    /// 
    /// The `varType` parameter must be provided if the current type does not
    /// capture the pointee type information. No deallocation is generated for
    /// stack-based allocations (e.g., alloca).
    /// 
    /// Returns true if deallocation was successfully generated or determined to
    /// be unnecessary, false otherwise.
    bool genFree(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType) const {
      return false;
    }

    /// Generates copy operations for the pointer-like type. It copies the memory
    /// from the source to the destination. Typically used to initialize one
    /// variable of this type from another.
    /// 
    /// The `destination` and `source` parameters represent the target and source
    /// instances respectively. If the current type is represented in a way that it
    /// does not capture the pointee type, `varType` must be passed in to provide
    /// the necessary type information.
    /// 
    /// Returns true if copy was successfully generated, false otherwise.
    bool genCopy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType) const {
      return false;
    }

    /// Generates a load operation from the pointer-like type. This dereferences
    /// the pointer and returns the loaded value.
    /// 
    /// The `srcPtr` parameter is the pointer to load from. If the current type is
    /// represented in a way that it does not capture the pointee type, `valueType`
    /// must be passed in to provide the necessary type information.
    /// 
    /// Returns the loaded value, or an empty Value if load generation failed.
    ::mlir::Value genLoad(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType) const {
      return {};
    }

    /// Generates a store operation to the pointer-like type. This stores a value
    /// to the memory location pointed to by the pointer.
    /// 
    /// The `destPtr` parameter is the pointer to store to. The `valueToStore`
    /// parameter is the value to be stored. The type information is derived from
    /// the valueToStore parameter itself.
    /// 
    /// Returns true if store was successfully generated, false otherwise.
    bool genStore(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr) const {
      return false;
    }

    /// Returns true if the pointer points to device data.
    bool isDeviceData(::mlir::Value var) const {
      return false;
    }
  };

} // namespace mlir::acc::detail
namespace mlir::acc::detail {

  template <typename ConcreteType>
  struct MappableTypeTrait : public ::mlir::TypeInterface<MappableType, detail::MappableTypeInterfaceTraits>::Trait<ConcreteType> {

    /// Returns the pointer to the `var` if recoverable (such as in cases
    /// where the current operation is a load from a memory slot).
    ::mlir::TypedValue<::mlir::acc::PointerLikeType> getVarPtr(::mlir::Value var) const {
      if (auto ptr = mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(
              var))
          return ptr;
        return {};
    }

    /// Returns the size in bytes when computable. If this is an array-like
    /// type, avoiding passing `accBounds` ensures a computation of the size
    /// of whole type.
    ::std::optional<::llvm::TypeSize> getSizeInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const {
      // Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};
        return {dataLayout.getTypeSize((*static_cast<const ConcreteType *>(this)))};
    }

    /// Returns the offset in bytes when computable.
    ::std::optional<::int64_t> getOffsetInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const {
      // Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};

        // If the type size is computable, it means it is trivial. Assume
        // offset of 0.
        if (::mlir::cast<::mlir::acc::MappableType>((*static_cast<const ConcreteType *>(this))).getSizeInBytes(
              var, accBounds, dataLayout).has_value()) {
          return {0};
        }

        return {};
    }

    /// Returns explicit `acc.bounds` operations that envelop the whole
    /// data structure. These operations are inserted using the provided builder
    /// at the location set before calling this API.
    ::llvm::SmallVector<::mlir::Value> generateAccBounds(::mlir::Value var, ::mlir::OpBuilder & builder) const {
      return {};
    }

    /// Returns the OpenACC type category.
    ::mlir::acc::VariableTypeCategory getTypeCategory(::mlir::Value var) const {
      return ::mlir::acc::VariableTypeCategory::uncategorized;
    }

    /// Generates the operations that would be normally placed in a recipe's
    /// init region. It inserts at the builder's current location.
    /// It can be used either to directly "inline" the init region
    /// or if the caller sets the insertion point to inside a recipe body,
    /// it fills it in. This does not generate the `acc.yield` that normally
    /// would terminate a recipe.
    /// 
    /// The `extents` are optional and can be empty - it is only when a
    /// slice of the private variable needs allocation.
    /// The `initVal` can be empty - it is primarily needed for reductions
    /// to ensure the variable is also initialized with appropriate value.
    /// 
    /// The `needsDestroy` out-parameter is set by implementations to indicate
    /// that destruction code must be generated after the returned private
    /// variable usages, typically in the destroy region of recipe operations
    /// (for example, when heap allocations or temporaries requiring cleanup
    /// are created during initialization). When `needsDestroy` is set, callers
    /// should invoke `generatePrivateDestroy` in the recipe's destroy region
    /// with the privatized value returned by this method.
    /// 
    /// If the return value is empty, it means that recipe body was not
    /// successfully generated.
    ::mlir::Value generatePrivateInit(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool & needsDestroy) const {
      return {};
    }

    bool generateCopy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds) const {
      return false;
    }

    bool generateCombiner(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags) const {
      return false;
    }

    /// Generates destruction operations for a privatized value previously
    /// produced by `generatePrivateInit`. This is typically inserted in a
    /// recipe's destroy region, after all uses of the privatized value.
    /// 
    /// The `privatized` value is the SSA value yielded by the init region
    /// (and passed as the privatized argument to the destroy region).
    /// Implementations should free heap-allocated storage or perform any
    /// cleanup required for the given type. If no destruction is required,
    /// this function should be a no-op and return `true`.
    /// 
    /// The `bounds` must be passed when only a section of the variable was
    /// privatized. 
    /// 
    /// Returns true if destruction was successfully generated or deemed not
    /// necessary, false otherwise.
    bool generatePrivateDestroy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents) const {
      return true;
    }

    /// Returns true if the variable represents device data.
    bool isDeviceData(::mlir::Value var) const {
      return false;
    }
  };

} // namespace mlir::acc::detail
namespace mlir::acc {

template<typename ConcreteType>
::mlir::Type detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getElementType();
}
template<typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::getPointeeTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getPointeeTypeCategory(varPtr, varType);
}
template<typename ConcreteType>
::mlir::Value detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::genAllocate(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool & needsFree) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).genAllocate(builder, loc, varName, varType, originalVar, needsFree);
}
template<typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::genFree(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).genFree(builder, loc, varToFree, allocRes, varType);
}
template<typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::genCopy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).genCopy(builder, loc, destination, source, varType);
}
template<typename ConcreteType>
::mlir::Value detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::genLoad(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).genLoad(builder, loc, srcPtr, valueType);
}
template<typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::genStore(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).genStore(builder, loc, valueToStore, destPtr);
}
template<typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::Model<ConcreteType>::isDeviceData(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).isDeviceData(var);
}
template<typename ConcreteType>
::mlir::Type detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::getElementType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getElementType(tablegen_opaque_val);
}
template<typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::getPointeeTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) {
  return static_cast<const ConcreteType *>(impl)->getPointeeTypeCategory(tablegen_opaque_val, varPtr, varType);
}
template<typename ConcreteType>
::mlir::Value detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::genAllocate(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool & needsFree) {
  return static_cast<const ConcreteType *>(impl)->genAllocate(tablegen_opaque_val, builder, loc, varName, varType, originalVar, needsFree);
}
template<typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::genFree(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType) {
  return static_cast<const ConcreteType *>(impl)->genFree(tablegen_opaque_val, builder, loc, varToFree, allocRes, varType);
}
template<typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::genCopy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType) {
  return static_cast<const ConcreteType *>(impl)->genCopy(tablegen_opaque_val, builder, loc, destination, source, varType);
}
template<typename ConcreteType>
::mlir::Value detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::genLoad(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType) {
  return static_cast<const ConcreteType *>(impl)->genLoad(tablegen_opaque_val, builder, loc, srcPtr, valueType);
}
template<typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::genStore(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr) {
  return static_cast<const ConcreteType *>(impl)->genStore(tablegen_opaque_val, builder, loc, valueToStore, destPtr);
}
template<typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::FallbackModel<ConcreteType>::isDeviceData(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return static_cast<const ConcreteType *>(impl)->isDeviceData(tablegen_opaque_val, var);
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::PointerLikeTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getPointeeTypeCategory(::mlir::Type tablegen_opaque_val, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const {
return ::mlir::acc::VariableTypeCategory::uncategorized;
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::Value detail::PointerLikeTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::genAllocate(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool &needsFree) const {
return {};
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::genFree(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType) const {
return false;
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::genCopy(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType) const {
return false;
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::Value detail::PointerLikeTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::genLoad(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType) const {
return {};
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::genStore(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr) const {
return false;
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::PointerLikeTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::isDeviceData(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const {
return false;
}

} // namespace mlir::acc
namespace mlir::acc {

template<typename ConcreteType>
::mlir::TypedValue<::mlir::acc::PointerLikeType> detail::MappableTypeInterfaceTraits::Model<ConcreteType>::getVarPtr(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getVarPtr(var);
}
template<typename ConcreteType>
::std::optional<::llvm::TypeSize> detail::MappableTypeInterfaceTraits::Model<ConcreteType>::getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getSizeInBytes(var, accBounds, dataLayout);
}
template<typename ConcreteType>
::std::optional<::int64_t> detail::MappableTypeInterfaceTraits::Model<ConcreteType>::getOffsetInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getOffsetInBytes(var, accBounds, dataLayout);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::Model<ConcreteType>::hasUnknownDimensions(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).hasUnknownDimensions();
}
template<typename ConcreteType>
::llvm::SmallVector<::mlir::Value> detail::MappableTypeInterfaceTraits::Model<ConcreteType>::generateAccBounds(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder & builder) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).generateAccBounds(var, builder);
}
template<typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::MappableTypeInterfaceTraits::Model<ConcreteType>::getTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getTypeCategory(var);
}
template<typename ConcreteType>
::mlir::Value detail::MappableTypeInterfaceTraits::Model<ConcreteType>::generatePrivateInit(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool & needsDestroy) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).generatePrivateInit(builder, loc, var, varName, extents, initVal, needsDestroy);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::Model<ConcreteType>::generateCopy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).generateCopy(builder, loc, from, to, bounds);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::Model<ConcreteType>::generateCombiner(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).generateCombiner(builder, loc, dest, source, bounds, op, fastmathFlags);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::Model<ConcreteType>::generatePrivateDestroy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).generatePrivateDestroy(builder, loc, privatized, extents);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::Model<ConcreteType>::isDeviceData(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).isDeviceData(var);
}
template<typename ConcreteType>
::mlir::TypedValue<::mlir::acc::PointerLikeType> detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::getVarPtr(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return static_cast<const ConcreteType *>(impl)->getVarPtr(tablegen_opaque_val, var);
}
template<typename ConcreteType>
::std::optional<::llvm::TypeSize> detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::getSizeInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteType *>(impl)->getSizeInBytes(tablegen_opaque_val, var, accBounds, dataLayout);
}
template<typename ConcreteType>
::std::optional<::int64_t> detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::getOffsetInBytes(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) {
  return static_cast<const ConcreteType *>(impl)->getOffsetInBytes(tablegen_opaque_val, var, accBounds, dataLayout);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::hasUnknownDimensions(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->hasUnknownDimensions(tablegen_opaque_val);
}
template<typename ConcreteType>
::llvm::SmallVector<::mlir::Value> detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::generateAccBounds(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder & builder) {
  return static_cast<const ConcreteType *>(impl)->generateAccBounds(tablegen_opaque_val, var, builder);
}
template<typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::getTypeCategory(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return static_cast<const ConcreteType *>(impl)->getTypeCategory(tablegen_opaque_val, var);
}
template<typename ConcreteType>
::mlir::Value detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::generatePrivateInit(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool & needsDestroy) {
  return static_cast<const ConcreteType *>(impl)->generatePrivateInit(tablegen_opaque_val, builder, loc, var, varName, extents, initVal, needsDestroy);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::generateCopy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds) {
  return static_cast<const ConcreteType *>(impl)->generateCopy(tablegen_opaque_val, builder, loc, from, to, bounds);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::generateCombiner(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags) {
  return static_cast<const ConcreteType *>(impl)->generateCombiner(tablegen_opaque_val, builder, loc, dest, source, bounds, op, fastmathFlags);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::generatePrivateDestroy(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents) {
  return static_cast<const ConcreteType *>(impl)->generatePrivateDestroy(tablegen_opaque_val, builder, loc, privatized, extents);
}
template<typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::FallbackModel<ConcreteType>::isDeviceData(const Concept *impl, ::mlir::Type tablegen_opaque_val, ::mlir::Value var) {
  return static_cast<const ConcreteType *>(impl)->isDeviceData(tablegen_opaque_val, var);
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::TypedValue<::mlir::acc::PointerLikeType> detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getVarPtr(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const {
if (auto ptr = mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(
              var))
          return ptr;
        return {};
}
template<typename ConcreteModel, typename ConcreteType>
::std::optional<::llvm::TypeSize> detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getSizeInBytes(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout &dataLayout) const {
// Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};
        return {dataLayout.getTypeSize((::llvm::cast<ConcreteType>(tablegen_opaque_val)))};
}
template<typename ConcreteModel, typename ConcreteType>
::std::optional<::int64_t> detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getOffsetInBytes(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout &dataLayout) const {
// Bounds operations are typically created for array types. In the
        // generic implementation, it is not straightforward to distinguish
        // between array types and ensure the size and offset take into account
        // just the slice requested. Thus return not-computable for now.
        if (!accBounds.empty())
          return {};

        // If the type size is computable, it means it is trivial. Assume
        // offset of 0.
        if (::mlir::cast<::mlir::acc::MappableType>((::llvm::cast<ConcreteType>(tablegen_opaque_val))).getSizeInBytes(
              var, accBounds, dataLayout).has_value()) {
          return {0};
        }

        return {};
}
template<typename ConcreteModel, typename ConcreteType>
::llvm::SmallVector<::mlir::Value> detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::generateAccBounds(::mlir::Type tablegen_opaque_val, ::mlir::Value var, ::mlir::OpBuilder &builder) const {
return {};
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::acc::VariableTypeCategory detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getTypeCategory(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const {
return ::mlir::acc::VariableTypeCategory::uncategorized;
}
template<typename ConcreteModel, typename ConcreteType>
::mlir::Value detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::generatePrivateInit(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool &needsDestroy) const {
return {};
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::generateCopy(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds) const {
return false;
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::generateCombiner(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags) const {
return false;
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::generatePrivateDestroy(::mlir::Type tablegen_opaque_val, ::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents) const {
return true;
}
template<typename ConcreteModel, typename ConcreteType>
bool detail::MappableTypeInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::isDeviceData(::mlir::Type tablegen_opaque_val, ::mlir::Value var) const {
return false;
}

} // namespace mlir::acc
