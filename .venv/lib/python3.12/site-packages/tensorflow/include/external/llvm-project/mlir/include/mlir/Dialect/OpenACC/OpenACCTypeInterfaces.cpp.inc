/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Returns the pointer to the `var` if recoverable (such as in cases
/// where the current operation is a load from a memory slot).
::mlir::TypedValue<::mlir::acc::PointerLikeType> mlir::acc::MappableType::getVarPtr(::mlir::Value var) const {
      return getImpl()->getVarPtr(getImpl(), *this, var);
  }

/// Returns the size in bytes when computable. If this is an array-like
/// type, avoiding passing `accBounds` ensures a computation of the size
/// of whole type.
::std::optional<::llvm::TypeSize> mlir::acc::MappableType::getSizeInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const {
      return getImpl()->getSizeInBytes(getImpl(), *this, var, accBounds, dataLayout);
  }

/// Returns the offset in bytes when computable.
::std::optional<::int64_t> mlir::acc::MappableType::getOffsetInBytes(::mlir::Value var, ::mlir::ValueRange accBounds, const ::mlir::DataLayout & dataLayout) const {
      return getImpl()->getOffsetInBytes(getImpl(), *this, var, accBounds, dataLayout);
  }

/// Returns true if the dimensions of this type are not known. This occurs
/// when the MLIR type does not encode dimensional information and there is
/// no associated descriptor or metadata in the current entity that would
/// make this information extractable. For example, an opaque pointer type
/// pointing to an array without dimension information would have unknown
/// dimensions.
bool mlir::acc::MappableType::hasUnknownDimensions() const {
      return getImpl()->hasUnknownDimensions(getImpl(), *this);
  }

/// Returns explicit `acc.bounds` operations that envelop the whole
/// data structure. These operations are inserted using the provided builder
/// at the location set before calling this API.
::llvm::SmallVector<::mlir::Value> mlir::acc::MappableType::generateAccBounds(::mlir::Value var, ::mlir::OpBuilder & builder) const {
      return getImpl()->generateAccBounds(getImpl(), *this, var, builder);
  }

/// Returns the OpenACC type category.
::mlir::acc::VariableTypeCategory mlir::acc::MappableType::getTypeCategory(::mlir::Value var) const {
      return getImpl()->getTypeCategory(getImpl(), *this, var);
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
::mlir::Value mlir::acc::MappableType::generatePrivateInit(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> var, ::llvm::StringRef varName, ::mlir::ValueRange extents, ::mlir::Value initVal, bool & needsDestroy) const {
      return getImpl()->generatePrivateInit(getImpl(), *this, builder, loc, var, varName, extents, initVal, needsDestroy);
  }

bool mlir::acc::MappableType::generateCopy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> from, ::mlir::TypedValue<::mlir::acc::MappableType> to, ::mlir::ValueRange bounds) const {
      return getImpl()->generateCopy(getImpl(), *this, builder, loc, from, to, bounds);
  }

bool mlir::acc::MappableType::generateCombiner(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::MappableType> dest, ::mlir::TypedValue<::mlir::acc::MappableType> source, ::mlir::ValueRange bounds, ::mlir::acc::ReductionOperator op, ::mlir::Attribute fastmathFlags) const {
      return getImpl()->generateCombiner(getImpl(), *this, builder, loc, dest, source, bounds, op, fastmathFlags);
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
bool mlir::acc::MappableType::generatePrivateDestroy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value privatized, ::mlir::ValueRange extents) const {
      return getImpl()->generatePrivateDestroy(getImpl(), *this, builder, loc, privatized, extents);
  }

/// Returns true if the variable represents device data.
bool mlir::acc::MappableType::isDeviceData(::mlir::Value var) const {
      return getImpl()->isDeviceData(getImpl(), *this, var);
  }

/// Returns the pointee type or null if the pointer has no pointee type
::mlir::Type mlir::acc::PointerLikeType::getElementType() const {
      return getImpl()->getElementType(getImpl(), *this);
  }

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
::mlir::acc::VariableTypeCategory mlir::acc::PointerLikeType::getPointeeTypeCategory(::mlir::TypedValue<::mlir::acc::PointerLikeType> varPtr, ::mlir::Type varType) const {
      return getImpl()->getPointeeTypeCategory(getImpl(), *this, varPtr, varType);
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
::mlir::Value mlir::acc::PointerLikeType::genAllocate(::mlir::OpBuilder & builder, ::mlir::Location loc, ::llvm::StringRef varName, ::mlir::Type varType, ::mlir::Value originalVar, bool & needsFree) const {
      return getImpl()->genAllocate(getImpl(), *this, builder, loc, varName, varType, originalVar, needsFree);
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
bool mlir::acc::PointerLikeType::genFree(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> varToFree, ::mlir::Value allocRes, ::mlir::Type varType) const {
      return getImpl()->genFree(getImpl(), *this, builder, loc, varToFree, allocRes, varType);
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
bool mlir::acc::PointerLikeType::genCopy(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destination, ::mlir::TypedValue<::mlir::acc::PointerLikeType> source, ::mlir::Type varType) const {
      return getImpl()->genCopy(getImpl(), *this, builder, loc, destination, source, varType);
  }

/// Generates a load operation from the pointer-like type. This dereferences
/// the pointer and returns the loaded value.
/// 
/// The `srcPtr` parameter is the pointer to load from. If the current type is
/// represented in a way that it does not capture the pointee type, `valueType`
/// must be passed in to provide the necessary type information.
/// 
/// Returns the loaded value, or an empty Value if load generation failed.
::mlir::Value mlir::acc::PointerLikeType::genLoad(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::TypedValue<::mlir::acc::PointerLikeType> srcPtr, ::mlir::Type valueType) const {
      return getImpl()->genLoad(getImpl(), *this, builder, loc, srcPtr, valueType);
  }

/// Generates a store operation to the pointer-like type. This stores a value
/// to the memory location pointed to by the pointer.
/// 
/// The `destPtr` parameter is the pointer to store to. The `valueToStore`
/// parameter is the value to be stored. The type information is derived from
/// the valueToStore parameter itself.
/// 
/// Returns true if store was successfully generated, false otherwise.
bool mlir::acc::PointerLikeType::genStore(::mlir::OpBuilder & builder, ::mlir::Location loc, ::mlir::Value valueToStore, ::mlir::TypedValue<::mlir::acc::PointerLikeType> destPtr) const {
      return getImpl()->genStore(getImpl(), *this, builder, loc, valueToStore, destPtr);
  }

/// Returns true if the pointer points to device data.
bool mlir::acc::PointerLikeType::isDeviceData(::mlir::Value var) const {
      return getImpl()->isDeviceData(getImpl(), *this, var);
  }
