/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


/// Get the symbol reference to the global
::mlir::SymbolRefAttr mlir::acc::AddressOfGlobalOpInterface::getSymbol() {
      return getImpl()->getSymbol(getImpl(), getOperation());
  }

/// Get alloca block
::mlir::Block*mlir::acc::ComputeRegionOpInterface::getAllocaBlock() {
      return getImpl()->getAllocaBlock(getImpl(), getOperation());
  }

/// Check if the global variable is constant
bool mlir::acc::GlobalVariableOpInterface::isConstant() {
      return getImpl()->isConstant(getImpl(), getOperation());
  }

/// Get the initialization region (returns nullptr if none)
::mlir::Region*mlir::acc::GlobalVariableOpInterface::getInitRegion() {
      return getImpl()->getInitRegion(getImpl(), getOperation());
  }

/// Check if the global variable is device data
bool mlir::acc::GlobalVariableOpInterface::isDeviceData() {
      return getImpl()->isDeviceData(getImpl(), getOperation());
  }

/// Get the symbols referenced by this operation
void mlir::acc::IndirectGlobalAccessOpInterface::getReferencedSymbols(::llvm::SmallVectorImpl<::mlir::SymbolRefAttr>& symbols, ::mlir::SymbolTable * symbolTable) {
      return getImpl()->getReferencedSymbols(getImpl(), getOperation(), symbols, symbolTable);
  }

/// Get the base entity being accessed
::mlir::Value mlir::acc::PartialEntityAccessOpInterface::getBaseEntity() {
      return getImpl()->getBaseEntity(getImpl(), getOperation());
  }

/// Check if this is a complete view of the entity
bool mlir::acc::PartialEntityAccessOpInterface::isCompleteView() {
      return getImpl()->isCompleteView(getImpl(), getOperation());
  }
