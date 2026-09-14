/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonDialect.td                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton {

/// The Triton IR in MLIR
/// Triton Dialect.
/// 
///     Dependent Dialects:
///       * Arith:
///         * addf, addi, andi, cmpf, cmpi, divf, fptosi, ...
///       * Math:
///         * exp, sin, cos, log, ...
///       * StructuredControlFlow:
///         * for, if, while, yield, condition
///       * ControlFlow:
///         * br, cond_br
class TritonDialect : public ::mlir::Dialect {
  explicit TritonDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~TritonDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("tt");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

  /// Materialize a single constant operation from a given attribute value with
  /// the desired resultant type.
  ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc) override;

    /// Helper to manage the discardable attribute `num_stages`.
    class NumStagesAttrHelper {
      ::mlir::StringAttr name;
    public:
      static constexpr ::llvm::StringLiteral getNameStr() {
        return "tt.num_stages";
      }
      constexpr ::mlir::StringAttr getName() {
        return name;
      }

      NumStagesAttrHelper(::mlir::MLIRContext *ctx)
        : name(::mlir::StringAttr::get(ctx, getNameStr())) {}

     ::mlir::IntegerAttr getAttr(::mlir::Operation *op) {
       return op->getAttrOfType<::mlir::IntegerAttr>(name);
     }
     void setAttr(::mlir::Operation *op, ::mlir::IntegerAttr val) {
       op->setAttr(name, val);
     }
     bool isAttrPresent(::mlir::Operation *op) {
       return op->hasAttrOfType<::mlir::IntegerAttr>(name);
     }
     void removeAttr(::mlir::Operation *op) {
       assert(op->hasAttrOfType<::mlir::IntegerAttr>(name));
       op->removeAttr(name);
     }
   };
   NumStagesAttrHelper getNumStagesAttrHelper() {
     return numStagesAttrName;
   }
 private:
   NumStagesAttrHelper numStagesAttrName;
 public:

    /// Helper to manage the discardable attribute `latency`.
    class LatencyAttrHelper {
      ::mlir::StringAttr name;
    public:
      static constexpr ::llvm::StringLiteral getNameStr() {
        return "tt.latency";
      }
      constexpr ::mlir::StringAttr getName() {
        return name;
      }

      LatencyAttrHelper(::mlir::MLIRContext *ctx)
        : name(::mlir::StringAttr::get(ctx, getNameStr())) {}

     ::mlir::IntegerAttr getAttr(::mlir::Operation *op) {
       return op->getAttrOfType<::mlir::IntegerAttr>(name);
     }
     void setAttr(::mlir::Operation *op, ::mlir::IntegerAttr val) {
       op->setAttr(name, val);
     }
     bool isAttrPresent(::mlir::Operation *op) {
       return op->hasAttrOfType<::mlir::IntegerAttr>(name);
     }
     void removeAttr(::mlir::Operation *op) {
       assert(op->hasAttrOfType<::mlir::IntegerAttr>(name));
       op->removeAttr(name);
     }
   };
   LatencyAttrHelper getLatencyAttrHelper() {
     return latencyAttrName;
   }
 private:
   LatencyAttrHelper latencyAttrName;
 public:

    /// Helper to manage the discardable attribute `self_latency`.
    class SelfLatencyAttrHelper {
      ::mlir::StringAttr name;
    public:
      static constexpr ::llvm::StringLiteral getNameStr() {
        return "tt.self_latency";
      }
      constexpr ::mlir::StringAttr getName() {
        return name;
      }

      SelfLatencyAttrHelper(::mlir::MLIRContext *ctx)
        : name(::mlir::StringAttr::get(ctx, getNameStr())) {}

     ::mlir::IntegerAttr getAttr(::mlir::Operation *op) {
       return op->getAttrOfType<::mlir::IntegerAttr>(name);
     }
     void setAttr(::mlir::Operation *op, ::mlir::IntegerAttr val) {
       op->setAttr(name, val);
     }
     bool isAttrPresent(::mlir::Operation *op) {
       return op->hasAttrOfType<::mlir::IntegerAttr>(name);
     }
     void removeAttr(::mlir::Operation *op) {
       assert(op->hasAttrOfType<::mlir::IntegerAttr>(name));
       op->removeAttr(name);
     }
   };
   SelfLatencyAttrHelper getSelfLatencyAttrHelper() {
     return selfLatencyAttrName;
   }
 private:
   SelfLatencyAttrHelper selfLatencyAttrName;
 public:

    void registerTypes();

    static TritonDialect *getLoaded(MLIRContext *ctx) {
      return ctx->getLoadedDialect<TritonDialect>();
    }
    static TritonDialect *getLoaded(Operation *op) {
      return getLoaded(op->getContext());
    }
  };

} // namespace mlir::triton
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::TritonDialect)
