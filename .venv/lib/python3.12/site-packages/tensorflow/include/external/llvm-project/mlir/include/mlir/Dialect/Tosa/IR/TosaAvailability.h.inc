/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Availability Interface Declarations                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TosaOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::tosa {

class QueryProfileInterface;

namespace detail {
struct QueryProfileInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> getProfiles(const Concept *impl, Operation *tblgen_opaque_op) const = 0;
  };

  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = QueryProfileInterface;
    ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> getProfiles(const Concept *impl, Operation *tblgen_opaque_op) const final {
      auto op = llvm::cast<ConcreteOp>(tblgen_opaque_op);
      (void)op;
      return op.getProfiles();
    }
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = QueryProfileInterface;
    ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> getProfiles(const Concept *impl, Operation *tblgen_opaque_op) const final {
      auto op = llvm::cast<ConcreteOp>(tblgen_opaque_op);
      (void)op;
      return op.getProfiles();
    }
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteOp> {};
};
} // namespace detail

/*
Querying interface for the supported set of Tosa profile.

    This interface provides a `getProfiles()` method to query
    the supported set of Tosa profile. The returned value is a
    list of `mlir::Tosa::Profile` enum number.
*/
class QueryProfileInterface : public OpInterface<QueryProfileInterface, detail::QueryProfileInterfaceTraits> {
public:
  using OpInterface<QueryProfileInterface, detail::QueryProfileInterfaceTraits>::OpInterface;
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> getProfiles();
};


} // namespace mlir::tosa
namespace mlir::tosa {

class QueryExtensionInterface;

namespace detail {
struct QueryExtensionInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> getExtensions(const Concept *impl, Operation *tblgen_opaque_op) const = 0;
  };

  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = QueryExtensionInterface;
    ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> getExtensions(const Concept *impl, Operation *tblgen_opaque_op) const final {
      auto op = llvm::cast<ConcreteOp>(tblgen_opaque_op);
      (void)op;
      return op.getExtensions();
    }
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = QueryExtensionInterface;
    ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> getExtensions(const Concept *impl, Operation *tblgen_opaque_op) const final {
      auto op = llvm::cast<ConcreteOp>(tblgen_opaque_op);
      (void)op;
      return op.getExtensions();
    }
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteOp> {};
};
} // namespace detail

/*
Querying interface for the supported set of TOSA extension.

    This interface provides a `getExtensions()` method to query
    the supported set of Tosa extension. The returned value is a
    list of `mlir::Tosa::Extension` enum number.
*/
class QueryExtensionInterface : public OpInterface<QueryExtensionInterface, detail::QueryExtensionInterfaceTraits> {
public:
  using OpInterface<QueryExtensionInterface, detail::QueryExtensionInterfaceTraits>::OpInterface;
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> getExtensions();
};


} // namespace mlir::tosa
