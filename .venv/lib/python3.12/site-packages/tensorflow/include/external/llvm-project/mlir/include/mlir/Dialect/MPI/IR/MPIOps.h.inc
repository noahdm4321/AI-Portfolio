/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MPIOps.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::mpi {

/// Equivalent to `MPI_Allgather(sendbuf, sendcount, sendtype,
///                                  recvbuf, recvcount, recvtype,
///                                  comm)`.
/// MPI_Allgather collects data from all processes in a given communicator and
///     stores the gathered data in the receive buffer of each process.
/// 
///     Each process contributes the same amount of data defined by `sendbuf`.
///     The MPI call specifies the number of elements contributed by each process
///     via the `recvcount` parameter. However, this operation, assumes `recvbuf`
///     to be sufficiently large to hold the data contributed by all processes.
///     Therefore, `recvcount` is implicitly defined as
///     `num_elements(recvbuf) / MPI_Comm_size(comm)`.
/// 
///     This operation may optionally return an !mpi.retval value, which can be
///     used for error checking.
class AllGatherOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Equivalent to `MPI_Allreduce(sendbuf, recvbuf, op, comm)`
/// MPI_Allreduce performs a reduction operation on the values in the sendbuf
///     array and stores the result in the recvbuf array. The operation is 
///     performed across all processes in the communicator.
/// 
///     The `op` attribute specifies the reduction operation to be performed.
///     Currently only the `MPI_Op` predefined in the standard (e.g. `MPI_SUM`) are
///     supported.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class AllReduceOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Equivalent to `MPI_Barrier(comm)`
/// MPI_Barrier blocks execution until all processes in the communicator have
///     reached this routine.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class Barrier;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Get the current rank, equivalent to `MPI_Comm_rank(comm, &rank)`
/// This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class CommRankOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Get the size of the group associated to the communicator, equivalent to `MPI_Comm_size(comm, &size)`
/// This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class CommSizeOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Partition the group associated with the given communicator into disjoint subgroups
/// This operation splits the communicator into multiple sub-communicators.
///     The color value determines the group of processes that will be part of the
///     new communicator. The key value determines the rank of the calling process
///     in the new communicator.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class CommSplitOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Get the World communicator, equivalent to `MPI_COMM_WORLD`
/// This operation returns the predefined MPI_COMM_WORLD communicator.
class CommWorldOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Get the error class from an error code, equivalent to the `MPI_Error_class` function
/// `MPI_Error_class` maps return values from MPI calls to a set of well-known
///     MPI error classes.
class ErrorClassOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Finalize the MPI library, equivalent to `MPI_Finalize()`
/// This function cleans up the MPI state. Afterwards, no MPI methods may 
///     be invoked (excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized).
///     Notably, MPI_Init cannot be called again in the same program.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class FinalizeOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Equivalent to `MPI_Irecv(ptr, size, dtype, source, tag, comm, &req)`
/// MPI_Irecv begins a non-blocking receive of `size` elements of type `dtype` 
///     from rank `source`. The `tag` value and communicator enables the library to
///     determine the matching of multiple sends and receives between the same 
///     ranks.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class IRecvOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Equivalent to `MPI_Isend(ptr, size, dtype, dest, tag, comm)`
/// MPI_Isend begins a non-blocking send of `size` elements of type `dtype` to
///     rank `dest`. The `tag` value and communicator enables the library to
///     determine the matching of multiple sends and receives between the same
///     ranks.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class ISendOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Initialize the MPI library, equivalent to `MPI_Init(NULL, NULL)`
/// This operation must preceed most MPI calls (except for very few exceptions,
///     please consult with the MPI specification on these).
/// 
///     Passing &argc, &argv is not supported currently.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class InitOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Equivalent to `MPI_Recv(ptr, size, dtype, source, tag, comm, MPI_STATUS_IGNORE)`
/// MPI_Recv performs a blocking receive of `size` elements of type `dtype` 
///     from rank `source`. The `tag` value and communicator enables the library to
///     determine the matching of multiple sends and receives between the same 
///     ranks.
/// 
///     The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object 
///     is not yet ported to MLIR.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class RecvOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Check an MPI return value against an error class
/// This operation compares MPI status codes to known error class
///     constants such as `MPI_SUCCESS`, or `MPI_ERR_COMM`.
class RetvalCheckOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Equivalent to `MPI_Send(ptr, size, dtype, dest, tag, comm)`
/// MPI_Send performs a blocking send of `size` elements of type `dtype` to rank
///     `dest`. The `tag` value and communicator enables the library to determine 
///     the matching of multiple sends and receives between the same ranks.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class SendOp;

} // namespace mlir::mpi
namespace mlir::mpi {

/// Equivalent to `MPI_Wait(req, MPI_STATUS_IGNORE)`
/// MPI_Wait blocks execution until the request has completed.
/// 
///     The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object 
///     is not yet ported to MLIR.
/// 
///     This operation can optionally return an `!mpi.retval` value that can be used
///     to check for errors.
class Wait;

} // namespace mlir::mpi
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::AllGatherOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AllGatherOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AllGatherOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.allgather", odsAttrs.getContext());
  }

  AllGatherOpGenericAdaptorBase(AllGatherOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class AllGatherOpGenericAdaptor : public detail::AllGatherOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllGatherOpGenericAdaptorBase;
  static constexpr int odsIndex_sendbuf = 0;
  static constexpr int odsIndex_recvbuf = 1;
  static constexpr int odsIndex_comm = 2;
public:
  AllGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AllGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AllGatherOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AllGatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AllGatherOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  AllGatherOpGenericAdaptor(RangeT values, const AllGatherOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AllGatherOp, typename = std::enable_if_t<std::is_same_v<LateInst, AllGatherOp>>>
  AllGatherOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSendbuf() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRecvbuf() {
    return (*getODSOperands(1).begin());
  }

  ValueT getComm() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllGatherOpAdaptor : public AllGatherOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllGatherOpGenericAdaptor::AllGatherOpGenericAdaptor;
  AllGatherOpAdaptor(AllGatherOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AllGatherOp : public ::mlir::Op<AllGatherOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllGatherOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllGatherOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_sendbuf = 0;
  static constexpr int odsIndex_recvbuf = 1;
  static constexpr int odsIndex_comm = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.allgather");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getSendbuf() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getRecvbuf() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getSendbufMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRecvbufMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::Value comm);
  static AllGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::Value comm);
  static AllGatherOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::Value comm);
  static AllGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::Value comm);
  static AllGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllGatherOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllGatherOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::AllGatherOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::AllReduceOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AllReduceOpGenericAdaptorBase {
public:
  struct Properties {
    using opTy = ::mlir::mpi::MPI_ReductionOpEnumAttr;
    opTy op;

    auto getOp() const {
      auto &propStorage = this->op;
      return ::llvm::cast<::mlir::mpi::MPI_ReductionOpEnumAttr>(propStorage);
    }
    void setOp(const ::mlir::mpi::MPI_ReductionOpEnumAttr &propValue) {
      this->op = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.op == this->op &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AllReduceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.allreduce", odsAttrs.getContext());
  }

  AllReduceOpGenericAdaptorBase(AllReduceOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::mpi::MPI_ReductionOpEnumAttr getOpAttr() {
    auto attr = ::llvm::cast<::mlir::mpi::MPI_ReductionOpEnumAttr>(getProperties().op);
    return attr;
  }

  ::mlir::mpi::MPI_ReductionOpEnum getOp();
};

} // namespace detail
template <typename RangeT>
class AllReduceOpGenericAdaptor : public detail::AllReduceOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllReduceOpGenericAdaptorBase;
  static constexpr int odsIndex_sendbuf = 0;
  static constexpr int odsIndex_recvbuf = 1;
  static constexpr int odsIndex_comm = 2;
public:
  AllReduceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AllReduceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AllReduceOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AllReduceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AllReduceOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  AllReduceOpGenericAdaptor(RangeT values, const AllReduceOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AllReduceOp, typename = std::enable_if_t<std::is_same_v<LateInst, AllReduceOp>>>
  AllReduceOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSendbuf() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRecvbuf() {
    return (*getODSOperands(1).begin());
  }

  ValueT getComm() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllReduceOpAdaptor : public AllReduceOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllReduceOpGenericAdaptor::AllReduceOpGenericAdaptor;
  AllReduceOpAdaptor(AllReduceOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AllReduceOp : public ::mlir::Op<AllReduceOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllReduceOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllReduceOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_sendbuf = 0;
  static constexpr int odsIndex_recvbuf = 1;
  static constexpr int odsIndex_comm = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("op")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOpAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.allreduce");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getSendbuf() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::MemRefType> getRecvbuf() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getSendbufMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRecvbufMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::mpi::MPI_ReductionOpEnumAttr getOpAttr() {
    return ::llvm::cast<::mlir::mpi::MPI_ReductionOpEnumAttr>(getProperties().op);
  }

  ::mlir::mpi::MPI_ReductionOpEnum getOp();
  void setOpAttr(::mlir::mpi::MPI_ReductionOpEnumAttr attr) {
    getProperties().op = attr;
  }

  void setOp(::mlir::mpi::MPI_ReductionOpEnum attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnumAttr op, ::mlir::Value comm);
  static AllReduceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnumAttr op, ::mlir::Value comm);
  static AllReduceOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnumAttr op, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnumAttr op, ::mlir::Value comm);
  static AllReduceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnumAttr op, ::mlir::Value comm);
  static AllReduceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnumAttr op, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnum op, ::mlir::Value comm);
  static AllReduceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnum op, ::mlir::Value comm);
  static AllReduceOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnum op, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnum op, ::mlir::Value comm);
  static AllReduceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnum op, ::mlir::Value comm);
  static AllReduceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value sendbuf, ::mlir::Value recvbuf, ::mlir::mpi::MPI_ReductionOpEnum op, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllReduceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllReduceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllReduceOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllReduceOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::AllReduceOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::Barrier declarations
//===----------------------------------------------------------------------===//

namespace detail {

class BarrierGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  BarrierGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.barrier", odsAttrs.getContext());
  }

  BarrierGenericAdaptorBase(Barrier op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class BarrierGenericAdaptor : public detail::BarrierGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::BarrierGenericAdaptorBase;
  static constexpr int odsIndex_comm = 0;
public:
  BarrierGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  BarrierGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : BarrierGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  BarrierGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : BarrierGenericAdaptor(values, attrs, Properties{}, {}) {}

  BarrierGenericAdaptor(RangeT values, const BarrierGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Barrier, typename = std::enable_if_t<std::is_same_v<LateInst, Barrier>>>
  BarrierGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getComm() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class BarrierAdaptor : public BarrierGenericAdaptor<::mlir::ValueRange> {
public:
  using BarrierGenericAdaptor::BarrierGenericAdaptor;
  BarrierAdaptor(Barrier op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Barrier : public ::mlir::Op<Barrier, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BarrierAdaptor;
  template <typename RangeT>
  using GenericAdaptor = BarrierGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_comm = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.barrier");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Value comm);
  static Barrier create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Value comm);
  static Barrier create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static Barrier create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static Barrier create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Barrier create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Barrier create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Barrier create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Barrier create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::Barrier)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::CommRankOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CommRankOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  CommRankOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.comm_rank", odsAttrs.getContext());
  }

  CommRankOpGenericAdaptorBase(CommRankOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class CommRankOpGenericAdaptor : public detail::CommRankOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CommRankOpGenericAdaptorBase;
  static constexpr int odsIndex_comm = 0;
public:
  CommRankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CommRankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CommRankOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CommRankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CommRankOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CommRankOpGenericAdaptor(RangeT values, const CommRankOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CommRankOp, typename = std::enable_if_t<std::is_same_v<LateInst, CommRankOp>>>
  CommRankOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getComm() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CommRankOpAdaptor : public CommRankOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CommRankOpGenericAdaptor::CommRankOpGenericAdaptor;
  CommRankOpAdaptor(CommRankOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CommRankOp : public ::mlir::Op<CommRankOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CommRankOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CommRankOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_comm = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.comm_rank");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getRank() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(1).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Type rank, ::mlir::Value comm);
  static CommRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Type rank, ::mlir::Value comm);
  static CommRankOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Type rank, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static CommRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static CommRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CommRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CommRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CommRankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CommRankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::CommRankOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::CommSizeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CommSizeOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  CommSizeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.comm_size", odsAttrs.getContext());
  }

  CommSizeOpGenericAdaptorBase(CommSizeOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class CommSizeOpGenericAdaptor : public detail::CommSizeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CommSizeOpGenericAdaptorBase;
  static constexpr int odsIndex_comm = 0;
public:
  CommSizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CommSizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CommSizeOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CommSizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CommSizeOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CommSizeOpGenericAdaptor(RangeT values, const CommSizeOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CommSizeOp, typename = std::enable_if_t<std::is_same_v<LateInst, CommSizeOp>>>
  CommSizeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getComm() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CommSizeOpAdaptor : public CommSizeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CommSizeOpGenericAdaptor::CommSizeOpGenericAdaptor;
  CommSizeOpAdaptor(CommSizeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CommSizeOp : public ::mlir::Op<CommSizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CommSizeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CommSizeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_comm = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.comm_size");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getSize() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(1).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Type size, ::mlir::Value comm);
  static CommSizeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Type size, ::mlir::Value comm);
  static CommSizeOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Type size, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static CommSizeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static CommSizeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CommSizeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CommSizeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CommSizeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CommSizeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::CommSizeOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::CommSplitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CommSplitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  CommSplitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.comm_split", odsAttrs.getContext());
  }

  CommSplitOpGenericAdaptorBase(CommSplitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class CommSplitOpGenericAdaptor : public detail::CommSplitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CommSplitOpGenericAdaptorBase;
  static constexpr int odsIndex_comm = 0;
  static constexpr int odsIndex_color = 1;
  static constexpr int odsIndex_key = 2;
public:
  CommSplitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CommSplitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CommSplitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CommSplitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CommSplitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CommSplitOpGenericAdaptor(RangeT values, const CommSplitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CommSplitOp, typename = std::enable_if_t<std::is_same_v<LateInst, CommSplitOp>>>
  CommSplitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getComm() {
    return (*getODSOperands(0).begin());
  }

  ValueT getColor() {
    return (*getODSOperands(1).begin());
  }

  ValueT getKey() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CommSplitOpAdaptor : public CommSplitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CommSplitOpGenericAdaptor::CommSplitOpGenericAdaptor;
  CommSplitOpAdaptor(CommSplitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CommSplitOp : public ::mlir::Op<CommSplitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CommSplitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CommSplitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_comm = 0;
  static constexpr int odsIndex_color = 1;
  static constexpr int odsIndex_key = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.comm_split");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getColor() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getKey() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getColorMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getKeyMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getNewcomm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSResults(1).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Type newcomm, ::mlir::Value comm, ::mlir::Value color, ::mlir::Value key);
  static CommSplitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Type newcomm, ::mlir::Value comm, ::mlir::Value color, ::mlir::Value key);
  static CommSplitOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Type newcomm, ::mlir::Value comm, ::mlir::Value color, ::mlir::Value key);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value comm, ::mlir::Value color, ::mlir::Value key);
  static CommSplitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value comm, ::mlir::Value color, ::mlir::Value key);
  static CommSplitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value comm, ::mlir::Value color, ::mlir::Value key);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CommSplitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CommSplitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CommSplitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CommSplitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::CommSplitOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::CommWorldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CommWorldOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  CommWorldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.comm_world", odsAttrs.getContext());
  }

  CommWorldOpGenericAdaptorBase(CommWorldOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class CommWorldOpGenericAdaptor : public detail::CommWorldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CommWorldOpGenericAdaptorBase;
public:
  CommWorldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CommWorldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : CommWorldOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CommWorldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CommWorldOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CommWorldOpGenericAdaptor(RangeT values, const CommWorldOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CommWorldOp, typename = std::enable_if_t<std::is_same_v<LateInst, CommWorldOp>>>
  CommWorldOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CommWorldOpAdaptor : public CommWorldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CommWorldOpGenericAdaptor::CommWorldOpGenericAdaptor;
  CommWorldOpAdaptor(CommWorldOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CommWorldOp : public ::mlir::Op<CommWorldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::mpi::CommType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CommWorldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CommWorldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.comm_world");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type comm);
  static CommWorldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type comm);
  static CommWorldOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static CommWorldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static CommWorldOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CommWorldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CommWorldOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CommWorldOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CommWorldOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::CommWorldOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::ErrorClassOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ErrorClassOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ErrorClassOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.error_class", odsAttrs.getContext());
  }

  ErrorClassOpGenericAdaptorBase(ErrorClassOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ErrorClassOpGenericAdaptor : public detail::ErrorClassOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ErrorClassOpGenericAdaptorBase;
  static constexpr int odsIndex_val = 0;
public:
  ErrorClassOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ErrorClassOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ErrorClassOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ErrorClassOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ErrorClassOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ErrorClassOpGenericAdaptor(RangeT values, const ErrorClassOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ErrorClassOp, typename = std::enable_if_t<std::is_same_v<LateInst, ErrorClassOp>>>
  ErrorClassOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVal() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ErrorClassOpAdaptor : public ErrorClassOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ErrorClassOpGenericAdaptor::ErrorClassOpGenericAdaptor;
  ErrorClassOpAdaptor(ErrorClassOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ErrorClassOp : public ::mlir::Op<ErrorClassOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::mpi::RetvalType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ErrorClassOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ErrorClassOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_val = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.error_class");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getVal() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getValMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getErrclass() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type errclass, ::mlir::Value val);
  static ErrorClassOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type errclass, ::mlir::Value val);
  static ErrorClassOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type errclass, ::mlir::Value val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static ErrorClassOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static ErrorClassOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value val);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ErrorClassOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ErrorClassOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ErrorClassOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ErrorClassOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::ErrorClassOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::FinalizeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class FinalizeOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  FinalizeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.finalize", odsAttrs.getContext());
  }

  FinalizeOpGenericAdaptorBase(FinalizeOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class FinalizeOpGenericAdaptor : public detail::FinalizeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FinalizeOpGenericAdaptorBase;
public:
  FinalizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  FinalizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : FinalizeOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  FinalizeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : FinalizeOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  FinalizeOpGenericAdaptor(RangeT values, const FinalizeOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = FinalizeOp, typename = std::enable_if_t<std::is_same_v<LateInst, FinalizeOp>>>
  FinalizeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FinalizeOpAdaptor : public FinalizeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FinalizeOpGenericAdaptor::FinalizeOpGenericAdaptor;
  FinalizeOpAdaptor(FinalizeOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class FinalizeOp : public ::mlir::Op<FinalizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FinalizeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FinalizeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.finalize");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval);
  static FinalizeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval);
  static FinalizeOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static FinalizeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static FinalizeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FinalizeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static FinalizeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static FinalizeOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static FinalizeOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::FinalizeOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::IRecvOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class IRecvOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  IRecvOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.irecv", odsAttrs.getContext());
  }

  IRecvOpGenericAdaptorBase(IRecvOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class IRecvOpGenericAdaptor : public detail::IRecvOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::IRecvOpGenericAdaptorBase;
  static constexpr int odsIndex_ref = 0;
  static constexpr int odsIndex_tag = 1;
  static constexpr int odsIndex_source = 2;
  static constexpr int odsIndex_comm = 3;
public:
  IRecvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  IRecvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : IRecvOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  IRecvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : IRecvOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  IRecvOpGenericAdaptor(RangeT values, const IRecvOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = IRecvOp, typename = std::enable_if_t<std::is_same_v<LateInst, IRecvOp>>>
  IRecvOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRef() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTag() {
    return (*getODSOperands(1).begin());
  }

  ValueT getSource() {
    return (*getODSOperands(2).begin());
  }

  ValueT getComm() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class IRecvOpAdaptor : public IRecvOpGenericAdaptor<::mlir::ValueRange> {
public:
  using IRecvOpGenericAdaptor::IRecvOpGenericAdaptor;
  IRecvOpAdaptor(IRecvOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class IRecvOp : public ::mlir::Op<IRecvOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IRecvOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = IRecvOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ref = 0;
  static constexpr int odsIndex_tag = 1;
  static constexpr int odsIndex_source = 2;
  static constexpr int odsIndex_comm = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.irecv");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getRef() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getTag() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getSource() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getRefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTagMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getSourceMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  ::mlir::TypedValue<::mlir::mpi::RequestType> getReq() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RequestType>>(*getODSResults(1).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Type req, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static IRecvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Type req, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static IRecvOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Type req, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static IRecvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static IRecvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static IRecvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static IRecvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static IRecvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static IRecvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::IRecvOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::ISendOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ISendOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ISendOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.isend", odsAttrs.getContext());
  }

  ISendOpGenericAdaptorBase(ISendOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ISendOpGenericAdaptor : public detail::ISendOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ISendOpGenericAdaptorBase;
  static constexpr int odsIndex_ref = 0;
  static constexpr int odsIndex_tag = 1;
  static constexpr int odsIndex_dest = 2;
  static constexpr int odsIndex_comm = 3;
public:
  ISendOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ISendOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ISendOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ISendOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ISendOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ISendOpGenericAdaptor(RangeT values, const ISendOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ISendOp, typename = std::enable_if_t<std::is_same_v<LateInst, ISendOp>>>
  ISendOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRef() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTag() {
    return (*getODSOperands(1).begin());
  }

  ValueT getDest() {
    return (*getODSOperands(2).begin());
  }

  ValueT getComm() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ISendOpAdaptor : public ISendOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ISendOpGenericAdaptor::ISendOpGenericAdaptor;
  ISendOpAdaptor(ISendOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ISendOp : public ::mlir::Op<ISendOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ISendOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ISendOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ref = 0;
  static constexpr int odsIndex_tag = 1;
  static constexpr int odsIndex_dest = 2;
  static constexpr int odsIndex_comm = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.isend");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getRef() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getTag() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getDest() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getRefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTagMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDestMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  ::mlir::TypedValue<::mlir::mpi::RequestType> getReq() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RequestType>>(*getODSResults(1).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Type req, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static ISendOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Type req, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static ISendOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Type req, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static ISendOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static ISendOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ISendOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ISendOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ISendOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ISendOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::ISendOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::InitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class InitOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  InitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.init", odsAttrs.getContext());
  }

  InitOpGenericAdaptorBase(InitOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class InitOpGenericAdaptor : public detail::InitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InitOpGenericAdaptorBase;
public:
  InitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  InitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : InitOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  InitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : InitOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  InitOpGenericAdaptor(RangeT values, const InitOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = InitOp, typename = std::enable_if_t<std::is_same_v<LateInst, InitOp>>>
  InitOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InitOpAdaptor : public InitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InitOpGenericAdaptor::InitOpGenericAdaptor;
  InitOpAdaptor(InitOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class InitOp : public ::mlir::Op<InitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.init");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval);
  static InitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval);
  static InitOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static InitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes);
  static InitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static InitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static InitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static InitOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static InitOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::InitOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::RecvOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RecvOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  RecvOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.recv", odsAttrs.getContext());
  }

  RecvOpGenericAdaptorBase(RecvOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class RecvOpGenericAdaptor : public detail::RecvOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RecvOpGenericAdaptorBase;
  static constexpr int odsIndex_ref = 0;
  static constexpr int odsIndex_tag = 1;
  static constexpr int odsIndex_source = 2;
  static constexpr int odsIndex_comm = 3;
public:
  RecvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RecvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : RecvOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RecvOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RecvOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  RecvOpGenericAdaptor(RangeT values, const RecvOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = RecvOp, typename = std::enable_if_t<std::is_same_v<LateInst, RecvOp>>>
  RecvOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRef() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTag() {
    return (*getODSOperands(1).begin());
  }

  ValueT getSource() {
    return (*getODSOperands(2).begin());
  }

  ValueT getComm() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RecvOpAdaptor : public RecvOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RecvOpGenericAdaptor::RecvOpGenericAdaptor;
  RecvOpAdaptor(RecvOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RecvOp : public ::mlir::Op<RecvOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RecvOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RecvOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ref = 0;
  static constexpr int odsIndex_tag = 1;
  static constexpr int odsIndex_source = 2;
  static constexpr int odsIndex_comm = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.recv");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getRef() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getTag() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getSource() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getRefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTagMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getSourceMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static RecvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static RecvOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static RecvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static RecvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value source, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RecvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RecvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RecvOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RecvOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::RecvOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::RetvalCheckOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RetvalCheckOpGenericAdaptorBase {
public:
  struct Properties {
    using errclassTy = ::mlir::mpi::MPI_ErrorClassEnumAttr;
    errclassTy errclass;

    auto getErrclass() const {
      auto &propStorage = this->errclass;
      return ::llvm::cast<::mlir::mpi::MPI_ErrorClassEnumAttr>(propStorage);
    }
    void setErrclass(const ::mlir::mpi::MPI_ErrorClassEnumAttr &propValue) {
      this->errclass = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.errclass == this->errclass &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  RetvalCheckOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.retval_check", odsAttrs.getContext());
  }

  RetvalCheckOpGenericAdaptorBase(RetvalCheckOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::mpi::MPI_ErrorClassEnumAttr getErrclassAttr() {
    auto attr = ::llvm::cast<::mlir::mpi::MPI_ErrorClassEnumAttr>(getProperties().errclass);
    return attr;
  }

  ::mlir::mpi::MPI_ErrorClassEnum getErrclass();
};

} // namespace detail
template <typename RangeT>
class RetvalCheckOpGenericAdaptor : public detail::RetvalCheckOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RetvalCheckOpGenericAdaptorBase;
  static constexpr int odsIndex_val = 0;
public:
  RetvalCheckOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RetvalCheckOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : RetvalCheckOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RetvalCheckOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RetvalCheckOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  RetvalCheckOpGenericAdaptor(RangeT values, const RetvalCheckOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = RetvalCheckOp, typename = std::enable_if_t<std::is_same_v<LateInst, RetvalCheckOp>>>
  RetvalCheckOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getVal() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RetvalCheckOpAdaptor : public RetvalCheckOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RetvalCheckOpGenericAdaptor::RetvalCheckOpGenericAdaptor;
  RetvalCheckOpAdaptor(RetvalCheckOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RetvalCheckOp : public ::mlir::Op<RetvalCheckOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RetvalCheckOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RetvalCheckOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_val = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("errclass")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getErrclassAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getErrclassAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.retval_check");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getVal() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getValMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getRes() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::mpi::MPI_ErrorClassEnumAttr getErrclassAttr() {
    return ::llvm::cast<::mlir::mpi::MPI_ErrorClassEnumAttr>(getProperties().errclass);
  }

  ::mlir::mpi::MPI_ErrorClassEnum getErrclass();
  void setErrclassAttr(::mlir::mpi::MPI_ErrorClassEnumAttr attr) {
    getProperties().errclass = attr;
  }

  void setErrclass(::mlir::mpi::MPI_ErrorClassEnum attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnumAttr errclass);
  static RetvalCheckOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type res, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnumAttr errclass);
  static RetvalCheckOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type res, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnumAttr errclass);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnumAttr errclass);
  static RetvalCheckOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnumAttr errclass);
  static RetvalCheckOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnumAttr errclass);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnum errclass);
  static RetvalCheckOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type res, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnum errclass);
  static RetvalCheckOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type res, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnum errclass);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnum errclass);
  static RetvalCheckOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnum errclass);
  static RetvalCheckOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::mpi::MPI_ErrorClassEnum errclass);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RetvalCheckOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RetvalCheckOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RetvalCheckOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RetvalCheckOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::RetvalCheckOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::SendOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SendOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  SendOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.send", odsAttrs.getContext());
  }

  SendOpGenericAdaptorBase(SendOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class SendOpGenericAdaptor : public detail::SendOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SendOpGenericAdaptorBase;
  static constexpr int odsIndex_ref = 0;
  static constexpr int odsIndex_tag = 1;
  static constexpr int odsIndex_dest = 2;
  static constexpr int odsIndex_comm = 3;
public:
  SendOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SendOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : SendOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  SendOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : SendOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  SendOpGenericAdaptor(RangeT values, const SendOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SendOp, typename = std::enable_if_t<std::is_same_v<LateInst, SendOp>>>
  SendOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRef() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTag() {
    return (*getODSOperands(1).begin());
  }

  ValueT getDest() {
    return (*getODSOperands(2).begin());
  }

  ValueT getComm() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SendOpAdaptor : public SendOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SendOpGenericAdaptor::SendOpGenericAdaptor;
  SendOpAdaptor(SendOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SendOp : public ::mlir::Op<SendOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SendOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SendOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_ref = 0;
  static constexpr int odsIndex_tag = 1;
  static constexpr int odsIndex_dest = 2;
  static constexpr int odsIndex_comm = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.send");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::MemRefType> getRef() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getTag() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getDest() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::mpi::CommType> getComm() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::CommType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getRefMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTagMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDestMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getCommMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static SendOp create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static SendOp create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static SendOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static SendOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value ref, ::mlir::Value tag, ::mlir::Value dest, ::mlir::Value comm);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SendOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SendOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static SendOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static SendOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::SendOp)

namespace mlir::mpi {


//===----------------------------------------------------------------------===//
// ::mlir::mpi::Wait declarations
//===----------------------------------------------------------------------===//

namespace detail {

class WaitGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  WaitGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("mpi.wait", odsAttrs.getContext());
  }

  WaitGenericAdaptorBase(Wait op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class WaitGenericAdaptor : public detail::WaitGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WaitGenericAdaptorBase;
  static constexpr int odsIndex_req = 0;
public:
  WaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  WaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : WaitGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  WaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : WaitGenericAdaptor(values, attrs, Properties{}, {}) {}

  WaitGenericAdaptor(RangeT values, const WaitGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Wait, typename = std::enable_if_t<std::is_same_v<LateInst, Wait>>>
  WaitGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getReq() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WaitAdaptor : public WaitGenericAdaptor<::mlir::ValueRange> {
public:
  using WaitGenericAdaptor::WaitGenericAdaptor;
  WaitAdaptor(Wait op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Wait : public ::mlir::Op<Wait, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WaitAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WaitGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_req = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("mpi.wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RequestType> getReq() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RequestType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getReqMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::mpi::RetvalType> getRetval() {
    auto results = getODSResults(0);
    return results.empty() ? ::mlir::TypedValue<::mlir::mpi::RetvalType>() : ::llvm::cast<::mlir::TypedValue<::mlir::mpi::RetvalType>>(*results.begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type retval, ::mlir::Value req);
  static Wait create(::mlir::OpBuilder &builder, ::mlir::Location location, /*optional*/::mlir::Type retval, ::mlir::Value req);
  static Wait create(::mlir::ImplicitLocOpBuilder &builder, /*optional*/::mlir::Type retval, ::mlir::Value req);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value req);
  static Wait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value req);
  static Wait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value req);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Wait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Wait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Wait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Wait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::mpi
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mpi::Wait)


#endif // GET_OP_CLASSES

