/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::tosa {

/// Attribute for UnaryOp quantization information.
class UnaryOpQuantizationAttr;
/// Attribute for Conv type op quantization information.
class ConvOpQuantizationAttr;
/// Attribute for MatMulOp quantization information.
class MatMulOpQuantizationAttr;
/// Attribute for PadOp quantization information.
class PadOpQuantizationAttr;
/// supported TOSA profiles
class ProfileAttr;
/// supported TOSA extensions
class ExtensionAttr;
/// supported TOSA levels
class LevelAttr;
/// TOSA specification version
class SpecificationVersionAttr;
/// Target environment information.
class TargetEnvAttr;
/// Supported resize/upsampling strategies
class ResizeModeAttr;
/// Supported NaN propagation strategies
class NanPropagationModeAttr;
/// Supported rounding modes
class RoundingModeAttr;
/// Block size for the block_scaled formats
class BlockSizeAttr;
namespace detail {

struct UnaryOpQuantizationAttrStorage;

} // namespace detail
class UnaryOpQuantizationAttr : public ::mlir::Attribute::AttrBase<UnaryOpQuantizationAttr, ::mlir::Attribute, detail::UnaryOpQuantizationAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.unary_quant";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static UnaryOpQuantizationAttr get(::mlir::MLIRContext *context, int64_t input_zp, int64_t output_zp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"unary_quant"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getInputZp() const;
  int64_t getOutputZp() const;
};
namespace detail {

struct ConvOpQuantizationAttrStorage;

} // namespace detail
class ConvOpQuantizationAttr : public ::mlir::Attribute::AttrBase<ConvOpQuantizationAttr, ::mlir::Attribute, detail::ConvOpQuantizationAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.conv_quant";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static ConvOpQuantizationAttr get(::mlir::MLIRContext *context, int64_t input_zp, int64_t weight_zp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"conv_quant"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getInputZp() const;
  int64_t getWeightZp() const;
};
namespace detail {

struct MatMulOpQuantizationAttrStorage;

} // namespace detail
class MatMulOpQuantizationAttr : public ::mlir::Attribute::AttrBase<MatMulOpQuantizationAttr, ::mlir::Attribute, detail::MatMulOpQuantizationAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.matmul_quant";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static MatMulOpQuantizationAttr get(::mlir::MLIRContext *context, int64_t a_zp, int64_t b_zp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"matmul_quant"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getAZp() const;
  int64_t getBZp() const;
};
namespace detail {

struct PadOpQuantizationAttrStorage;

} // namespace detail
class PadOpQuantizationAttr : public ::mlir::Attribute::AttrBase<PadOpQuantizationAttr, ::mlir::Attribute, detail::PadOpQuantizationAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.pad_quant";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static PadOpQuantizationAttr get(::mlir::MLIRContext *context, int64_t input_zp);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"pad_quant"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int64_t getInputZp() const;
};
namespace detail {

struct ProfileAttrStorage;

} // namespace detail
class ProfileAttr : public ::mlir::Attribute::AttrBase<ProfileAttr, ::mlir::Attribute, detail::ProfileAttrStorage> {
public:
  using Base::Base;
  static llvm::SmallVector<Profile, 2> getAllValues() {
    return {Profile::pro_int, Profile::pro_fp};
  }
  static constexpr ::llvm::StringLiteral name = "tosa.prof";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static ProfileAttr get(::mlir::MLIRContext *context, ::mlir::tosa::Profile value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"prof"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::tosa::Profile getValue() const;
};
namespace detail {

struct ExtensionAttrStorage;

} // namespace detail
class ExtensionAttr : public ::mlir::Attribute::AttrBase<ExtensionAttr, ::mlir::Attribute, detail::ExtensionAttrStorage> {
public:
  using Base::Base;
  static llvm::SmallVector<Extension, 14> getAllValues() {
    return {
      Extension::int16, Extension::int4, Extension::bf16,
      Extension::fp8e4m3, Extension::fp8e5m2, Extension::fft,
      Extension::variable, Extension::controlflow, Extension::doubleround,
      Extension::inexactround, Extension::dynamic, Extension::mxfp,
      Extension::int64, Extension::mxfp_conv, Extension::shape
    };
  }
  static constexpr ::llvm::StringLiteral name = "tosa.ext";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static ExtensionAttr get(::mlir::MLIRContext *context, ::mlir::tosa::Extension value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ext"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::tosa::Extension getValue() const;
};
namespace detail {

struct LevelAttrStorage;

} // namespace detail
class LevelAttr : public ::mlir::Attribute::AttrBase<LevelAttr, ::mlir::Attribute, detail::LevelAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.level";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static LevelAttr get(::mlir::MLIRContext *context, ::mlir::tosa::Level value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"level"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::tosa::Level getValue() const;
};
namespace detail {

struct SpecificationVersionAttrStorage;

} // namespace detail
class SpecificationVersionAttr : public ::mlir::Attribute::AttrBase<SpecificationVersionAttr, ::mlir::Attribute, detail::SpecificationVersionAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.specification_version";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static SpecificationVersionAttr get(::mlir::MLIRContext *context, ::mlir::tosa::SpecificationVersion value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"specification_version"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::tosa::SpecificationVersion getValue() const;
};
namespace detail {

struct TargetEnvAttrStorage;

} // namespace detail
class TargetEnvAttr : public ::mlir::Attribute::AttrBase<TargetEnvAttr, ::mlir::Attribute, detail::TargetEnvAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.target_env";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static TargetEnvAttr get(::mlir::MLIRContext *context, SpecificationVersion specification_version, Level level, ::llvm::ArrayRef<Profile> profiles, ::llvm::ArrayRef<Extension> extensions);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"target_env"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  SpecificationVersion getSpecificationVersion() const;
  Level getLevel() const;
  ::llvm::ArrayRef<Profile> getProfiles() const;
  ::llvm::ArrayRef<Extension> getExtensions() const;
};
namespace detail {

struct ResizeModeAttrStorage;

} // namespace detail
class ResizeModeAttr : public ::mlir::Attribute::AttrBase<ResizeModeAttr, ::mlir::Attribute, detail::ResizeModeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.resize_mode";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static ResizeModeAttr get(::mlir::MLIRContext *context, ::mlir::tosa::ResizeMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"resize_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::tosa::ResizeMode getValue() const;
};
namespace detail {

struct NanPropagationModeAttrStorage;

} // namespace detail
class NanPropagationModeAttr : public ::mlir::Attribute::AttrBase<NanPropagationModeAttr, ::mlir::Attribute, detail::NanPropagationModeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.nan_mode";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static NanPropagationModeAttr get(::mlir::MLIRContext *context, ::mlir::tosa::NanPropagationMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"nan_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::tosa::NanPropagationMode getValue() const;
};
namespace detail {

struct RoundingModeAttrStorage;

} // namespace detail
class RoundingModeAttr : public ::mlir::Attribute::AttrBase<RoundingModeAttr, ::mlir::Attribute, detail::RoundingModeAttrStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tosa.rounding_mode";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static RoundingModeAttr get(::mlir::MLIRContext *context, ::mlir::tosa::RoundingMode value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"rounding_mode"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::tosa::RoundingMode getValue() const;
};
namespace detail {

struct BlockSizeAttrStorage;

} // namespace detail
class BlockSizeAttr : public ::mlir::Attribute::AttrBase<BlockSizeAttr, ::mlir::Attribute, detail::BlockSizeAttrStorage> {
public:
  using Base::Base;
  static uint32_t getBlockSizeValue(BlockSize blockSize) {
    return static_cast<uint32_t>(blockSize);
  }
  static constexpr ::llvm::StringLiteral name = "tosa.block_size";
  static constexpr ::llvm::StringLiteral dialectName = "tosa";
  static BlockSizeAttr get(::mlir::MLIRContext *context, ::mlir::tosa::BlockSize value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"block_size"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::tosa::BlockSize getValue() const;
};

} // namespace mlir::tosa
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::UnaryOpQuantizationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::ConvOpQuantizationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::MatMulOpQuantizationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::PadOpQuantizationAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::ProfileAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::ExtensionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::LevelAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::SpecificationVersionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::TargetEnvAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::ResizeModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::NanPropagationModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::RoundingModeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::tosa::BlockSizeAttr)

#endif // GET_ATTRDEF_CLASSES

