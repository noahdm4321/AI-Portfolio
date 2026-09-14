/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Tosa Op Availability Implementations                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TosaOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> AbsOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> AbsOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> AddOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> AddOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> AddShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> AddShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ApplyScaleOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ApplyScaleOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ArgMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ArgMaxOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ArithmeticRightShiftOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ArithmeticRightShiftOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> AssertEqualShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> AssertEqualShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> AvgPool2dOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> AvgPool2dOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> BitwiseAndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> BitwiseAndOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> BitwiseNotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> BitwiseNotOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> BitwiseOrOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> BitwiseOrOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> BitwiseXorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> BitwiseXorOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> CastFromBlockScaledOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::mxfp}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> CastFromBlockScaledOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> CastOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> CastOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> CastToBlockScaledOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::mxfp}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> CastToBlockScaledOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> CeilOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> CeilOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ClampOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ClampOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ClzOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ClzOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ConcatOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ConcatOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ConcatShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ConcatShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ConstOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int4, ::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::mxfp, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ConstOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ConstShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ConstShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> Conv2DBlockScaledOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::mxfp_conv}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> Conv2DBlockScaledOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> Conv2DOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int4, ::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> Conv2DOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> Conv3DOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int4, ::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> Conv3DOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> CosOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> CosOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> CustomOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> CustomOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> DepthwiseConv2DOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int4, ::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> DepthwiseConv2DOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> DimOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> DimOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> DivCeilShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> DivCeilShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> DivFloorShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> DivFloorShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> EqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> EqualOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ErfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ErfOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> Exp2ShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> Exp2ShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ExpOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ExpOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> FFT2dOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fft}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> FFT2dOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> FloorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> FloorOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> GatherOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> GatherOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> GreaterEqualOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> GreaterEqualOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> GreaterOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> GreaterOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> IdentityOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int4, ::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> IdentityOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> IfOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::controlflow}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> IfOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> IntDivOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> IntDivOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> Log2CeilShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> Log2CeilShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> Log2FloorShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> Log2FloorShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> LogOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> LogOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> LogicalAndOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> LogicalAndOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> LogicalLeftShiftOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> LogicalLeftShiftOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> LogicalNotOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> LogicalNotOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> LogicalOrOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> LogicalOrOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> LogicalRightShiftOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> LogicalRightShiftOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> LogicalXorOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> LogicalXorOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MatMulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MatMulOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MatmulTBlockScaledOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::mxfp}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MatmulTBlockScaledOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MaxPool2dOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MaxPool2dOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MaxShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MaxShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MaximumOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MaximumOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MinShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MinShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MinimumOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MinimumOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ModShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ModShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MulOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MulOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> MulShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> MulShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> NegateOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> NegateOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> PadOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> PadOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> PowOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> PowOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> RFFT2dOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fft}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> RFFT2dOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReciprocalOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReciprocalOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReduceAllOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReduceAllOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReduceAnyOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReduceAnyOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReduceMaxOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReduceMaxOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReduceMinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReduceMinOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReduceProductOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReduceProductOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReduceSumOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReduceSumOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> RescaleOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> RescaleOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReshapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReshapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ResizeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ResizeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ReverseOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ReverseOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> RsqrtOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> RsqrtOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> ScatterOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> ScatterOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> SelectOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> SelectOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> SigmoidOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> SigmoidOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> SinOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> SinOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> SliceOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> SliceOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> SliceShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> SliceShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> SubOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> SubOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> SubShapeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::shape}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> SubShapeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> TableOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> TableOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> TanhOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> TanhOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> TileOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> TileOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> TransposeConv2DOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::int4, ::mlir::tosa::Extension::int16, ::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> TransposeConv2DOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> TransposeOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::fp8e4m3, ::mlir::tosa::Extension::fp8e5m2, ::mlir::tosa::Extension::bf16, ::mlir::tosa::Extension::int64}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> TransposeOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> VariableOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::variable}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> VariableOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> VariableReadOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::variable}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> VariableReadOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> VariableWriteOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::variable}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> VariableWriteOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Profile profs[] = {::mlir::tosa::Profile::pro_int, ::mlir::tosa::Profile::pro_fp}; ArrayRef<::mlir::tosa::Profile> ref(profs, std::size(profs));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> WhileOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  {
    static const ::mlir::tosa::Extension exts[] = {::mlir::tosa::Extension::controlflow}; ArrayRef<::mlir::tosa::Extension> ref(exts, std::size(exts));
    tblgen_overall.emplace_back(ref);
  }
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> WhileOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> YieldOp::getExtensions() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Extension>, 1> tblgen_overall = {};
  return tblgen_overall;
}
::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> YieldOp::getProfiles() {
  ::llvm::SmallVector<::llvm::ArrayRef<::mlir::tosa::Profile>, 1> tblgen_overall = {};
  return tblgen_overall;
}
