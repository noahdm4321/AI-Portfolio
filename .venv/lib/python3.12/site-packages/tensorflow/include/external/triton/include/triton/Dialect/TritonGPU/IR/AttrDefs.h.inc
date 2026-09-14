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
namespace mlir::triton::gpu {

/// An encoding for tensors whose elements may be simultaneously accessed by
/// different GPU threads in the programs, via shared memory. In other words,
/// for all indices i \in Z^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.
/// 
/// In order to avoid shared memory bank conflicts, elements may be swizzled.
/// Here are some examples.  In all cases, the input tensor is [0, 1, ..., n-1].
/// 
/// 1. Basic swizzling
/// 
///   #ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>
///   [ 0,  1,  2,  3],  // xor with 0
///   [ 5,  4,  7,  6],  // xor with 1
///   [10, 11,  8,  9],  // xor with 2
///   [15, 14, 13, 12]   // xor with 3
/// 
/// Here elements of row r are xor'ed with r (or more properly, in[r][c] ->
/// out[r][c^r]).
/// 
/// 2. Multiple rows per phase
/// 
///   #ttg.swizzled_shared<{vec=1, perPhase=2, maxPhase=4, order=[1,0]}>
///   [ 0,  1,  2,  3],  // phase 0 (xor with 0)
///   [ 4,  5,  6,  7],
///   [ 9,  8, 11, 10],  // phase 1 (xor with 1)
///   [13, 12, 15, 14]
/// 
/// Elements of row r are xor'ed with r/2.  In other words, perPhase=2
/// means that pairs of 2 rows get the same swizzling.
/// 
/// 3. Max-phase applied
/// 
///   #ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=2, order=[1,0]}>
///   [ 0,  1,  2,  3],  // phase 0 (xor with 0)
///   [ 5,  4,  7,  6],  // phase 1 (xor with 1)
///   [ 8,  9, 10, 11],  // phase 0
///   [13, 12, 15, 14],  // phase 1
///   [16, 17, 18, 19],  // ...
///   [21, 20, 23, 22],
///   [24, 25, 26, 27],
///   [29, 28, 31, 30]
/// 
/// Elements of row r are xor'ed with (r/2) % 2.  In other words, maxPhase=m has the
/// effect of limiting the maximum value of the xor to m-1.
/// 
/// 4. Max-phase and per-phase
/// 
///   #ttg.swizzled_shared<{vec=1, perPhase=2, maxPhase=2, order=[1,0]}>
///   [ 0,  1,  2,  3],  // phase 0 (xor with 0)
///   [ 4,  5,  6,  7],  // phase 0
///   [ 9,  8, 11, 10],  // phase 1 (xor with 1)
///   [13, 12, 15, 14],  // phase 1
///   [16, 17, 18, 19],  // phase 0
///   [20, 21, 22, 23],  // phase 0
///   [25, 24, 27, 26],  // phase 1
///   [29, 28, 31, 30]]  // phase 1
/// 
/// Here the xor value (the "phase", I guess?) changes every perPhase rows, up to a
/// maximum value of maxPhase-1.  In other words, elements of row r are xor'ed with
/// (r/2) % 2.
/// 
/// 5. Adding vec
/// 
///   #ttg.swizzled_shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0]}>
///   [ 0,  1,  2,  3,  4,  5,  6,  7],
///   [10, 11,  8,  9, 14, 15, 12, 13],
///   [20, 21, 22, 23, 16, 17, 18, 19],
///   [30, 31, 28, 29, 26, 27, 24, 25]
/// 
/// When vec=2, elements are swizzled in pairs of 2.  In other words, the element at
/// (r,c) has value
/// 
///   ((c / 2) ^ r) * 2 + (c % 2).
class SwizzledSharedEncodingAttr;
/// An encoding for tensors whose elements may be simultaneously accessed by
/// different GPU threads in the programs, via shared memory. In other words,
/// for all indices i \in Z^d, \mathcal{L}(i) = {0, 1, ..., 32*num_warps - 1}.
/// Compared to SwizzledSharedEncodingAttr, this encoding combines padding with
/// element reordering via linear transformation (e.g. row permutation) to avoid
/// shared memory bank conflicts.
/// 
/// Formally, given a layout:
///     padded_shared<[<interval_0>:+<pad_0>, <interval_1>:+<pad_1>, ...]>
/// We insert a padding of `<pad_i>` elements after every `<interval_i>` elements.
/// Multi interval-padding pairs are supported for flexibility of multi tiered
/// padding schemes; they compose in an additive manner. So for a 1-D tensor element
/// at index i, the corresponding shared memory location index is
///     i + \sum_{k} (i / interval_k) * pad_k = 1
/// `<interval_i>` and `<pad_i>` all need to be power of two.
/// 
/// Some concrete examples ignoring the linear component, using `eM` to mean tensor
/// elements and `pN` to mean padding:
/// 
/// 1. Single interval-padding pair:
/// 
///    #ttg.padded_shared<[2:+2], {...}>
///    [e0, e1, p0, p1,
///     e2, e3, p2, p3,
///     ...]
/// 
/// 2. Double interval-padding pairs:
/// 
///    #ttg.padded_shared<[2:+1, 4:+2], {...}>
///    [e0, e1, p0,
///     e2, e3, p1, p2, p3,
///     e4, e5, p4,
///     e6, e7, p5, p6, p7,
///     ...]
/// 
/// Furthermore this encoding allows for a linear remapping from the 1-D shared
/// memory offset to logical n-D tensor elements. The remapping is given in the form
/// of linear bases mapping from offset to [dim0, dim1...dimN-1].
/// See LinearLayout.h for more details how linear layouts are applied to remap
/// elements.
/// Some concrete examples using `xN` and `yN` to mean the logical n-D tensor elements
/// and `pN` to mean padding:
/// 
/// 1. 1D Single interval-padding with strided elements
/// 
///     #ttg.padded_shared<[2:+2] {offset = [[2], [1]], block = []}>
///     [x0, x2, p0 p1,
///      x1, x3, p2, p3
///      ...]
/// 
/// 2. 2D single interval-padding with rearranged rows.
/// 
///     #ttg.padded_shared<[16:+1] {offset = [[0, 1], [0, 2], /*gap, stride by 2 rows*/[2, 0], [4, 0], [1, 0]]], block = []}>
///     [
///       x0y0, x0y1, x0y2, x0y3,
///       x2y0, x2y1, x2y2, x2y3,
///       x4y0, x4y1, x4y2, x4y3,
///       x6y0, x6y1, x6y2, x6y3,
///       p0,
///       x1y0, x1y1, x1y2, x1y3,
///       x3y0, x3y1, x3y2, x3y3,
///       x5y0, x5y1, x5y2, x5y3,
///       x7y0, x7y1, x7y2, x7y3,
///       p1,
///     ]
/// 
/// For identity mappings a short form based on order and shape is used to increase readability. The following two encodings are the same:
/// 
///     #ttg.padded_shared<[2:+2] {order = [1, 0], shape = [16, 32]}>
///     #ttg.padded_shared<[2:+2] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0]], block = []}>
class PaddedSharedEncodingAttr;
/// Linear shared encodings mirror LinearEncodingAttr but operate on shared
///     memory layouts. The LinearLayout parameter captures how shared memory
///     offsets (and optionally blocks) map to logical tensor indices.
class SharedLinearEncodingAttr;
/// Represent blocked shared memory matching MMAv3/MMAv5 shared memory input.
///     This is meant to represent 2d tiled blocked layout.
///     The full layout representation is described here:
///     https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout
///     When the memdesc has more than 2 dimensions the tiling is applied to 8 rows even if the first outer dimension is smaller than 8.
///     In this case `transposed` means that the contiguous dimension is the most outer dimension of the memdesc.
class NVMMASharedEncodingAttr;
/// This shared encoding is similar to SwizzledSharedEncodingAttr, but instead of
/// repeating swizzling pattern every `maxPhase*perPhase` rows of the memory object,
/// called a block, this layout changes swizzling pattern `maxPhase` times, then
/// repeats the pattern. The name "rotating" comes from the fact that first tensor
/// element of each block is swizzled with different phase, which is equal to
/// current block number: 0, 1, 2.. maxPhase-1, 0, 1, 2 ...
/// 
/// This layout is used to reduce bank conflicts in cases where shared memory writes
/// and reads are performed on layouts with different order. It's meant for hardware
/// without native shared memory tranpose support.
/// 
/// Swizzling pattern affects only 2 fastest dimensions of a tensor.
/// In the following text these two dimensions are called row and column:
/// - row is a fastest dimension
/// - column is a second fastest dimension
/// 
/// Elements in a row dimension are stored in memory contiguously.
/// 
/// If a matrix of size [128x64] is stored in this shared layout with order [1, 0],
/// dim 1 (64) will be stored contiguously and called row, dim 0 (128) is will be
/// called column. If order of shared layout is [0, 1], dim 0 (128) is stored
/// contiguously becomes a row, dim 1 (64) becomes a column.
/// 
/// Swizzling pattern is following:
/// 
/// Let's consider an element with logical coordinates = (inRowId, inColId).
/// For simplicity, we do not vectorize memory in examples,
/// i.e. vec == 1 and layout swizzles inidividual elements.
/// For vec != 1 example, take a look at SwizzledSharedEncodingAttr documentation.
/// 
/// Swizzled coordinates within memory object are (outRowId, outColId):
/// 
///   outRowId = inRowId
///   phase   = (inRowId / perPhase) % maxPhase
///   blockNo = (inRowId / (perPhase * maxPhase)) % maxPhase
///   combinedPhase = phase ^ blockNo
///   outColId   = inColId ^ combinedPhase
/// 
/// Actual offset in memory could be computed with following function:
/// 
/// memmory_offset = (outColId + outRowId * num_of_element_in_row) * sizeof(element)
/// 
/// 
/// Swizzling examples (matrix is filled with numbers 0, 1, 2, .. columns*rows-1):
/// 
///   #shared<{vec=1, perPhase=1, maxPhase=2, order=[1,0]}>
///   row      elements
///     0  [ 0,  1,  2,  3],  // phase = 0 blockNo = 0 (xor with 0)
///     1  [ 5,  4,  7,  6],  // phase = 1 blockNo = 0 (xor with 1)
///     2  [ 9,  8, 11, 10],  // phase = 0 blockNo = 1 (xor with 1)
///     3  [12, 13, 14, 15]   // phase = 1 blockNo = 1 (xor with 0)
///     4  [16, 17, 18, 19],  // phase = 0 blockNo = 0 (xor with 0)
///     5  [21, 20, 23, 22],  // phase = 1 blockNo = 0 (xor with 1)
///     6  [25, 24, 27, 26],  // phase = 0 blockNo = 1 (xor with 1)
///     7  [28, 29, 30, 31]   // phase = 1 blockNo = 1 (xor with 0)
/// 
///   #shared<{vec=1, perPhase=2, maxPhase=2, order=[1,0]}>
///   row      elements
///     0  [ 0,  1,  2,  3],  // phase = 0 blockNo = 0 (xor with 0)
///     1  [ 4,  5,  6,  7],  // phase = 0 blockNo = 0 (xor with 0)
///     2  [ 9,  8, 11, 10],  // phase = 1 blockNo = 0 (xor with 1)
///     3  [13, 12, 15, 14]   // phase = 1 blockNo = 0 (xor with 1)
///     4  [17, 16, 19, 18],  // phase = 0 blockNo = 1 (xor with 1)
///     5  [21, 20, 23, 22],  // phase = 0 blockNo = 1 (xor with 1)
///     6  [24, 25, 26, 27],  // phase = 1 blockNo = 1 (xor with 0)
///     7  [28, 29, 30, 31]   // phase = 1 blockNo = 1 (xor with 0)
/// 
///   #shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>
///   row      elements
///     0  [ 0,  1,  2,  3],  // phase = 0 blockNo = 0 (xor with 0)
///     1  [ 5,  4,  7,  6],  // phase = 1 blockNo = 0 (xor with 1)
///     2  [10, 11,  8,  9],  // phase = 2 blockNo = 0 (xor with 2)
///     3  [15, 14, 13, 12]   // phase = 3 blockNo = 0 (xor with 3)
///     4  [17, 16, 19, 18],  // phase = 0 blockNo = 1 (xor with 1)
///     5  [20, 21, 22, 23],  // phase = 1 blockNo = 1 (xor with 0)
///     6  [27, 26, 25, 24],  // phase = 2 blockNo = 1 (xor with 3)
///     7  [30, 31, 28, 29]   // phase = 3 blockNo = 1 (xor with 2)
class AMDRotatingSharedEncodingAttr;
/// See the docs in LinearLayout.h for the definition of linear layouts.
class LinearEncodingAttr;
/// An encoding where each warp owns a contiguous portion of the target tensor. This is typically the kind of data layout
/// used to promote memory coalescing in LoadInst and StoreInst.
/// It is characterized by three tuples -- thread tile size, warp tile size, and block tile size -- which
/// specify the amount of elements owned by each CUDA thread, warp and CTA respectively.
/// 
/// Example 1, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows:
/// 
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// ...
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// 
/// for
/// 
/// #ttg.blocked_layout<{
///   sizePerThread = {2, 2}
///   threadsPerWarp = {8, 4}
///   blocked = {{0, 1}}
/// }>
/// 
/// Example 2, a row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) as follows:
/// 
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// ...                                                 ...
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// ...                                                 ...
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// for
/// 
/// #ttg.blocked_layout<{
///   sizePerThread = {2, 2}
///   threadsPerWarp = {8, 4}
///   blocked = {{0, 1}}
/// }>
/// 
/// Example 3, A row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) and
/// 4 CTAs (taking 2x2 for example) as follows:
/// 
/// CTA [0,0]                                              CTA [0,1]
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// ...                                                    ...
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// 
/// CTA [1,0]                                              CTA [1,1]
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
/// ...                                                    ...
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
/// for
/// 
/// #ttg.blocked_layout<{
///   sizePerThread = {2, 2}
///   threadsPerWarp = {8, 4}
///   blocked = {{0, 1}, {1, 0}}
/// }>
class BlockedEncodingAttr;
/// An encoding for tensors that have been produced by MFMA matrix core instructions,
/// available on AMD Instinct GPUs of CDNA architectures.
/// 
/// It is characterized by the following parameters:
/// - `version`: The GPU architecture:
///   - 1: gfx908: CDNA1
///   - 2: gfx90a: CDNA2
///   - 3: gfx942: CDNA3
///   - 4: gfx950: CDNA4
/// - `warpsPerCTA`: The warp layout in the block.
/// - `instrShape`: The shape in the form of (M, N, K) of the matrix.
/// - `isTransposed`: Indicates the result tensor is transposed so that it can be converted to dotOperand layout
/// without going to shared memory. This is used in the case of chained dot (E.g. Flash-Attention kernel).
/// - `tilesPerWarp`: The tile layout within a warp. Defaults to unit tile layout, i.e., single tile on all dimensions.
/// - `elementBitWidth`: Bit width of the output element type. Supported values are 32 and 64. Defaults to 32.
/// 
/// Example 1:
/// Suppose we have a tensor with a shape of [32, 64], warpsPerCTA set to [1, 2] and MDim=NDim=32.
/// The data will be distributed between threads as follows:
/// 
///                 warp 0                                 warp 1
/// -----------------/\--------------      -----------------/\--------------
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 0   1   2   3  ...... 30  31 ]      [ 64  65  66  67 ...... 94   95  ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// [ 32  33  34  35 ...... 62  63 ]      [ 96  97  98  99 ...... 126  127 ]
/// 
/// Example 2:
/// Suppose we have a tensor with a shape of [16, 32], warpsPerCTA set to [1, 2] and MDim=NDim=16.
/// The data will be distributed between threads as follows:
/// 
///                 warp 0                                 warp 1
/// -----------------/\-------------      ------------------/\---------------
/// [ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
/// [ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
/// [ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
/// [ 0   1   2   3  ...... 14  15 ]      [ 64  65  66  67  ...... 78   79  ]
/// [ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
/// [ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
/// [ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
/// [ 16  17  18  19 ...... 30  31 ]      [ 80  81  82  83  ...... 94   95  ]
/// [ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
/// [ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
/// [ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
/// [ 32  33  34  35 ...... 46  47 ]      [ 96  97  98  99  ...... 110  111 ]
/// [ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
/// [ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
/// [ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
/// [ 48  49  50  51 ...... 62  63 ]      [ 112 113 114 115 ...... 126  127 ]
/// 
/// Example 3:
/// Suppose we have a tensor with a shape of [8, 8], warpsPerCTA set to [2, 2] and nonKDim set to 4.
/// The data will be distributed between threads as follows(note that each element is duplicated in 16 threads):
/// Suppose we have a tensor with a shape of [8, 8], warpsPerCTA set to [2, 2] and MDim=NDim=4.
/// The data will be distributed between threads as follows(note that each element is duplicated in 16 threads):
/// 
/// M  N ->                    warp 0                                                       warp 2
/// | --------------------------/\--------------------------   ------------------------------/\------------------------------
/// V [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
///   [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
///   [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
///   [ 0,4,8...60   1,5...61     2,6...62     3,7...63    ]   [ 128,132...188  129,133...189  130,134...190  131,135...191 ]
///                            warp 1                                                       warp 3
///   --------------------------/\--------------------------   ------------------------------/\------------------------------
///   [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
///   [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
///   [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
///   [ 64,68...124  65,69...125  66,70...126  67,71...127 ]   [ 192,196...252  193,197...253  194,198...254  195,199...255 ]
/// 
/// Example 4:
/// This example demonstrates semantics of tilesPerWarp parameter. The MFMA layout (with tilesPerWarp=[1,1])
/// assumes that each warp within a CTA tile computes a single MFMA tile. When the tensor is larger than
/// a single CTA tile, these tiles are repeated across the tensor. In this setup, the output tiles computed
/// by each warp were strided by the number of warps per CTA tile in both row and column dimensions.
/// 
/// For instance, with 16 MFMA tiles and warpsPerCTA = [2, 2], the distribution of warps across the MFMA
/// tiles looked like:
/// 
/// w0 w1 w0 w1
/// w2 w3 w2 w3
/// w0 w1 w0 w1
/// w2 w3 w2 w3
/// 
/// tilesPerWarp parameter allows each warp to compute contiguous MFMA tiles in the row and/or column dimensions.
/// Using the same example with tilesPerWarp = [2, 2], the layout becomes:
/// 
/// w0 w0 w1 w1
/// w0 w0 w1 w1
/// w2 w2 w3 w3
/// w2 w2 w3 w3
class AMDMfmaEncodingAttr;
/// An encoding for tensors that have been produced by WMMA matrix core instructions,
/// available on AMD Radeon GPUs of RDNA architectures.
/// 
/// It is characterized by the following parameters:
/// - `version` indicates the GPU architecture:
///   - 1: RDNA3; e.g., gfx1100, gfx1101
///   - 2: RDNA4; e.g., gfx1200, gfx1201
///   - 3: gfx1250
/// - `warpsPerCTA` indicates the warp layout in the block.
/// - `tilesPerWarp`  The tile layout within a warp. Defaults to unit tile layout, i.e., single tile on all dimensions.
/// - `instrShape` indicates the shape in the form of (M, N, K) of the matrix
///    operation performed by a single WMMA instruction. Defaults to (16, 16, 16).
/// - `isTransposed` indicates the layout of the result tensor is transposed.
/// 
/// Example 1:
/// Suppose we have a tensor with shape [32, 64], `warpsPerCTA` set to [2, 2].
/// Matrix elements represent which lane owns the element. Currently only wave32 mode
/// is supported.
/// 
/// // ----------------------------------- version = 1 ----------------------------------- //
/// 
/// Row |                  warp 0                                    warp 1
///     |/-------------------^-------------------\ /-------------------^-------------------\
/// 0   |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 1   |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 2   |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 3   |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
///     | ...                  ...                  ...                  ...
/// 14  |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 15  |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 
///     |                  warp 2                                    warp 3
/// 16  |/-------------------^-------------------\ /-------------------^-------------------\
/// 17  |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 18  |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 19  |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 20  |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
///     | ...                  ...                  ...                  ...
/// 30  |[0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 31  |[16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 
/// // ------------------------ version = 2/3, isTransposed = false ------------------------ //
/// 
/// Row |       warp 0                warp 1
///     |/--------^---------\ /---------^--------\
/// 0   |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 1   |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// ..  | ...                    ...
/// 6   |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 7   |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 8   |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 9   |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// ..  | ...                  ...
/// 14  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 15  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
///     |
///     |       warp 2                warp 3
///     |/--------^---------\ /---------^--------\
/// 16  |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 17  |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// ..  | ...                    ...
/// 22  |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 23  |[0  1  2  ... 14 15] [0  1  2  ... 14 15]
/// 24  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 25  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// ..  | ...                  ...
/// 30  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 31  |[16 17 18 ... 30 31] [16 17 18 ... 30 31]
/// 
/// // ------------------------ version = 2/3, isTransposed = true ------------------------ //
/// 
///     |               warp 0                     warp 1
///     |/----------------^----------------\ /-------^-------\
/// Col>| 0  1  2  3  4  5  6  7  8  ... 15  16 17 18  ... 32
/// Row |
/// 0   |[0  0  0  0  0  0  0  0  16 ... 16] [0  0  0  ... 16]
/// 1   |[1  1  1  1  1  1  1  1  17 ... 17] [1  1  1  ... 17]
/// ..  | ...                  ...
/// 14  |[14 14 14 14 14 14 14 14 30 ... 30] [14 14 14 ... 30]
/// 15  |[15 15 15 15 15 15 15 15 31 ... 31] [15 15 15 ... 31]
///     |
///     |               warp 2                     warp 3
///     |/----------------^----------------\ /-------^-------\
/// 16  |[0  0  0  0  0  0  0  0  16 ... 16] [0  0  0  ... 16]
/// 17  |[1  1  1  1  1  1  1  1  17 ... 17] [1  1  1  ... 17]
/// ..  | ...                  ...
/// 30  |[14 14 14 14 14 14 14 14 30 ... 30] [14 14 14 ... 30]
/// 31  |[15 15 15 15 15 15 15 15 31 ... 31] [15 15 15 ... 31]
/// 
/// Example 2:
/// This example demonstrates the tilesPerWarp parameter, which shares the same sematics with
/// AMDMfmaEncodingAttr.
/// 
/// By default, WMMA layout assumes that each warp within a CTA tile computes a single WMMA tile.
/// When the tensor is larger than a single CTA tile, these tiles are repeated across the tensor.
/// In this setup, the output tiles computed by each warp are strided by the number of warps per CTA
/// tile in both row and column dimensions.
/// 
/// For instance, with 16 WMMA tiles and warpsPerCTA = [2, 2], the default(tilesPerWarp = [1, 1])
/// distribution of warps across the WMMA tiles looked like:
/// 
/// w0 w1 w0 w1
/// w2 w3 w2 w3
/// w0 w1 w0 w1
/// w2 w3 w2 w3
/// 
/// * Each unit reprsents a WMMA tile. w* shows which warp occupies that WMMA tile.
/// 
/// tilesPerWarp parameter allows each warp to compute contiguous WMMA tiles in the row and/or column dimensions.
/// Using the same example with tilesPerWarp = [2, 2], the layout becomes:
/// 
/// w0 w0 w1 w1
/// w0 w0 w1 w1
/// w2 w2 w3 w3
/// w2 w2 w3 w3
class AMDWmmaEncodingAttr;
/// An encoding for tensors that have been produced by tensor cores.
/// 
/// It is characterized by two parameters:
/// - A 'versionMajor' which specifies the generation the tensor cores
///   whose output is being partitioned:
///   - 1 for first-gen tensor cores (Volta), and
///   - 2 for second-gen tensor cores (Turing/Ampere).
/// - A 'versionMinor' which indicates the specific layout of a tensor core
///   generation, e.g. for Volta, there might be multiple kinds of layouts
///   annotated by 0,1,2 and so on.
/// - A `blockTileSize` to indicate how data should be partitioned between warps.
/// 
/// // -------------------------------- version = 1 --------------------------- //
/// 
/// For first-gen tensor cores, the implicit warpTileSize is [16, 16].
/// Note: the layout is different from the recommended in PTX ISA
/// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
/// (mma.884 section, FP32 accumulator).
/// 
/// For example, when versionMinor=1, the matrix L corresponding to
/// blockTileSize=[32,16] is:
/// 
///                                warp 0
/// --------------------------------/\-------------------------------
/// [ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
/// [ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
/// [ 0   0   2   2   8   8   10  10   0   0   2   2   8   8   10  10 ]
/// [ 1   1   3   3   9   9   11  11   1   1   3   3   9   9   11  11 ]
/// [ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
/// [ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
/// [ 4   4   6   6   12  12  14  14   4   4   6   6   12  12  14  14 ]
/// [ 5   5   7   7   13  13  15  15   5   5   7   7   13  13  15  15 ]
/// [ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
/// [ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
/// [ 16  16  18  18  20  20  22  22   16  16  18  18  20  20  22  22 ]
/// [ 17  17  19  19  21  21  23  23   17  17  19  19  21  21  23  23 ]
/// [ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
/// [ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]
/// [ 24  24  26  26  28  28  30  30   24  24  26  26  28  28  30  30 ]
/// [ 25  25  27  27  29  29  31  31   25  25  27  27  29  29  31  31 ]
/// 
///                           warp 1 = warp0 + 32
/// --------------------------------/\-------------------------------
/// [ 32  32  34  34  40  40  42  42   32  32  34  34  40  40  42  42 ]
/// [ 33  33  35  35  41  41  43  43   33  33  35  35  41  41  43  43 ]
/// [ ............................................................... ]
/// 
/// 
/// // -------------------------------- version = 2 --------------------------- //
/// 
/// For second-gen tensor cores, the implicit warpTileSize is [16, 8].
/// Information about this layout can be found in the official PTX documentation
/// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
/// (mma.16816 section, FP32 accumulator).
/// 
/// For example, the matrix L corresponding to blockTileSize=[32,16] is:
///                 warp 0                          warp 2
/// -----------------/\-------------  ----------------/\-------------
/// [ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
/// [ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
/// [ ..............................  ..............................
/// [ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63
/// [ 0   0   1   1   2   2   3   3   32  32  33  33  34  34  35  35
/// [ 4   4   5   5   6   6   7   7   36  36  37  37  38  38  39  39
/// [ ..............................  ..............................
/// [ 28  28  29  29  30  30  31  31  60  60  61  61  62  62  63  63
/// 
///               warp 1                           warp 3
/// ----------------/\-------------   ----------------/\-------------
/// [ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
/// [ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
/// [ ..............................  ...............................
/// [ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127
/// [ 64  64  65  65  66  66  67  67  96  96  97  97  98  98  99  99
/// [ 68  68  69  69  70  70  71  71  100 100 101 101 102 102 103 103
/// [ ..............................  ...............................
/// [ 92  92  93  93  94  94  95  95  124 124 125 125 126 126 127 127
class NvidiaMmaEncodingAttr;
/// Given a `parent` layout and a `dim`, squeezes the given `dim` in the `parent`
///     layout and distributes values in a tensor T according to the new layout.
/// 
///     For example, given
/// 
///     T = [x  x  x  x  x  x  x  x]
///     L_parent = [0  1  2  3 ]
///                [4  5  6  7 ]
///                [8  9  10 11]
///                [12 13 14 15] (with 16 CUDA threads)
/// 
///     With dim = 0, squeezing out dim 0, we have
///     L = [{0,4,8,12},  {1,5,9,13}, {2,6,10,14},  {3,7,11,15} ]
/// 
///     Then the data of T would be distributed as follow between the 16 CUDA threads:
///     L(T) = [ {0,4,8,12} , {1,5,9,13} , ... {3,7,11,15}, {0,4,8,12} , ..., {3,7,11,15} ]
/// 
///     With dim = 1, squeezing out dim 1, we have
///     L = [ {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15} ]
/// 
///     Then the data of T would be distributed as follow between the 16 CUDA threads:
///     L = [ {0,1,2,3}, {4,5,6,7}, ..., {12,13,14,15}, {0,1,2,3}, ..., {12,13,14,15} ]
/// 
///     This is useful for constructing the inverse layout of an expand_dims operation
///     during some optimization passes.
class SliceEncodingAttr;
/// In the TritonGPU dialect, given `d = tt.dot a, b, c` tt.dot's operands a and b
/// must be of DotOperandEncodingAttr layout, if the dot is MMA v1 or v2 (i.e.
/// pre-Hopper).  For MMA v3, the operands are *almost always* in a regular shared
/// encoding, but sometimes the LHS is also a dot-operand encoding.
/// 
/// a's opIdx is 0, b's opIdx is 1.
/// 
/// The parent field is the layout of d.
/// 
/// kWidth defines number of consecutive elements stored by one thread along k dimension.
/// Some layouts do not use this parameter, either because they have a fixed number of
/// elements along the K dim, or they use all elements of the tensor along the K dim.
/// 
/// # WGMMA Notes
/// We require kWidth to be provided for Hopper because the dtype at loading might be
/// different from the dtype at WGMMA, due to casting. The kWidth is determined by the
/// dtype at WGMMA.
/// 
/// The encoded tensor consists of operand A for possibly multiple wgmma instructions.
/// For each wgmma, each warp in a warp group feeds a single "warp matrix"
/// Each warp matrix consists of 2x2 "quads".
/// Each thread holds several elements in each quad. Right before a wgmma,
/// the sum of bitwidth of
/// the elements in each quad should add up to 32.
/// 
/// These values are stored unrolled in `elements`.
/// The ordering of dimensions is as follows by convention:
/// batch (only 1 batch for Hopper currently)
/// matM (m-index of the "warp matrix")
/// matK (k-index of the "warp matrix")
/// quadK (k-index of the "quad" in the core matrix)
/// quadM (m-index of the "quad" in the core matrix)
/// vecIdx (index of the element in the quad; this is always along the k-dim)
class DotOperandEncodingAttr;
/// Attribute to indicate that the memory descriptor points to shared memory.
class SharedMemorySpaceAttr;
namespace detail {

struct SwizzledSharedEncodingAttrStorage;

} // namespace detail
class SwizzledSharedEncodingAttr : public ::mlir::Attribute::AttrBase<SwizzledSharedEncodingAttr, ::mlir::Attribute, detail::SwizzledSharedEncodingAttrStorage, ::mlir::triton::gpu::SharedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.swizzled_shared_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<SwizzledSharedEncodingAttr>::getRank;
  static SwizzledSharedEncodingAttr get(::mlir::MLIRContext *context, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static SwizzledSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static SwizzledSharedEncodingAttr get(::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned typeWidthInBit);
  static SwizzledSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned typeWidthInBit);
  static SwizzledSharedEncodingAttr get(::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned typeWidthInBit, bool needTrans);
  static SwizzledSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned typeWidthInBit, bool needTrans);
  static SwizzledSharedEncodingAttr get(::mlir::MLIRContext *context, int opIdx, unsigned kWidth, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned bitwidth, bool needTrans);
  static SwizzledSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, int opIdx, unsigned kWidth, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, unsigned bitwidth, bool needTrans);
  static SwizzledSharedEncodingAttr get(::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy);
  static SwizzledSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy);
  static SwizzledSharedEncodingAttr get(::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy, bool needTrans);
  static SwizzledSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, DotOperandEncodingAttr dotOpEnc, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy, bool needTrans);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"swizzled_shared"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getVec() const;
  unsigned getPerPhase() const;
  unsigned getMaxPhase() const;
  ::llvm::ArrayRef<unsigned> getOrder() const;
  CGAEncodingAttr getCGALayout() const;
};
namespace detail {

struct PaddedSharedEncodingAttrStorage;

} // namespace detail
class PaddedSharedEncodingAttr : public ::mlir::Attribute::AttrBase<PaddedSharedEncodingAttr, ::mlir::Attribute, detail::PaddedSharedEncodingAttrStorage, ::mlir::triton::gpu::SharedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  // Returns the order of the dimensions `dimName` of the layout.
  // If more than dimension is of size one, it uses defaultOrder to determine
  // the order of the dimensions of size one.
  SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                    ArrayRef<unsigned> defaultOrder) const;
  SmallVector<unsigned> getOrder() const;

  // Returns the bases of the dimensions `dimName` of the linear_component.
  // If skipBroadcast is false, we count a base zero
  SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                    bool skipBroadcast = true) const;

  unsigned getMinInterval() const {
    return *llvm::min_element(getIntervals());
  }

  // Returns the total number of elements including padding given the input
  // tensor shape.
  int64_t getPaddedSize(ArrayRef<int64_t> shape) const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.padded_shared_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<PaddedSharedEncodingAttr>::getRank;
  static PaddedSharedEncodingAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<unsigned> intervals, ::llvm::ArrayRef<unsigned> paddings, LinearLayout linearComponent);
  static PaddedSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<unsigned> intervals, ::llvm::ArrayRef<unsigned> paddings, LinearLayout linearComponent);
  static PaddedSharedEncodingAttr get(::mlir::MLIRContext *context, ArrayRef<std::pair<unsigned, unsigned>> intervalPads, LinearLayout linearComponent);
  static PaddedSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<std::pair<unsigned, unsigned>> intervalPads, LinearLayout linearComponent);
  static PaddedSharedEncodingAttr get(::mlir::MLIRContext *context, ArrayRef<std::pair<unsigned, unsigned>> intervalPads, ArrayRef<unsigned> order, ArrayRef<int64_t> shape, CGAEncodingAttr cgaLayout);
  static PaddedSharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<std::pair<unsigned, unsigned>> intervalPads, ArrayRef<unsigned> order, ArrayRef<int64_t> shape, CGAEncodingAttr cgaLayout);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<unsigned> intervals, ::llvm::ArrayRef<unsigned> paddings, LinearLayout linearComponent);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<unsigned> intervals, ::llvm::ArrayRef<unsigned> paddings, LinearLayout linearComponent);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"padded_shared"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<unsigned> getIntervals() const;
  ::llvm::ArrayRef<unsigned> getPaddings() const;
  const LinearLayout &getLinearComponent() const;
  CGAEncodingAttr getCGALayout() const;
};
namespace detail {

struct SharedLinearEncodingAttrStorage;

} // namespace detail
class SharedLinearEncodingAttr : public ::mlir::Attribute::AttrBase<SharedLinearEncodingAttr, ::mlir::Attribute, detail::SharedLinearEncodingAttrStorage, ::mlir::triton::gpu::SharedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                    bool skipBroadcast = true) const;
  SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                    ArrayRef<unsigned> defaultOrder) const;

  SmallVector<unsigned> getOrder() const;

  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

  int32_t getAlignment() const { return static_cast<int32_t>(getLayoutAlignment()); }
  static constexpr ::llvm::StringLiteral name = "triton.gpu.shared_linear_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<SharedLinearEncodingAttr>::getRank;
  static SharedLinearEncodingAttr get(::mlir::MLIRContext *context, LinearLayout linearLayout, unsigned layoutAlignment);
  static SharedLinearEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, LinearLayout linearLayout, unsigned layoutAlignment);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout, unsigned layoutAlignment);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout, unsigned layoutAlignment);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shared_linear"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  const LinearLayout &getLinearLayout() const;
  unsigned getLayoutAlignment() const;
  CGAEncodingAttr getCGALayout() const;
};
namespace detail {

struct NVMMASharedEncodingAttrStorage;

} // namespace detail
class NVMMASharedEncodingAttr : public ::mlir::Attribute::AttrBase<NVMMASharedEncodingAttr, ::mlir::Attribute, detail::NVMMASharedEncodingAttrStorage, ::mlir::triton::gpu::SharedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  int getPerPhase() const;
  int getMaxPhase() const;
  int getVec() const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.nvmma_shared_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<NVMMASharedEncodingAttr>::getRank;
  static NVMMASharedEncodingAttr get(::mlir::MLIRContext *context, unsigned swizzlingByteWidth, bool transposed, unsigned elementBitWidth, bool fp4Padded, CGAEncodingAttr CGALayout);
  static NVMMASharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned swizzlingByteWidth, bool transposed, unsigned elementBitWidth, bool fp4Padded, CGAEncodingAttr CGALayout);
  static NVMMASharedEncodingAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy, bool fp4Padded);
  static NVMMASharedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> order, CGAEncodingAttr CGALayout, Type eltTy, bool fp4Padded);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned swizzlingByteWidth, bool transposed, unsigned elementBitWidth, bool fp4Padded, CGAEncodingAttr CGALayout);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned swizzlingByteWidth, bool transposed, unsigned elementBitWidth, bool fp4Padded, CGAEncodingAttr CGALayout);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"nvmma_shared"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getSwizzlingByteWidth() const;
  bool getTransposed() const;
  unsigned getElementBitWidth() const;
  bool getFp4Padded() const;
  CGAEncodingAttr getCGALayout() const;
  int32_t getAlignment() const;
};
namespace detail {

struct AMDRotatingSharedEncodingAttrStorage;

} // namespace detail
class AMDRotatingSharedEncodingAttr : public ::mlir::Attribute::AttrBase<AMDRotatingSharedEncodingAttr, ::mlir::Attribute, detail::AMDRotatingSharedEncodingAttrStorage, ::mlir::triton::gpu::SharedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.amd_rotating_shared_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<AMDRotatingSharedEncodingAttr>::getRank;
  static AMDRotatingSharedEncodingAttr get(::mlir::MLIRContext *context, unsigned vec, unsigned perPhase, unsigned maxPhase, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"amd_rotating_shared"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getVec() const;
  unsigned getPerPhase() const;
  unsigned getMaxPhase() const;
  ::llvm::ArrayRef<unsigned> getOrder() const;
  CGAEncodingAttr getCGALayout() const;
};
namespace detail {

struct LinearEncodingAttrStorage;

} // namespace detail
class LinearEncodingAttr : public ::mlir::Attribute::AttrBase<LinearEncodingAttr, ::mlir::Attribute, detail::LinearEncodingAttrStorage, ::mlir::triton::gpu::DistributedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  // Implemented in subclasses
  SmallVector<unsigned> getRepOrder() const;

  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

  // Generic distributed encoding methods
  unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape) const;
  SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape) const;

  SmallVector<unsigned int> getContig(const char *, SmallVector<unsigned int>) const;
  SmallVector<unsigned> getContigPerThread() const;
  SmallVector<unsigned> getContigPerWarp() const;
  SmallVector<unsigned> getOrder() const;
  SmallVector<unsigned> getWarpOrder() const;
  SmallVector<unsigned> getThreadOrder() const;


  // Generalizes get{Warp,Thread,CTA}Order to linear layouts.
  // Returns the order of the dimensions `dimName` of the layout.
  // If more than dimension is of size one, it uses defaultOrder to determine
  // the order of the dimensions of size one.
  SmallVector<unsigned> orderPerDim(StringAttr dimName,
                                    ArrayRef<unsigned> defaultOrder) const;

  // Generalizes getThreadsPerWarp, getWarpsPerCTA, getCTAsPerCGA to linear layouts.
  // Returns the bases of the dimensions `dimName` of the layout.
  // If skipBroadcast is false, we count a base zero
  SmallVector<unsigned> basesPerDim(StringAttr dimName,
                                    bool skipBroadcast = true) const;
  SmallVector<unsigned> getThreadsPerWarp() const;
  SmallVector<unsigned> getWarpsPerCTA() const;

  // [FIXME LL] Supports legacy behaviour. We should remove these functions
  SmallVector<unsigned> getSizePerThread() const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.linear_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<LinearEncodingAttr>::getRank;
  static LinearEncodingAttr get(::mlir::MLIRContext *context, LinearLayout linearLayout);
  static LinearEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, LinearLayout linearLayout);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, LinearLayout linearLayout);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"linear"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  const LinearLayout &getLinearLayout() const;
  CGAEncodingAttr getCGALayout() const;
};
namespace detail {

struct BlockedEncodingAttrStorage;

} // namespace detail
class BlockedEncodingAttr : public ::mlir::Attribute::AttrBase<BlockedEncodingAttr, ::mlir::Attribute, detail::BlockedEncodingAttrStorage, ::mlir::triton::gpu::DistributedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  // Implemented in subclasses
  SmallVector<unsigned> getRepOrder() const;

  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.blocked_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<BlockedEncodingAttr>::getRank;
  static BlockedEncodingAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<unsigned> sizePerThread, ::llvm::ArrayRef<unsigned> threadsPerWarp, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static BlockedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ::llvm::ArrayRef<unsigned> sizePerThread, ::llvm::ArrayRef<unsigned> threadsPerWarp, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static BlockedEncodingAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order, unsigned numWarps, unsigned numThreadsPerWarp, CGAEncodingAttr CGALayout);
  static BlockedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order, unsigned numWarps, unsigned numThreadsPerWarp, CGAEncodingAttr CGALayout);
  static BlockedEncodingAttr get(::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order, unsigned numWarps, unsigned numThreadsPerWarp, unsigned numCTAs);
  static BlockedEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order, unsigned numWarps, unsigned numThreadsPerWarp, unsigned numCTAs);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<unsigned> sizePerThread, ::llvm::ArrayRef<unsigned> threadsPerWarp, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::llvm::ArrayRef<unsigned> sizePerThread, ::llvm::ArrayRef<unsigned> threadsPerWarp, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> order, CGAEncodingAttr CGALayout);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"blocked"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<unsigned> getSizePerThread() const;
  ::llvm::ArrayRef<unsigned> getThreadsPerWarp() const;
  ::llvm::ArrayRef<unsigned> getWarpsPerCTA() const;
  ::llvm::ArrayRef<unsigned> getOrder() const;
  CGAEncodingAttr getCGALayout() const;
};
namespace detail {

struct AMDMfmaEncodingAttrStorage;

} // namespace detail
class AMDMfmaEncodingAttr : public ::mlir::Attribute::AttrBase<AMDMfmaEncodingAttr, ::mlir::Attribute, detail::AMDMfmaEncodingAttrStorage, ::mlir::triton::gpu::DistributedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait, ::mlir::triton::gpu::MmaEncodingTrait::Trait> {
public:
  using Base::Base;
  // Implemented in subclasses
  SmallVector<unsigned> getRepOrder() const;

  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

  SmallVector<int64_t> getInstrShapeForOperand(int kWidth, int opIdx) const;
  SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> operandShape, int kWidth, int opIdx) const;
  SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;

  // Check if tilesPerWarp is 1 in every dimension.
  bool hasUnitTilesPerWarp() const;

  // Returns a swizzled shared layout matching this MFMA layout for the
  // dot operand at the given |operandIdx| with |operandShape|.
  SwizzledSharedEncodingAttr composeSharedLayoutForOperand(
      CGAEncodingAttr cgaLayout, int operandIdx, ArrayRef<int64_t> operandShape,
      ArrayRef<unsigned> sharedOrder, unsigned vectorSize,
      unsigned elemBitWidth, bool needTrans) const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.amd_mfma_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<AMDMfmaEncodingAttr>::getRank;
  static AMDMfmaEncodingAttr get(::mlir::MLIRContext *context, unsigned version, ArrayRef<unsigned> warpsPerCTA, ArrayRef<unsigned> instrShape, bool isTransposed, CGAEncodingAttr CGALayout, ArrayRef<unsigned> tpw = {}, unsigned elementBitWidth = 0);
  static AMDMfmaEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned version, ArrayRef<unsigned> warpsPerCTA, ArrayRef<unsigned> instrShape, bool isTransposed, CGAEncodingAttr CGALayout, ArrayRef<unsigned> tpw = {}, unsigned elementBitWidth = 0);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned version, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> instrShape, bool isTransposed, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> tilesPerWarp, unsigned elementBitWidth);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned version, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> instrShape, bool isTransposed, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> tilesPerWarp, unsigned elementBitWidth);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"amd_mfma"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getVersion() const;
  ::llvm::ArrayRef<unsigned> getWarpsPerCTA() const;
  ::llvm::ArrayRef<unsigned> getInstrShape() const;
  bool getIsTransposed() const;
  CGAEncodingAttr getCGALayout() const;
  ::llvm::ArrayRef<unsigned> getTilesPerWarp() const;
  unsigned getElementBitWidth() const;
};
namespace detail {

struct AMDWmmaEncodingAttrStorage;

} // namespace detail
class AMDWmmaEncodingAttr : public ::mlir::Attribute::AttrBase<AMDWmmaEncodingAttr, ::mlir::Attribute, detail::AMDWmmaEncodingAttrStorage, ::mlir::triton::gpu::DistributedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait, ::mlir::triton::gpu::MmaEncodingTrait::Trait> {
public:
  using Base::Base;
  // Implemented in subclasses
  SmallVector<unsigned> getRepOrder() const;

  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

  SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> operandShape, int kDim, int opIdx) const;
  SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;

  static SmallVector<unsigned, 3> getDefaultInstrShape() {
    return {16, 16, 16};
  }

  // Check if tilesPerWarp is 1 in every dimension.
  bool hasUnitTilesPerWarp() const;

  // Returns a swizzled shared layout matching this WMMA layout for the
  // dot operand at the given |operandIdx| with |operandShape|.
  SwizzledSharedEncodingAttr composeSharedLayoutForOperand(
      CGAEncodingAttr cgaLayout, int operandIdx, ArrayRef<int64_t> operandShape,
      ArrayRef<unsigned> sharedOrder, unsigned kWidth,
      unsigned elemBitWidth, bool needTrans) const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.amd_wmma_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<AMDWmmaEncodingAttr>::getRank;
  static AMDWmmaEncodingAttr get(::mlir::MLIRContext *context, unsigned version, bool isTransposed, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> tilesPerWarp, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape);
  static AMDWmmaEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned version, bool isTransposed, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> tilesPerWarp, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape);
  static AMDWmmaEncodingAttr get(::mlir::MLIRContext *context, unsigned version, bool isTransposed, ArrayRef<unsigned> warpsPerCTA, CGAEncodingAttr CGALayout, ArrayRef<unsigned> instrShape);
  static AMDWmmaEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned version, bool isTransposed, ArrayRef<unsigned> warpsPerCTA, CGAEncodingAttr CGALayout, ArrayRef<unsigned> instrShape);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned version, bool isTransposed, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> tilesPerWarp, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned version, bool isTransposed, ::llvm::ArrayRef<unsigned> warpsPerCTA, ::llvm::ArrayRef<unsigned> tilesPerWarp, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"amd_wmma"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getVersion() const;
  bool getIsTransposed() const;
  ::llvm::ArrayRef<unsigned> getWarpsPerCTA() const;
  ::llvm::ArrayRef<unsigned> getTilesPerWarp() const;
  CGAEncodingAttr getCGALayout() const;
  ::llvm::ArrayRef<unsigned> getInstrShape() const;
};
namespace detail {

struct NvidiaMmaEncodingAttrStorage;

} // namespace detail
class NvidiaMmaEncodingAttr : public ::mlir::Attribute::AttrBase<NvidiaMmaEncodingAttr, ::mlir::Attribute, detail::NvidiaMmaEncodingAttrStorage, ::mlir::triton::gpu::DistributedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait, ::mlir::triton::gpu::MmaEncodingTrait::Trait> {
public:
  using Base::Base;
  // Implemented in subclasses
  SmallVector<unsigned> getRepOrder() const;

  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

  bool isVolta() const;
  bool isTuring() const;
  bool isAmpere() const;
  bool isHopper() const;

  SmallVector<int64_t> getRepForOperand(ArrayRef<int64_t> shape,
                                        int bitwidth, int kWidth,
                                        int opIdx) const;
  SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.nvidia_mma_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<NvidiaMmaEncodingAttr>::getRank;
  static NvidiaMmaEncodingAttr get(::mlir::MLIRContext *context, unsigned versionMajor, unsigned versionMinor, ::llvm::ArrayRef<unsigned> warpsPerCTA, CGAEncodingAttr CGALayout, ::llvm::ArrayRef<unsigned> instrShape);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"nvidia_mma"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getVersionMajor() const;
  unsigned getVersionMinor() const;
  ::llvm::ArrayRef<unsigned> getWarpsPerCTA() const;
  CGAEncodingAttr getCGALayout() const;
  ::llvm::ArrayRef<unsigned> getInstrShape() const;
};
namespace detail {

struct SliceEncodingAttrStorage;

} // namespace detail
class SliceEncodingAttr : public ::mlir::Attribute::AttrBase<SliceEncodingAttr, ::mlir::Attribute, detail::SliceEncodingAttrStorage, ::mlir::triton::gpu::DistributedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  // Implemented in subclasses
  SmallVector<unsigned> getRepOrder() const;

  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;

  template<class T>
  SmallVector<T> paddedShape(ArrayRef<T> shape) const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.slice_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<SliceEncodingAttr>::getRank;
  static SliceEncodingAttr get(::mlir::MLIRContext *context, unsigned dim, DistributedEncodingTrait parent);
  static SliceEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned dim, DistributedEncodingTrait parent);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned dim, DistributedEncodingTrait parent);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned dim, DistributedEncodingTrait parent);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"slice"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getDim() const;
  DistributedEncodingTrait getParent() const;
  CGAEncodingAttr getCGALayout() const;
};
namespace detail {

struct DotOperandEncodingAttrStorage;

} // namespace detail
class DotOperandEncodingAttr : public ::mlir::Attribute::AttrBase<DotOperandEncodingAttr, ::mlir::Attribute, detail::DotOperandEncodingAttrStorage, ::mlir::triton::gpu::DistributedEncodingTrait::Trait, ::mlir::triton::gpu::LayoutEncodingTrait::Trait> {
public:
  using Base::Base;
  // Implemented in subclasses
  SmallVector<unsigned> getRepOrder() const;

  LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  static constexpr ::llvm::StringLiteral name = "triton.gpu.dot_operand_encoding";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  using Base::getChecked;
  using ::mlir::triton::gpu::LayoutEncodingTrait::Trait<DotOperandEncodingAttr>::getRank;
  static DotOperandEncodingAttr get(::mlir::MLIRContext *context, unsigned opIdx, Attribute parent, unsigned kWidth);
  static DotOperandEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned opIdx, Attribute parent, unsigned kWidth);
  static DotOperandEncodingAttr get(::mlir::MLIRContext *context, unsigned opIdx, Attribute parent, Type eltTy);
  static DotOperandEncodingAttr getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, ::mlir::MLIRContext *context, unsigned opIdx, Attribute parent, Type eltTy);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned opIdx, Attribute parent, unsigned kWidth);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned opIdx, Attribute parent, unsigned kWidth);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"dot_op"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  unsigned getOpIdx() const;
  Attribute getParent() const;
  unsigned getKWidth() const;
  CGAEncodingAttr getCGALayout() const;
};
class SharedMemorySpaceAttr : public ::mlir::Attribute::AttrBase<SharedMemorySpaceAttr, ::mlir::Attribute, ::mlir::AttributeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "ttg.shared_memory";
  static constexpr ::llvm::StringLiteral dialectName = "ttg";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shared_memory"};
  }

};

} // namespace mlir::triton::gpu
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::SwizzledSharedEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::PaddedSharedEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::SharedLinearEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::NVMMASharedEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::AMDRotatingSharedEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::LinearEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::BlockedEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::AMDMfmaEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::AMDWmmaEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::NvidiaMmaEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::SliceEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::DotOperandEncodingAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::gpu::SharedMemorySpaceAttr)

#endif // GET_ATTRDEF_CLASSES

