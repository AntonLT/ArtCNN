#import "ArtCNNMetal.h"

@interface ArtCNNPass : NSObject
@property (nonatomic) id<MTLComputePipelineState> pipeline;
@property (nonatomic) NSArray<NSString *> *binds;
@property (nonatomic) NSArray<NSString *> *saves;
@property (nonatomic) MTLSize block;
@property (nonatomic) MTLSize threads;
@property (nonatomic) float scaleX;
@property (nonatomic) float scaleY;
@property (nonatomic) NSString *name;
@end

@implementation ArtCNNPass
@end

// The Winograd kernels fill weight tiles per lane, assuming lane l holds elements
// (r, c) and (r, c + 1) of an 8x8 simdgroup matrix with q = l / 4, r = (q & 4) + (l / 2) % 4,
// c = (q & 2) * 2 + (l % 2) * 2. Metal does not document this layout, so it is checked on
// the device before the shader is used.
static NSString *const kLaneLayoutCheck =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void check_lane_layout(device uint *bad [[buffer(0)]], ushort lane [[thread_index_in_simdgroup]])\n"
     "{\n"
     "    threadgroup half m[64];\n"
     "    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8, 8>(-1.0h);\n"
     "    const uint q = lane / 4;\n"
     "    const uint r = (q & 4) + (lane / 2) % 4, c = (q & 2) * 2 + (lane % 2) * 2;\n"
     "    a.thread_elements()[0] = half(r * 8 + c);\n"
     "    a.thread_elements()[1] = half(r * 8 + c + 1);\n"
     "    simdgroup_store(a, m, 8);\n"
     "    simdgroup_barrier(mem_flags::mem_threadgroup);\n"
     "    for (uint i = lane; i < 64; i += 32) {\n"
     "        if (m[i] != half(i)) {\n"
     "            atomic_fetch_add_explicit((device atomic_uint *)bad, 1, memory_order_relaxed);\n"
     "        }\n"
     "    }\n"
     "}\n";

static BOOL checkLaneLayout(id<MTLDevice> device, NSError **error)
{
    id<MTLLibrary> library = [device newLibraryWithSource:kLaneLayoutCheck options:nil error:error];
    id<MTLFunction> function = [library newFunctionWithName:@"check_lane_layout"];
    id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:error] : nil;
    if (!pipeline) {
        return NO;
    }
    id<MTLBuffer> bad = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    *(uint32_t *)bad.contents = 0;
    id<MTLCommandBuffer> commandBuffer = [[device newCommandQueue] commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bad offset:0 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (*(uint32_t *)bad.contents != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"ArtCNNMetal" code:2 userInfo:@{
                NSLocalizedDescriptionKey: @"simdgroup_matrix lane layout differs from the one the shader assumes"}];
        }
        return NO;
    }
    return YES;
}

// Reuse mask in two steps: artcnn_changed marks cells (one thread per pixel) whose luma
// differs from the previous frame; artcnn_dilate marks a cell dirty if a changed cell lies
// within the shader's dependency radius, rounded up to whole cells.
static NSString *const kDirtyMaskSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void artcnn_changed(texture2d<half> luma [[texture(0)]], texture2d<half> prev [[texture(1)]],\n"
     "    device atomic_uint *changed [[buffer(0)]], constant uint4 &cfg [[buffer(1)]],\n"
     "    uint2 gid [[thread_position_in_grid]])\n"
     "{\n"
     "    // cfg = (cells_x, cells_y, cell_w, cell_h)\n"
     "    if (gid.x >= luma.get_width() || gid.y >= luma.get_height()) {\n"
     "        return;\n"
     "    }\n"
     "    if (luma.read(gid).x != prev.read(gid).x) {\n"
     "        atomic_store_explicit(&changed[(gid.y / cfg.w) * cfg.x + gid.x / cfg.z], 1u, memory_order_relaxed);\n"
     "    }\n"
     "}\n"
     "kernel void artcnn_dilate(device const uint *changed [[buffer(0)]], device uchar *dirty [[buffer(1)]],\n"
     "    constant uint4 &cfg [[buffer(2)]], uint2 gid [[thread_position_in_grid]])\n"
     "{\n"
     "    // cfg = (cells_x, cells_y, reach_x, reach_y) in cells\n"
     "    if (gid.x >= cfg.x || gid.y >= cfg.y) {\n"
     "        return;\n"
     "    }\n"
     "    bool any = false;\n"
     "    for (int dy = -int(cfg.w); dy <= int(cfg.w); dy++) {\n"
     "        for (int dx = -int(cfg.z); dx <= int(cfg.z); dx++) {\n"
     "            const int2 c = int2(gid) + int2(dx, dy);\n"
     "            if (all(c >= 0) && c.x < int(cfg.x) && c.y < int(cfg.y)) {\n"
     "                any |= changed[c.y * cfg.x + c.x] != 0;\n"
     "            }\n"
     "        }\n"
     "    }\n"
     "    dirty[gid.y * cfg.x + gid.x] = any;\n"
     "}\n";

@implementation ArtCNNMetal {
    id<MTLDevice> _device;
    NSArray<ArtCNNPass *> *_passes;
    NSMutableDictionary<NSString *, id<MTLTexture>> *_textures;
    NSUInteger _width;
    NSUInteger _height;
    BOOL _float32;
    // Frame-to-frame reuse (see reuseUnchangedBlocks).
    NSUInteger _reuseRadiusX;
    NSUInteger _reuseRadiusY;
    MTLSize _cell;
    id<MTLComputePipelineState> _changedPipeline;
    id<MTLComputePipelineState> _dilatePipeline;
    id<MTLBuffer> _changed;
    id<MTLBuffer> _dirty;
    uint32_t _cellsX;
    uint32_t _cellsY;
    id<MTLTexture> _previousLuma;
    __weak id<MTLTexture> _previousOutput;
    BOOL _haveHistory;
    BOOL _maskAllDirty;  // the mask is known to be all ones
    NSUInteger _outputScale;
    MTLPixelFormat _outputFormat;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device source:(NSString *)source error:(NSError **)error
{
    if (!(self = [super init])) {
        return nil;
    }
    _device = device;
    _textures = [NSMutableDictionary dictionary];

    if ([source containsString:@"thread_elements()"] && !checkLaneLayout(device, error)) {
        return nil;
    }

    MTLCompileOptions *options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_1;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:error];
    if (!library) {
        return nil;
    }

    // The pass manifest is emitted by translate.py as "// PASS <n> key=value ..." lines.
    NSRegularExpression *re = [NSRegularExpression regularExpressionWithPattern:
        @"^// PASS (\\d+) binds=(\\S+) save=(\\S+) block=(\\d+)x(\\d+) threads=(\\d+)x(\\d+) scale=([\\d.]+)x([\\d.]+)"
                                                                       options:NSRegularExpressionAnchorsMatchLines
                                                                         error:nil];
    NSMutableArray<ArtCNNPass *> *passes = [NSMutableArray array];
    for (NSTextCheckingResult *m in [re matchesInString:source options:0 range:NSMakeRange(0, source.length)]) {
        NSString * (^group)(NSUInteger) = ^(NSUInteger i) { return [source substringWithRange:[m rangeAtIndex:i]]; };
        ArtCNNPass *pass = [ArtCNNPass new];
        NSString *name = [NSString stringWithFormat:@"artcnn_pass%@", group(1)];
        id<MTLFunction> function = [library newFunctionWithName:name];
        pass.pipeline = function ? [device newComputePipelineStateWithFunction:function error:error] : nil;
        if (!pass.pipeline) {
            return nil;
        }
        pass.name = [NSString stringWithFormat:@"%@: %@", name, [source substringWithRange:NSMakeRange(NSMaxRange(m.range), [source lineRangeForRange:m.range].location + [source lineRangeForRange:m.range].length - NSMaxRange(m.range))]];
        pass.binds = [group(2) componentsSeparatedByString:@","];
        pass.saves = [group(3) componentsSeparatedByString:@","];
        pass.block = MTLSizeMake(group(4).integerValue, group(5).integerValue, 1);
        pass.threads = MTLSizeMake(group(6).integerValue, group(7).integerValue, 1);
        pass.scaleX = group(8).floatValue;
        pass.scaleY = group(9).floatValue;
        [passes addObject:pass];
    }
    if (passes.count == 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"ArtCNNMetal" code:1
                                     userInfo:@{NSLocalizedDescriptionKey: @"No passes found in shader source"}];
        }
        return nil;
    }
    _passes = passes;
    _float32 = [source containsString:@"\n// PRECISION float32\n"];
    // Default output: the luma doubled (ArtCNN). Shaders can declare another one.
    _outputScale = 2;
    _outputFormat = _float32 ? MTLPixelFormatR32Float : MTLPixelFormatR16Float;
    if ([source containsString:@"\n// OUTPUT scale=1 format=rgba16f\n"]) {
        _outputScale = 1;
        _outputFormat = MTLPixelFormatRGBA16Float;
    }

    NSTextCheckingResult *reuse = [[NSRegularExpression regularExpressionWithPattern:@"^// REUSE radius=(\\d+)x(\\d+) cell=(\\d+)x(\\d+)"
                                                                            options:NSRegularExpressionAnchorsMatchLines
                                                                              error:nil]
                                   firstMatchInString:source options:0 range:NSMakeRange(0, source.length)];
    if (reuse) {
        _reuseRadiusX = [source substringWithRange:[reuse rangeAtIndex:1]].integerValue;
        _reuseRadiusY = [source substringWithRange:[reuse rangeAtIndex:2]].integerValue;
        _cell = MTLSizeMake([source substringWithRange:[reuse rangeAtIndex:3]].integerValue,
                            [source substringWithRange:[reuse rangeAtIndex:4]].integerValue, 1);
        id<MTLLibrary> maskLibrary = [device newLibraryWithSource:kDirtyMaskSource options:nil error:error];
        id<MTLFunction> changed = [maskLibrary newFunctionWithName:@"artcnn_changed"];
        id<MTLFunction> dilate = [maskLibrary newFunctionWithName:@"artcnn_dilate"];
        _changedPipeline = changed ? [device newComputePipelineStateWithFunction:changed error:error] : nil;
        _dilatePipeline = dilate ? [device newComputePipelineStateWithFunction:dilate error:error] : nil;
        if (!_changedPipeline || !_dilatePipeline) {
            return nil;
        }
    }
    return self;
}

- (id<MTLTexture>)newTextureWithFormat:(MTLPixelFormat)format width:(NSUInteger)width height:(NSUInteger)height
{
    return [self newTextureWithFormat:format width:width height:height slices:1];
}

- (id<MTLTexture>)newTextureWithFormat:(MTLPixelFormat)format
                                 width:(NSUInteger)width
                                height:(NSUInteger)height
                                slices:(NSUInteger)slices
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    if (slices > 1) {
        desc.textureType = MTLTextureType2DArray;
        desc.arrayLength = slices;
    }
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    return [_device newTextureWithDescriptor:desc];
}

- (id<MTLTexture>)newOutputTextureForInputWidth:(NSUInteger)width height:(NSUInteger)height
{
    return [self newTextureWithFormat:_outputFormat width:width * _outputScale height:height * _outputScale];
}

- (BOOL)prepareForWidth:(NSUInteger)width height:(NSUInteger)height
{
    if (width == _width && height == _height) {
        return YES;
    }
    [_textures removeAllObjects];
    for (ArtCNNPass *pass in _passes) {
        for (NSString *save in pass.saves) {
        if ([save isEqualToString:@"OUTPUT"]) {
            continue;
        }
        // A pass at 2x scale produces 16 channels, stored as 4 slices at input size;
        // the generated kernels map the GLSL's 2x2 tiled coordinates onto the slices.
        BOOL tiled = pass.scaleX == 2 && pass.scaleY == 2;
        id<MTLTexture> texture = [self newTextureWithFormat:_float32 ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA16Float
                                                      width:tiled ? width : (NSUInteger)(width * pass.scaleX)
                                                     height:tiled ? height : (NSUInteger)(height * pass.scaleY)
                                                     slices:tiled ? 4 : 1];
        if (!texture) {
            return NO;
        }
        _textures[save] = texture;
        }
    }
    _width = width;
    _height = height;

    // The mask is all ones unless reuse finds clean cells, so shaders that read it work either way.
    NSUInteger cellW = _cell.width ? _cell.width : 16, cellH = _cell.height ? _cell.height : 8;
    _cellsX = (uint32_t)((width + cellW - 1) / cellW);
    _cellsY = (uint32_t)((height + cellH - 1) / cellH);
    _dirty = [_device newBufferWithLength:_cellsX * _cellsY options:MTLResourceStorageModeShared];
    _changed = [_device newBufferWithLength:_cellsX * _cellsY * sizeof(uint32_t) options:MTLResourceStorageModePrivate];
    memset(_dirty.contents, 1, _dirty.length);
    _maskAllDirty = YES;
    _previousLuma = nil;
    _haveHistory = NO;
    return YES;
}

- (double)lastDirtyFraction
{
    const uint8_t *d = _dirty.contents;
    NSUInteger n = 0;
    for (NSUInteger i = 0; i < _dirty.length; i++) {
        n += d[i] != 0;
    }
    return _dirty.length ? (double)n / _dirty.length : 1.0;
}

- (NSUInteger)passCount
{
    return _passes.count;
}

- (BOOL)encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                        input:(id<MTLTexture>)input
                       output:(id<MTLTexture>)output
{
    return [self encodeToCommandBuffer:commandBuffer input:input output:output passes:NSMakeRange(0, _passes.count)];
}

- (BOOL)encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                        input:(id<MTLTexture>)input
                       output:(id<MTLTexture>)output
                       passes:(NSRange)range
{
    if (![self prepareForWidth:input.width height:input.height]) {
        return NO;
    }

    BOOL fullFrame = range.location == 0 && range.length == _passes.count;
    BOOL reuse = _reuseUnchangedBlocks && _changedPipeline && fullFrame && _haveHistory &&
                 output == _previousOutput && _previousLuma.pixelFormat == input.pixelFormat;

    if (fullFrame && (reuse || !_maskAllDirty)) {
        // Filled on the GPU timeline so frames still in flight keep the mask they were encoded with.
        id<MTLBlitCommandEncoder> fill = [commandBuffer blitCommandEncoder];
        if (reuse) {
            [fill fillBuffer:_changed range:NSMakeRange(0, _changed.length) value:0];
        } else {
            [fill fillBuffer:_dirty range:NSMakeRange(0, _dirty.length) value:1];
        }
        [fill endEncoding];
        _maskAllDirty = !reuse;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (reuse) {
        uint32_t cellCfg[4] = { _cellsX, _cellsY, (uint32_t)_cell.width, (uint32_t)_cell.height };
        [encoder setComputePipelineState:_changedPipeline];
        [encoder setTexture:input atIndex:0];
        [encoder setTexture:_previousLuma atIndex:1];
        [encoder setBuffer:_changed offset:0 atIndex:0];
        [encoder setBytes:cellCfg length:sizeof(cellCfg) atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(input.width, input.height, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
        uint32_t reach[4] = { _cellsX, _cellsY,
                              (uint32_t)((_reuseRadiusX + _cell.width - 1) / _cell.width),
                              (uint32_t)((_reuseRadiusY + _cell.height - 1) / _cell.height) };
        [encoder setComputePipelineState:_dilatePipeline];
        [encoder setBuffer:_changed offset:0 atIndex:0];
        [encoder setBuffer:_dirty offset:0 atIndex:1];
        [encoder setBytes:reach length:sizeof(reach) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(_cellsX, _cellsY, 1) threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
    }
    for (ArtCNNPass *pass in [_passes subarrayWithRange:range]) {
        [encoder pushDebugGroup:pass.name];
        [encoder setComputePipelineState:pass.pipeline];
        NSUInteger index = 0;
        for (NSString *bind in pass.binds) {
            BOOL isInput = [bind isEqualToString:@"LUMA"] || [bind isEqualToString:@"MAIN"];
            [encoder setTexture:isInput ? input : _textures[bind] atIndex:index++];
        }
        for (NSString *save in pass.saves) {
            [encoder setTexture:[save isEqualToString:@"OUTPUT"] ? output : _textures[save] atIndex:index++];
        }
        [encoder setBuffer:_dirty offset:0 atIndex:0];
        [encoder setBytes:&_cellsX length:sizeof(_cellsX) atIndex:1];

        // Block sizes are in the pass's GLSL output coordinates (input size times scale).
        NSUInteger outW = (NSUInteger)(input.width * pass.scaleX), outH = (NSUInteger)(input.height * pass.scaleY);
        MTLSize groups = MTLSizeMake((outW + pass.block.width - 1) / pass.block.width,
                                     (outH + pass.block.height - 1) / pass.block.height,
                                     1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:pass.threads];
        [encoder popDebugGroup];
    }
    [encoder endEncoding];

    if (_reuseUnchangedBlocks && _changedPipeline && fullFrame) {
        if (!_previousLuma || _previousLuma.pixelFormat != input.pixelFormat) {
            MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:input.pixelFormat
                                                                                            width:input.width
                                                                                           height:input.height
                                                                                        mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModePrivate;
            _previousLuma = [_device newTextureWithDescriptor:desc];
        }
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        [blit copyFromTexture:input toTexture:_previousLuma];
        [blit endEncoding];
        _previousOutput = output;
        _haveHistory = YES;
    }
    return YES;
}

@end
