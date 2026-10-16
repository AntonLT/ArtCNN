#import "ArtCNNMetal.h"

@interface ArtCNNPass : NSObject
@property (nonatomic) id<MTLComputePipelineState> pipeline;
@property (nonatomic) NSArray<NSString *> *binds;
@property (nonatomic) NSString *save;
@property (nonatomic) MTLSize block;
@property (nonatomic) MTLSize threads;
@property (nonatomic) float scaleX;
@property (nonatomic) float scaleY;
@end

@implementation ArtCNNPass
@end

@implementation ArtCNNMetal {
    id<MTLDevice> _device;
    NSArray<ArtCNNPass *> *_passes;
    NSMutableDictionary<NSString *, id<MTLTexture>> *_textures;
    NSUInteger _width;
    NSUInteger _height;
    BOOL _float32;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device source:(NSString *)source error:(NSError **)error
{
    if (!(self = [super init])) {
        return nil;
    }
    _device = device;
    _textures = [NSMutableDictionary dictionary];

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
        pass.binds = [group(2) componentsSeparatedByString:@","];
        pass.save = group(3);
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
    return self;
}

- (id<MTLTexture>)newTextureWithFormat:(MTLPixelFormat)format width:(NSUInteger)width height:(NSUInteger)height
{
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    return [_device newTextureWithDescriptor:desc];
}

- (id<MTLTexture>)newOutputTextureForInputWidth:(NSUInteger)width height:(NSUInteger)height
{
    return [self newTextureWithFormat:_float32 ? MTLPixelFormatR32Float : MTLPixelFormatR16Float
                                width:width * 2
                               height:height * 2];
}

- (BOOL)prepareForWidth:(NSUInteger)width height:(NSUInteger)height
{
    if (width == _width && height == _height) {
        return YES;
    }
    [_textures removeAllObjects];
    for (ArtCNNPass *pass in _passes) {
        if ([pass.save isEqualToString:@"OUTPUT"]) {
            continue;
        }
        id<MTLTexture> texture = [self newTextureWithFormat:_float32 ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA16Float
                                                      width:(NSUInteger)(width * pass.scaleX)
                                                     height:(NSUInteger)(height * pass.scaleY)];
        if (!texture) {
            return NO;
        }
        _textures[pass.save] = texture;
    }
    _width = width;
    _height = height;
    return YES;
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

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    for (ArtCNNPass *pass in [_passes subarrayWithRange:range]) {
        [encoder setComputePipelineState:pass.pipeline];
        NSUInteger index = 0;
        for (NSString *bind in pass.binds) {
            [encoder setTexture:[bind isEqualToString:@"LUMA"] ? input : _textures[bind] atIndex:index++];
        }
        id<MTLTexture> target = [pass.save isEqualToString:@"OUTPUT"] ? output : _textures[pass.save];
        [encoder setTexture:target atIndex:index];

        MTLSize groups = MTLSizeMake((target.width + pass.block.width - 1) / pass.block.width,
                                     (target.height + pass.block.height - 1) / pass.block.height,
                                     1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:pass.threads];
    }
    [encoder endEncoding];
    return YES;
}

@end
