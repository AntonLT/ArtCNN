// artcnn_metal_cli <shader.metal> <input.f32> <width> <height> <output.f32> [iterations]
//
// Reads a raw float32 luma image, doubles it with ArtCNNMetal and writes the raw
// float32 result. With iterations > 0 it also reports per-frame GPU time.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import "../ArtCNNMetal.h"

int main(int argc, const char *argv[])
{
    @autoreleasepool {
        if (argc < 6) {
            fprintf(stderr, "usage: %s shader.metal input.f32 width height output.f32 [iterations]\n", argv[0]);
            return 2;
        }
        NSString *source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
        NSData *inputData = [NSData dataWithContentsOfFile:@(argv[2])];
        NSUInteger width = strtoul(argv[3], NULL, 10), height = strtoul(argv[4], NULL, 10);
        int iterations = argc > 6 ? atoi(argv[6]) : 0;
        // ARTCNN_RGB=1: interleaved float32 RGB in and out at the input size (Anime4K Restore).
        BOOL rgbMode = getenv("ARTCNN_RGB") != NULL;
        NSUInteger inChannels = rgbMode ? 3 : 1;
        if (!source || inputData.length != width * height * inChannels * sizeof(float)) {
            fprintf(stderr, "bad shader or input size\n");
            return 2;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        ArtCNNMetal *artcnn = [[ArtCNNMetal alloc] initWithDevice:device source:source error:&error];
        if (!artcnn) {
            fprintf(stderr, "init failed: %s\n", error.description.UTF8String);
            return 1;
        }

        BOOL float32 = [source containsString:@"\n// PRECISION float32\n"];
        size_t texel = float32 ? sizeof(float) : sizeof(__fp16);
        MTLPixelFormat inFormat = rgbMode ? MTLPixelFormatRGBA16Float : float32 ? MTLPixelFormatR32Float : MTLPixelFormatR16Float;
        MTLTextureDescriptor *inDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:inFormat
                                                                                         width:width height:height mipmapped:NO];
        inDesc.usage = MTLTextureUsageShaderRead;
        inDesc.storageMode = MTLStorageModeShared;
        id<MTLTexture> input = [device newTextureWithDescriptor:inDesc];
        const float *src = inputData.bytes;
        if (rgbMode) {
            __fp16 *rgba = malloc(width * height * 4 * sizeof(__fp16));
            for (NSUInteger i = 0; i < width * height; i++) {
                for (NSUInteger c = 0; c < 4; c++) {
                    rgba[i * 4 + c] = (__fp16)(c < 3 ? src[i * 3 + c] : 1.0f);
                }
            }
            [input replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:rgba bytesPerRow:width * 8];
            free(rgba);
        } else {
            __fp16 *halfPixels = malloc(width * height * sizeof(__fp16));
            for (NSUInteger i = 0; i < width * height; i++) {
                halfPixels[i] = (__fp16)src[i];
            }
            [input replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0
                       withBytes:float32 ? (const void *)src : halfPixels bytesPerRow:width * texel];
            free(halfPixels);
        }

        id<MTLTexture> output = [artcnn newOutputTextureForInputWidth:width height:height];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // ARTCNN_REUSE_TEST=1: frame B is frame A with a rectangle inverted. With reuse on, A then
        // B must give byte-identical output to B computed in full; then time alternating A/B
        // frames for several rectangle sizes.
        if (getenv("ARTCNN_REUSE_TEST")) {
            NSUInteger outW = output.width, outH = output.height, outBytes = rgbMode ? 8 : 2;
            NSData * (^readback)(id<MTLTexture>) = ^(id<MTLTexture> texture) {
                id<MTLBuffer> buf = [device newBufferWithLength:outW * outH * outBytes options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(outW, outH, 1) toBuffer:buf destinationOffset:0
               destinationBytesPerRow:outW * outBytes destinationBytesPerImage:outW * outH * outBytes];
                [blit endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                return [NSData dataWithBytes:buf.contents length:buf.length];
            };
            void (^run)(ArtCNNMetal *, id<MTLTexture>, id<MTLTexture>) = ^(ArtCNNMetal *net, id<MTLTexture> in, id<MTLTexture> out) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                [net encodeToCommandBuffer:cb input:in output:out];
                [cb commit];
                [cb waitUntilCompleted];
            };
            for (NSNumber *fraction in @[@0.0, @0.02, @0.1, @0.3, @1.0]) {
                // B: invert a centered rectangle covering `fraction` of the image (all channels).
                const float *a = inputData.bytes;
                NSUInteger rw = (NSUInteger)(width * sqrt(fraction.doubleValue)), rh = (NSUInteger)(height * sqrt(fraction.doubleValue));
                NSUInteger x0 = (width - rw) / 2, y0 = (height - rh) / 2;
                NSUInteger components = rgbMode ? 4 : 1;
                __fp16 *pixels = malloc(width * height * components * sizeof(__fp16));
                for (NSUInteger y = 0; y < height; y++) {
                    for (NSUInteger x = 0; x < width; x++) {
                        BOOL inside = x >= x0 && x < x0 + rw && y >= y0 && y < y0 + rh;
                        for (NSUInteger c = 0; c < components; c++) {
                            float v = rgbMode ? (c < 3 ? a[(y * width + x) * 3 + c] : 1.0f) : a[y * width + x];
                            pixels[(y * width + x) * components + c] = (__fp16)(inside && c < 3 ? 1.0f - v : v);
                        }
                    }
                }
                id<MTLTexture> inputB = [device newTextureWithDescriptor:inDesc];
                [inputB replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:pixels
                          bytesPerRow:width * components * sizeof(__fp16)];
                free(pixels);

                ArtCNNMetal *reuse = [[ArtCNNMetal alloc] initWithDevice:device source:source error:&error];
                reuse.reuseUnchangedBlocks = YES;
                id<MTLTexture> out = [reuse newOutputTextureForInputWidth:width height:height];
                run(reuse, input, out);
                run(reuse, inputB, out);
                double dirty = [reuse lastDirtyFraction];
                NSData *got = readback(out);

                ArtCNNMetal *full = [[ArtCNNMetal alloc] initWithDevice:device source:source error:&error];
                id<MTLTexture> fullOut = [full newOutputTextureForInputWidth:width height:height];
                run(full, inputB, fullOut);
                NSData *want = readback(fullOut);
                BOOL same = [got isEqualToData:want];
                if (!same && getenv("ARTCNN_REUSE_DUMP")) {
                    [got writeToFile:@"/tmp/reuse_got.f16" atomically:NO];
                    [want writeToFile:@"/tmp/reuse_want.f16" atomically:NO];
                }

                int n = 200;
                CFTimeInterval start = CACurrentMediaTime();
                id<MTLCommandBuffer> last = nil;
                for (int i = 0; i < n; i++) {
                    last = [queue commandBuffer];
                    [reuse encodeToCommandBuffer:last input:(i & 1) ? input : inputB output:out];
                    [last commit];
                }
                [last waitUntilCompleted];
                printf("changed %4.0f%% of image: %5.1f%% of cells recomputed, output %s full computation, %.3f ms per frame\n",
                       fraction.doubleValue * 100, dirty * 100, same ? "identical to" : "DIFFERS from",
                       (CACurrentMediaTime() - start) * 1000.0 / n);
            }
            return 0;
        }

        // ARTCNN_CAPTURE=path.gputrace saves one full frame as a Metal GPU trace for Xcode
        // (needs MTL_CAPTURE_ENABLED=1). Xcode shows registers, spills and occupancy per kernel.
        if (getenv("ARTCNN_CAPTURE")) {
            MTLCaptureDescriptor *capture = [MTLCaptureDescriptor new];
            capture.captureObject = queue;
            capture.destination = MTLCaptureDestinationGPUTraceDocument;
            capture.outputURL = [NSURL fileURLWithPath:@(getenv("ARTCNN_CAPTURE"))];
            NSError *captureError = nil;
            if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:capture error:&captureError]) {
                fprintf(stderr, "capture failed: %s\n", captureError.description.UTF8String);
                return 1;
            }
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            cb.label = @"ArtCNN frame";
            [artcnn encodeToCommandBuffer:cb input:input output:output];
            [cb commit];
            [cb waitUntilCompleted];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
            printf("saved GPU trace to %s\n", getenv("ARTCNN_CAPTURE"));
        }

        // ARTCNN_FPS=n ARTCNN_SECONDS=s renders one frame per 1/n s for s seconds, waiting for
        // each frame, like a paced stream; used for power measurements.
        if (getenv("ARTCNN_FPS")) {
            double period = 1.0 / atof(getenv("ARTCNN_FPS"));
            double seconds = getenv("ARTCNN_SECONDS") ? atof(getenv("ARTCNN_SECONDS")) : 10.0;
            CFTimeInterval start = CACurrentMediaTime(), next = start;
            int frames = 0, late = 0;
            while (CACurrentMediaTime() - start < seconds) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                [artcnn encodeToCommandBuffer:cb input:input output:output];
                [cb commit];
                [cb waitUntilCompleted];
                frames++;
                next += period;
                double wait = next - CACurrentMediaTime();
                if (wait > 0) {
                    usleep((useconds_t)(wait * 1e6));
                } else {
                    late++;
                }
            }
            printf("metal paced: %d frames in %.2f s, %d late\n", frames, CACurrentMediaTime() - start, late);
            return 0;
        }

        // ARTCNN_ONLY_PASS=n runs just that pass in the timing loops, for counter captures.
        NSRange passes = NSMakeRange(0, artcnn.passCount);
        if (getenv("ARTCNN_ONLY_PASS")) {
            passes = NSMakeRange(strtoul(getenv("ARTCNN_ONLY_PASS"), NULL, 10), 1);
            [artcnn encodeToCommandBuffer:[queue commandBuffer] input:input output:output];
        }
        NSMutableArray<NSNumber *> *times = [NSMutableArray array];
        NSMutableArray<NSNumber *> *wall = [NSMutableArray array];
        for (int i = -10; i < iterations || i < 0; i++) {
            CFTimeInterval start = CACurrentMediaTime();
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            if (![artcnn encodeToCommandBuffer:cb input:input output:output passes:passes]) {
                fprintf(stderr, "encode failed\n");
                return 1;
            }
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                fprintf(stderr, "GPU error: %s\n", cb.error.description.UTF8String);
                return 1;
            }
            if (i >= 0) {
                [wall addObject:@((CACurrentMediaTime() - start) * 1000.0)];
                [times addObject:@((cb.GPUEndTime - cb.GPUStartTime) * 1000.0)];
            }
        }

        if (iterations > 0 && getenv("ARTCNN_PROFILE")) {
            for (NSUInteger p = 0; p < artcnn.passCount; p++) {
                CFTimeInterval start = CACurrentMediaTime();
                id<MTLCommandBuffer> last = nil;
                for (int i = 0; i < iterations; i++) {
                    last = [queue commandBuffer];
                    [artcnn encodeToCommandBuffer:last input:input output:output passes:NSMakeRange(p, 1)];
                    [last commit];
                }
                [last waitUntilCompleted];
                printf("  pass %lu: %.3f ms\n", (unsigned long)p, (CACurrentMediaTime() - start) * 1000.0 / iterations);
            }
        }

        double batchMs = 0;
        if (iterations > 0) {
            CFTimeInterval start = CACurrentMediaTime();
            id<MTLCommandBuffer> last = nil;
            for (int i = 0; i < iterations; i++) {
                last = [queue commandBuffer];
                [artcnn encodeToCommandBuffer:last input:input output:output passes:passes];
                [last commit];
            }
            [last waitUntilCompleted];
            batchMs = (CACurrentMediaTime() - start) * 1000.0 / iterations;
        }

        NSUInteger outW = output.width, outH = output.height;
        NSUInteger outComponents = rgbMode ? 4 : 1, outChannels = rgbMode ? 3 : 1;
        if (rgbMode) {
            texel = 4 * sizeof(__fp16);
        }
        id<MTLBuffer> readback = [device newBufferWithLength:outW * outH * texel options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:output sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(outW, outH, 1) toBuffer:readback destinationOffset:0
       destinationBytesPerRow:outW * texel destinationBytesPerImage:outW * outH * texel];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        const __fp16 *outHalf = readback.contents;
        float *outFloat = malloc(outW * outH * outChannels * sizeof(float));
        for (NSUInteger i = 0; i < outW * outH; i++) {
            for (NSUInteger c = 0; c < outChannels; c++) {
                outFloat[i * outChannels + c] = float32 && !rgbMode ? ((const float *)readback.contents)[i]
                                                                    : outHalf[i * outComponents + c];
            }
        }
        [[NSData dataWithBytesNoCopy:outFloat length:outW * outH * outChannels * sizeof(float)] writeToFile:@(argv[5]) atomically:NO];

        void (^report)(const char *, NSArray<NSNumber *> *) = ^(const char *label, NSArray<NSNumber *> *values) {
            NSArray *sorted = [values sortedArrayUsingSelector:@selector(compare:)];
            printf("metal %s: median %.3f ms, p10 %.3f ms, p90 %.3f ms over %lu frames\n", label,
                   [sorted[sorted.count / 2] doubleValue],
                   [sorted[sorted.count / 10] doubleValue],
                   [sorted[sorted.count * 9 / 10] doubleValue],
                   (unsigned long)sorted.count);
        };
        if (times.count) {
            report("GPU timestamps", times);
            report("wall clock, one frame at a time", wall);
            printf("metal wall clock, pipelined: %.3f ms per frame\n", batchMs);
        }
    }
    return 0;
}
