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
        if (!source || inputData.length != width * height * sizeof(float)) {
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
        MTLTextureDescriptor *inDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:float32 ? MTLPixelFormatR32Float : MTLPixelFormatR16Float
                                                                                         width:width height:height mipmapped:NO];
        inDesc.usage = MTLTextureUsageShaderRead;
        inDesc.storageMode = MTLStorageModeShared;
        id<MTLTexture> input = [device newTextureWithDescriptor:inDesc];
        const float *src = inputData.bytes;
        __fp16 *halfPixels = malloc(width * height * sizeof(__fp16));
        for (NSUInteger i = 0; i < width * height; i++) {
            halfPixels[i] = (__fp16)src[i];
        }
        [input replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0
                   withBytes:float32 ? (const void *)src : halfPixels bytesPerRow:width * texel];
        free(halfPixels);

        id<MTLTexture> output = [artcnn newOutputTextureForInputWidth:width height:height];
        id<MTLCommandQueue> queue = [device newCommandQueue];

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

        NSUInteger outW = width * 2, outH = height * 2;
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
        float *outFloat = malloc(outW * outH * sizeof(float));
        for (NSUInteger i = 0; i < outW * outH; i++) {
            outFloat[i] = float32 ? ((const float *)readback.contents)[i] : outHalf[i];
        }
        [[NSData dataWithBytesNoCopy:outFloat length:outW * outH * sizeof(float)] writeToFile:@(argv[5]) atomically:NO];

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
