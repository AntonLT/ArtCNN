// mpsgraph_cli <weights.bin> <layers.txt> <input.f32> <width> <height> <output.f32> [iterations]
//
// Runs the ArtCNN network with MPSGraph convolutions in float16, using weights written by
// export_weights.py, and reports wall-clock time per frame.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <QuartzCore/QuartzCore.h>

static NSData *halfData(const float *values, NSUInteger count)
{
    NSMutableData *data = [NSMutableData dataWithLength:count * sizeof(__fp16)];
    __fp16 *out = data.mutableBytes;
    for (NSUInteger i = 0; i < count; i++) {
        out[i] = (__fp16)values[i];
    }
    return data;
}

int main(int argc, const char *argv[])
{
    @autoreleasepool {
        if (argc < 7) {
            fprintf(stderr, "usage: %s weights.bin layers.txt input.f32 width height output.f32 [iterations]\n", argv[0]);
            return 2;
        }
        NSData *weights = [NSData dataWithContentsOfFile:@(argv[1])];
        NSArray<NSString *> *layers = [[[NSString stringWithContentsOfFile:@(argv[2]) encoding:NSUTF8StringEncoding error:nil]
                                         stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]
                                        componentsSeparatedByString:@"\n"];
        NSData *inputData = [NSData dataWithContentsOfFile:@(argv[3])];
        NSUInteger width = strtoul(argv[4], NULL, 10), height = strtoul(argv[5], NULL, 10);
        int iterations = argc > 7 ? atoi(argv[7]) : 0;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        MPSGraph *graph = [MPSGraph new];
        MPSShape *inShape = @[@1, @(height), @(width), @1];
        MPSGraphTensor *input = [graph placeholderWithShape:inShape dataType:MPSDataTypeFloat16 name:@"luma"];

        MPSGraphConvolution2DOpDescriptor *conv = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1 groups:1
                       paddingStyle:MPSGraphPaddingStyleTF_SAME
                         dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        const float *w = weights.bytes;
        MPSGraphTensor *x = input, *first = nil;
        for (NSString *line in layers) {
            NSArray<NSString *> *f = [line componentsSeparatedByString:@","];
            NSUInteger outC = f[0].integerValue, inC = f[1].integerValue;
            BOOL relu = f[2].boolValue, skip = f[3].boolValue;
            NSUInteger wCount = outC * inC * 9;
            MPSGraphTensor *kernel = [graph constantWithData:halfData(w, wCount) shape:@[@(outC), @(inC), @3, @3] dataType:MPSDataTypeFloat16];
            w += wCount;
            MPSGraphTensor *bias = [graph constantWithData:halfData(w, outC) shape:@[@1, @1, @1, @(outC)] dataType:MPSDataTypeFloat16];
            w += outC;
            if (skip) {
                x = [graph additionWithPrimaryTensor:x secondaryTensor:first name:nil];
            }
            x = [graph convolution2DWithSourceTensor:x weightsTensor:kernel descriptor:conv name:nil];
            x = [graph additionWithPrimaryTensor:x secondaryTensor:bias name:nil];
            if (relu) {
                x = [graph reLUWithTensor:x name:nil];
            }
            if (!first) {
                first = x;
            }
        }
        x = [graph depthToSpace2DTensor:x widthAxis:2 heightAxis:1 depthAxis:3 blockSize:2 usePixelShuffleOrder:NO name:nil];
        x = [graph clampWithTensor:x
                    minValueTensor:[graph constantWithScalar:0.0 dataType:MPSDataTypeFloat16]
                    maxValueTensor:[graph constantWithScalar:1.0 dataType:MPSDataTypeFloat16]
                              name:nil];

        MPSGraphDevice *gdev = [MPSGraphDevice deviceWithMTLDevice:device];
        MPSGraphShapedType *inType = [[MPSGraphShapedType alloc] initWithShape:inShape dataType:MPSDataTypeFloat16];
        MPSGraphExecutable *exe = [graph compileWithDevice:gdev feeds:@{input: inType} targetTensors:@[x] targetOperations:nil compilationDescriptor:nil];

        id<MTLBuffer> inBuf = [device newBufferWithBytes:halfData(inputData.bytes, width * height).bytes
                                                  length:width * height * sizeof(__fp16) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outBuf = [device newBufferWithLength:width * height * 4 * sizeof(__fp16) options:MTLResourceStorageModeShared];
        MPSGraphTensorData *inTD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf shape:inShape dataType:MPSDataTypeFloat16];
        MPSGraphTensorData *outTD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf shape:@[@1, @(height * 2), @(width * 2), @1] dataType:MPSDataTypeFloat16];
        id<MTLCommandQueue> queue = [device newCommandQueue];

        void (^runOnce)(BOOL) = ^(BOOL wait) {
            MPSCommandBuffer *cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            [exe encodeToCommandBuffer:cb inputsArray:@[inTD] resultsArray:@[outTD] executionDescriptor:nil];
            [cb commit];
            if (wait) {
                [cb waitUntilCompleted];
            }
        };
        for (int i = 0; i < 10; i++) {
            runOnce(YES);
        }
        if (iterations > 0) {
            CFTimeInterval start = CACurrentMediaTime();
            for (int i = 0; i < iterations - 1; i++) {
                runOnce(NO);
            }
            runOnce(YES);
            printf("mpsgraph wall clock, pipelined: %.3f ms per frame\n", (CACurrentMediaTime() - start) * 1000.0 / iterations);
        }

        const __fp16 *o = outBuf.contents;
        NSUInteger n = width * height * 4;
        float *outFloat = malloc(n * sizeof(float));
        for (NSUInteger i = 0; i < n; i++) {
            outFloat[i] = o[i];
        }
        [[NSData dataWithBytesNoCopy:outFloat length:n * sizeof(float)] writeToFile:@(argv[6]) atomically:NO];
    }
    return 0;
}
