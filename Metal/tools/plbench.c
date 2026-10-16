// plbench <shader.glsl> <input.f32> <width> <height> <output.f32> [iterations]
//
// Runs an mpv user shader through libplacebo on Vulkan (MoltenVK on macOS), the way
// Moonlight's PlVkRenderer does, on a luma-only frame with neutral chroma. Writes the
// doubled luma and reports libplacebo's GPU timer results for the ArtCNN passes.
// With "none" as the shader it renders 1:1 without hooks, which measures the
// color path of this harness itself.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/vulkan.h>

extern PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance instance, const char *name);

#define MAX_FRAMES 4096

struct timings {
    int frame;
    double artcnn_ms[MAX_FRAMES];
    double total_ms[MAX_FRAMES];
};

static void info_cb(void *priv, const struct pl_render_info *info)
{
    struct timings *t = priv;
    if (t->frame < 0 || t->frame >= MAX_FRAMES || !info->pass->num_samples) {
        return;
    }
    double ms = info->pass->last / 1e6;
    t->total_ms[t->frame] += ms;
    if (info->pass->shader && info->pass->shader->description &&
        strstr(info->pass->shader->description, "ArtCNN")) {
        t->artcnn_ms[t->frame] += ms;
    }
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void report(const char *label, double *values, int n)
{
    qsort(values, n, sizeof(double), cmp_double);
    printf("%s: median %.3f ms, p10 %.3f ms, p90 %.3f ms over %d frames\n",
           label, values[n / 2], values[n / 10], values[n * 9 / 10], n);
}

static char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *data = malloc(*len + 1);
    *len = fread(data, 1, *len, f);
    data[*len] = 0;
    fclose(f);
    return data;
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s shader.glsl input.f32 width height output.f32 [iterations]\n", argv[0]);
        return 2;
    }
    int w = atoi(argv[3]), h = atoi(argv[4]);
    int iterations = argc > 6 ? atoi(argv[6]) : 0;
    if (iterations > MAX_FRAMES - 16) {
        iterations = MAX_FRAMES - 16;
    }
    size_t shader_len, input_len;
    int identity = strcmp(argv[1], "none") == 0;
    int scale = identity ? 1 : 2;
    char *shader_src = identity ? NULL : read_file(argv[1], &shader_len);
    float *luma = (float *)read_file(argv[2], &input_len);
    if ((!identity && !shader_src) || !luma || input_len != (size_t)w * h * sizeof(float)) {
        fprintf(stderr, "bad shader or input\n");
        return 2;
    }

    pl_log log = pl_log_create(PL_API_VER, pl_log_params(.log_cb = pl_log_color, .log_level = PL_LOG_WARN));
    pl_vk_inst inst = pl_vk_inst_create(log, pl_vk_inst_params(.get_proc_addr = vkGetInstanceProcAddr));
    pl_vulkan vk = inst ? pl_vulkan_create(log, pl_vulkan_params(.instance = inst->instance,
                                                                 .get_proc_addr = inst->get_proc_addr)) : NULL;
    if (!vk) {
        fprintf(stderr, "Vulkan init failed\n");
        return 1;
    }
    pl_gpu gpu = vk->gpu;

    const struct pl_hook *hook = identity ? NULL : pl_mpv_user_shader_parse(gpu, shader_src, shader_len);
    if (!identity && !hook) {
        fprintf(stderr, "shader parse failed\n");
        return 1;
    }

    // Luma as r16f (the shader converts to half anyway); neutral chroma so RGB == Y.
    uint16_t *luma_half = malloc((size_t)w * h * 2);
    for (int i = 0; i < w * h; i++) {
        _Float16 v = (_Float16)luma[i];
        memcpy(&luma_half[i], &v, 2);
    }
    int cw = (w + 1) / 2, ch = (h + 1) / 2;
    uint16_t *chroma_half = malloc((size_t)cw * ch * 4);
    _Float16 half_one_half = (_Float16)0.5f;
    for (int i = 0; i < cw * ch * 2; i++) {
        memcpy(&chroma_half[i], &half_one_half, 2);
    }

    pl_fmt r16f = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 16, PL_FMT_CAP_SAMPLEABLE);
    pl_fmt rg16f = pl_find_fmt(gpu, PL_FMT_FLOAT, 2, 16, 16, PL_FMT_CAP_SAMPLEABLE);
    pl_fmt rgba16f = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 16, PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
    if (!r16f || !rg16f || !rgba16f || r16f->texel_size != 2 || rg16f->texel_size != 4 || rgba16f->texel_size != 8) {
        fprintf(stderr, "missing half-float formats\n");
        return 1;
    }
    pl_tex y_tex = pl_tex_create(gpu, pl_tex_params(.w = w, .h = h, .format = r16f,
                                                    .sampleable = true, .host_writable = true,
                                                    .initial_data = luma_half));
    pl_tex c_tex = pl_tex_create(gpu, pl_tex_params(.w = cw, .h = ch, .format = rg16f,
                                                    .sampleable = true, .host_writable = true,
                                                    .initial_data = chroma_half));
    pl_tex out_tex = pl_tex_create(gpu, pl_tex_params(.w = w * scale, .h = h * scale,
                                                      .format = rgba16f,
                                                      .renderable = true, .host_readable = true));
    if (!y_tex || !c_tex || !out_tex) {
        fprintf(stderr, "texture creation failed\n");
        return 1;
    }

    struct pl_frame image = {
        .num_planes = 2,
        .planes = {
            { .texture = y_tex, .components = 1, .component_mapping = { PL_CHANNEL_Y } },
            { .texture = c_tex, .components = 2, .component_mapping = { PL_CHANNEL_CB, PL_CHANNEL_CR } },
        },
        .repr = { .sys = PL_COLOR_SYSTEM_BT_709, .levels = PL_COLOR_LEVELS_FULL },
        .color = pl_color_space_bt709,
    };
    pl_chroma_location_offset(PL_CHROMA_LEFT, &image.planes[1].shift_x, &image.planes[1].shift_y);
    struct pl_frame target = {
        .num_planes = 1,
        .planes = { { .texture = out_tex, .components = 4, .component_mapping = { 0, 1, 2, 3 } } },
        .repr = pl_color_repr_rgb,
        .color = pl_color_space_bt709,
    };

    struct timings *t = calloc(1, sizeof(*t));
    t->frame = -1;
    struct pl_render_params params = pl_render_fast_params;
    params.hooks = &hook;
    params.num_hooks = identity ? 0 : 1;
    params.info_callback = info_cb;
    params.info_priv = t;
    params.dither_params = NULL;

    pl_renderer renderer = pl_renderer_create(log, gpu);

    // ARTCNN_FPS=n ARTCNN_SECONDS=s renders one frame per 1/n s for s seconds, waiting for
    // each frame, like a paced stream; used for power measurements.
    if (getenv("ARTCNN_FPS")) {
        double period = 1000.0 / atof(getenv("ARTCNN_FPS"));
        double seconds = getenv("ARTCNN_SECONDS") ? atof(getenv("ARTCNN_SECONDS")) : 10.0;
        double start = now_ms(), next = start;
        int frames = 0, late = 0;
        params.info_callback = NULL;
        while (now_ms() - start < seconds * 1000.0) {
            pl_render_image(renderer, &image, &target, &params);
            pl_gpu_finish(gpu);
            frames++;
            next += period;
            double wait = next - now_ms();
            if (wait > 0) {
                struct timespec ts = { (time_t)(wait / 1000.0), (long)(fmod(wait, 1000.0) * 1e6) };
                nanosleep(&ts, NULL);
            } else {
                late++;
            }
        }
        printf("libplacebo paced: %d frames in %.2f s, %d late\n", frames, (now_ms() - start) / 1000.0, late);
        return 0;
    }
    // libplacebo reports a pass's timer result on a later frame, so warm up first
    // and attribute each frame's callbacks to that frame index.
    int total = iterations > 0 ? iterations + 16 : 1;
    for (int i = 0; i < total; i++) {
        t->frame = i - 16;
        if (!pl_render_image(renderer, &image, &target, &params)) {
            fprintf(stderr, "pl_render_image failed\n");
            return 1;
        }
        pl_gpu_finish(gpu);
    }

    size_t out_pixels = (size_t)w * h * scale * scale;
    // Wall-clock timing: one frame at a time, then a pipelined batch.
    double *wall = malloc(sizeof(double) * (iterations > 0 ? iterations : 1));
    params.info_callback = NULL;
    for (int i = 0; i < iterations; i++) {
        double start = now_ms();
        pl_render_image(renderer, &image, &target, &params);
        pl_gpu_finish(gpu);
        wall[i] = now_ms() - start;
    }
    double batch_ms = 0;
    if (iterations > 0) {
        double start = now_ms();
        for (int i = 0; i < iterations; i++) {
            pl_render_image(renderer, &image, &target, &params);
        }
        pl_gpu_finish(gpu);
        batch_ms = (now_ms() - start) / iterations;
    }

    float *out = malloc(out_pixels * sizeof(float));
    uint16_t *rgba_half = malloc(out_pixels * 4 * 2);
    if (!pl_tex_download(gpu, pl_tex_transfer_params(.tex = out_tex, .ptr = rgba_half))) {
        fprintf(stderr, "download failed\n");
        return 1;
    }
    for (size_t i = 0; i < out_pixels; i++) {
        _Float16 v;
        memcpy(&v, &rgba_half[i * 4], 2);
        out[i] = (float)v;
    }
    FILE *f = fopen(argv[5], "wb");
    fwrite(out, sizeof(float), out_pixels, f);
    fclose(f);

    if (iterations > 0) {
        report("libplacebo/MoltenVK ArtCNN passes", t->artcnn_ms, iterations);
        report("libplacebo/MoltenVK whole frame", t->total_ms, iterations);
        report("libplacebo/MoltenVK wall clock, one frame at a time", wall, iterations);
        printf("libplacebo/MoltenVK wall clock, pipelined: %.3f ms per frame\n", batch_ms);
    }

    pl_renderer_destroy(&renderer);
    pl_tex_destroy(gpu, &y_tex);
    pl_tex_destroy(gpu, &c_tex);
    pl_tex_destroy(gpu, &out_tex);
    if (hook) {
        pl_mpv_user_shader_destroy(&hook);
    }
    pl_vulkan_destroy(&vk);
    pl_vk_inst_destroy(&inst);
    pl_log_destroy(&log);
    return 0;
}
