#!/bin/zsh
# Verify and time a generated shader against ONNX and the committed shader.
#
# tools/check.sh [-p] [candidate.metal]
#
# Without a candidate, regenerates ArtCNN_C4F16_DS.metal from winograd.py into build/.
# Checks the candidate against the ONNX reference (1600x1300 and 97x61, edges included)
# and against the committed shader, then times both alternately at 1512x982.
# -p adds a per-pass time breakdown of the candidate.
set -e
cd "${0:A:h}/.."
P=../.venv-metal/bin/python
profile=0
if [[ "$1" == "-p" ]]; then profile=1; shift; fi

if [[ ! -x build/artcnn_metal_cli || tools/artcnn_metal_cli.m -nt build/artcnn_metal_cli || ArtCNNMetal.m -nt build/artcnn_metal_cli ]]; then
    clang -fobjc-arc -O2 tools/artcnn_metal_cli.m ArtCNNMetal.m -o build/artcnn_metal_cli \
        -framework Foundation -framework Metal -framework QuartzCore
fi

candidate=${1:-build/candidate.metal}
if [[ -z "$1" ]]; then
    $P winograd.py ../GLSL/ArtCNN_C4F16_DS.glsl $candidate
fi
git show HEAD:Metal/ArtCNN_C4F16_DS.metal > build/committed.metal

# Cached references: luma inputs and ONNX outputs.
if [[ ! -f build/onnx.f32 ]]; then
    $P tools/reference.py prepare ../Images/artcnn_examples.png build/in.f32 > /dev/null
    $P tools/reference.py onnx ../ONNX/ArtCNN_C4F16_DS.onnx build/in.f32 1600 1300 build/onnx.f32
fi
if [[ ! -f build/odd_o_97x61.f32 ]]; then
    $P -c "import numpy as np; np.fromfile('build/in.f32', np.float32).reshape(1300, 1600)[:61, :97].copy().tofile('build/odd_97x61.f32')"
    $P tools/reference.py onnx ../ONNX/ArtCNN_C4F16_DS.onnx build/odd_97x61.f32 97 61 build/odd_o_97x61.f32
fi
if [[ ! -f build/in_1512x982.f32 ]]; then
    $P -c "
import numpy as np
from PIL import Image
x = np.fromfile('build/in.f32', np.float32).reshape(1300, 1600)
np.asarray(Image.fromarray(x, mode='F').resize((1512, 982), Image.BICUBIC), dtype=np.float32).clip(0, 1).tofile('build/in_1512x982.f32')"
fi

run() { perl -e 'alarm 120; exec @ARGV' ./build/artcnn_metal_cli "$@"; }
run $candidate build/in.f32 1600 1300 build/c_big.f32 > /dev/null
run $candidate build/odd_97x61.f32 97 61 build/c_odd.f32 > /dev/null
run build/committed.metal build/in.f32 1600 1300 build/k_big.f32 > /dev/null
echo -n "vs ONNX 1600x1300:   "; $P tools/reference.py compare build/onnx.f32 build/c_big.f32 3200 2600
echo -n "vs ONNX 97x61:       "; $P tools/reference.py compare build/odd_o_97x61.f32 build/c_odd.f32 194 122
echo -n "vs committed shader: "; $P tools/reference.py compare build/k_big.f32 build/c_big.f32 3200 2600

for rep in 1 2; do
    for v in committed candidate; do
        f=build/committed.metal; [[ $v == candidate ]] && f=$candidate
        run $f build/in_1512x982.f32 1512 982 /tmp/check_out.f32 150 | awk -v v=$v '/pipelined/{printf "%-10s %s ms per frame\n", v, $5}'
    done
done
if (( profile )); then
    ARTCNN_PROFILE=1 run $candidate build/in_1512x982.f32 1512 982 /tmp/check_out.f32 150 | grep pass
fi
