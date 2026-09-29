#!/bin/bash
set -e

PT_ENV=/opt/vitis_ai/conda/envs/vitis-ai-pytorch
TF_ENV=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow
ARCH=${PT_ENV}/lib/python3.7/site-packages/vaic/arch/DPUCZDX8G/KV260/arch.json

INPUT_XMODEL=${1:-/workspace/quantize_result_det/GraphModule_int.xmodel}
OUTPUT_DIR=/workspace/compiled_model_det

export PATH=${TF_ENV}/bin:${PATH}
export PYTHONPATH=${PT_ENV}/lib/python3.7/site-packages/vaic:${PYTHONPATH}

mkdir -p "${OUTPUT_DIR}"

${PT_ENV}/bin/vai_c_xir \
  --xmodel  "${INPUT_XMODEL}" \
  --arch    "${ARCH}" \
  --net_name scrfd_det10g \
  --output_dir "${OUTPUT_DIR}"

echo "Compiled → ${OUTPUT_DIR}/scrfd_det10g.xmodel"
