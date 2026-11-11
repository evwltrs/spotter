# Half a model's precision: weights and the work inside it in 16-bit floats, which the graphics card
# does faster and moves half as much of, while its inputs and outputs stay 32-bit, so nothing that
# feeds or reads it changes. Uses ONNX Runtime's own converter (pip install onnxruntime), which keeps
# the inputs that must stay 32-bit (Resize's scales and the like) as they are.
#   python scripts/fp16.py model.onnx model.half.onnx
import os
import sys

import onnx
from onnxruntime.transformers import float16

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
half = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False)
big = half.ByteSize() > 1_900_000_000
# written as <name>.part and renamed once whole; its weights, if apart, under the final name
name = os.path.basename(dst).removesuffix('.part')
onnx.save(half, dst, save_as_external_data=big, all_tensors_to_one_file=True, location=(name + '_data') if big else None)
print('wrote', dst)
