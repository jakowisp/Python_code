import platform
import tflite_runtime.interpreter as tflite
import sys

_EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


#model_path='qaware_mnist_quant_edgetpu.tflite'
model_path=sys.argv[1]
interpreter = tflite.Interpreter(model_path=model_path)
for op in  interpreter._get_ops_details():
    print(op)
if 'edgetpu-custom-op' in [op['op_name'] for op in  interpreter._get_ops_details()]:
    try:
        interpreter = tflite.Interpreter(
           model_path=model_path, experimental_delegates=[tflite.load_delegate(_EDGETPU_SHARED_LIB, {})])
    except ValueError:
        print("Failed to find USB Coral Edge device")
        sys.exit()
interpreter.allocate_tensors()
