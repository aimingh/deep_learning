'''
참고
https://eehoeskrap.tistory.com/414
https://colab.research.google.com/github/vinhngx/tensorrt/blob/vinhn-tf20-notebook/tftrt/examples/image-classification/TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb#scrollTo=rf97K_rxvwRm
https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
'''
from tensorflow.keras import models
from tensorflow.python.compiler.tensorrt import trt_convert as trt

model_path = 'Unet_small.h5'
save_pb_dir = './Unet_small'
output_saved_model_dir = './Unet_small_trt'

# keras h5 model load
model = models.load_model(model_path)
# keras pb model save
model.save(save_pb_dir, save_format='tf')

# convert keras pb to TensorRT pb
# optimize
print('Converting to TF-TRT FP16...')
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=8000000000)
converter = trt.TrtGraphConverterV2(
   input_saved_model_dir=save_pb_dir, conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir=output_saved_model_dir)
print('Done Converting to TF-TRT FP16')