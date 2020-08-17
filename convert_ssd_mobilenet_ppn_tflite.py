import tensorflow as tf


converter = tf.lite.TFLiteConverter.from_frozen_graph(
  '/tmp/cache/tflite/model_300.pb',
  input_arrays=['Preprocessor/sub',
                'Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3'],
  output_arrays=['Concatenate/concat', 'concat', 'concat_1'],
  input_shapes={'Preprocessor/sub': [1, 300, 300, 3],
                'Preprocessor/map/TensorArrayStack_1/TensorArrayGatherV3': [1, 3]})


tflite_model = converter.convert()
with open("/cache/tflite/converted_model.tflite", "wb") as f:
  f.write(tflite_model)

