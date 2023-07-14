import tensorflow as tf
import numpy as np

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(180, 180)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

with open('labels.txt', 'r') as f:
    lines = f.read().splitlines()

class_names = np.array(lines)
print('class_names: {}\n'.format(class_names))

interpreter = tf.lite.Interpreter(model_path='model.tflite')

# Obtain input and output details of the model.
print("n--------Input Details of Model-------------------n")
input_details = interpreter.get_input_details()
print(input_details)

print("n--------Output Details of Model-------------------n")
output_details = interpreter.get_output_details()
print(output_details)

print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')
print(classify_lite)

predictions = classify_lite(sequential_input=img_array)['outputs']
for i in range(len(predictions)):
    print("predictions[" + str(i) + "]: ", predictions[i])
score_lite = tf.nn.softmax(predictions)
for i in range(len(score_lite)):
    print("score_lite[" + str(i) + "]: ", score_lite[i])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)

print('\n')

interpreter = tf.lite.Interpreter(model_path='model_with_metadata.tflite')

# Obtain input and output details of the model.
print("n--------Input Details of Model-------------------n")
input_details = interpreter.get_input_details()
print(input_details)

print("n--------Output Details of Model-------------------n")
output_details = interpreter.get_output_details()
print(output_details)

# Now allocate tensors so that we can use the set_tensor() method to feed the processed_image
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

predictions = interpreter.get_tensor(output_details[0]['index'])
for i in range(len(predictions)):
    print("predictions[" + str(i) + "]: ", predictions[i])
score_lite = tf.nn.softmax(predictions)
for i in range(len(score_lite)):
    print("score_lite[" + str(i) + "]: ", score_lite[i])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)
