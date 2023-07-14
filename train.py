import os
import matplotlib.pyplot as plt
import flatbuffers
import tensorflow as tf
import logging
import pathlib

from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

assert tf.__version__.startswith('2')


class MetadataPopulator(object):
    """Populates the metadata for an image classifier."""

    def __init__(self, model_file, label_file_path, name, version, image_width, image_height, image_min,
                 image_max, mean, std, num_classes, author):
        self.model_file = model_file
        self.label_file_path = label_file_path
        self.metadata_buf = None
        self.name = name
        self.version = version
        self.image_width = image_width
        self.image_height = image_height
        self.image_min = image_min
        self.image_max = image_max
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
        self.author = author

    def populate(self):
        """Creates metadata and then populates it for an image classifier."""
        self._create_metadata()
        self._populate_metadata()

    def _create_metadata(self):
        """Creates the metadata for an image classifier."""

        # Creates model info.
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.name
        model_meta.description = ("Identify the most prominent object in the "
                                  "image from a set of %d categories." %
                                  self.num_classes)
        model_meta.version = self.version
        model_meta.author = self.author
        model_meta.license = ("Apache License. Version 2.0 "
                              "http://www.apache.org/licenses/LICENSE-2.0.")

        # Creates input info.
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = (
            "Input image to be classified. The expected image is {0} x {1}, with "
            "three channels (red, blue, and green) per pixel. Each value in the "
            "tensor is a single byte between {2} and {3}.".format(
                self.image_width, self.image_height,
                self.image_min, self.image_max))
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = (
            _metadata_fb.ColorSpaceType.RGB)
        input_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.ImageProperties)
        input_normalization = _metadata_fb.ProcessUnitT()
        input_normalization.optionsType = (
            _metadata_fb.ProcessUnitOptions.NormalizationOptions)
        input_normalization.options = _metadata_fb.NormalizationOptionsT()
        input_normalization.options.mean = self.mean
        input_normalization.options.std = self.std
        input_meta.processUnits = [input_normalization]
        input_stats = _metadata_fb.StatsT()
        input_stats.max = [self.image_max]
        input_stats.min = [self.image_min]
        input_meta.stats = input_stats

        # Creates output info.
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "probability"
        output_meta.description = "Probabilities of the %d labels respectively." % self.num_classes
        output_meta.content = _metadata_fb.ContentT()
        output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
        output_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.FeatureProperties)
        output_stats = _metadata_fb.StatsT()
        output_stats.max = [1.0]
        output_stats.min = [0.0]
        output_meta.stats = output_stats
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = os.path.basename(self.label_file_path)
        label_file.description = "Labels for objects that the model can recognize."
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
        output_meta.associatedFiles = [label_file]

        # Creates subgraph info.
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(
            model_meta.Pack(b),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        self.metadata_buf = b.Output()

    def _populate_metadata(self):
        """Populates metadata and label file to the model file."""
        populator = _metadata.MetadataPopulator.with_model_file(self.model_file)
        populator.load_metadata_buffer(self.metadata_buf)
        populator.load_associated_files([self.label_file_path])
        populator.populate()


image_path = tf.keras.utils.get_file(
    'flower_photos.tgz',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    extract=True)
data_dir = os.path.join(os.path.dirname(image_path), 'flower_photos')

print("directory: {}".format(data_dir))

BATCH_SIZE = 32  # Number of training examples to process before updating our models variables
IMG_SHAPE = 180  # Our training data consists of images with width of 256 pixels and height of 256 pixels

# generator for training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    batch_size=BATCH_SIZE,
    shuffle=True,
    image_size=(IMG_SHAPE, IMG_SHAPE))

# generator for validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    batch_size=BATCH_SIZE,
    shuffle=True,
    image_size=(IMG_SHAPE, IMG_SHAPE))

class_names = train_ds.class_names
num_classes = len(class_names)
print('labels: {}'.format(class_names))

label_file_path = 'labels.txt'
with open(label_file_path, 'w') as f:
    f.write('\n'.join(class_names))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(input_shape=(IMG_SHAPE,
                                                IMG_SHAPE,
                                                3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

model = tf.keras.models.Sequential([
    data_augmentation,

    tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),

    tf.keras.layers.Normalization(),

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 25
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# quantization, set the optimization mode
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

tflite_model = converter.convert()

# save as tflite file
model_file = "model.tflite"
tflite_model_file = pathlib.Path(model_file)
tflite_model_file.write_bytes(tflite_model)

# copies model_file to export_path
export_model_path = "model_with_metadata.tflite"
tf.io.gfile.copy(model_file, export_model_path, overwrite=True)

# generate the metadata objects and put them in the model file
populator = MetadataPopulator(model_file=export_model_path,
                              label_file_path=label_file_path,
                              name="flower_classifier",
                              version="v1",
                              image_width=IMG_SHAPE,
                              image_height=IMG_SHAPE,
                              image_min=0,
                              image_max=255,
                              mean=[0],
                              std=[1],
                              num_classes=num_classes,
                              author="YOUR_NAME")
populator.populate()

# display history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
