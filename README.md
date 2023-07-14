# Custom ML model for ML Kit

By default, ML Kit’s APIs make use of Google trained machine learning models. Both the [Image Labeling](https://developers.google.com/ml-kit/vision/image-labeling) and the [Object Detection & Tracking](https://developers.google.com/ml-kit/vision/object-detection) API offer support for custom image classification models. 

In this tutorial is shown how to create a [TensorFlow Lite](https://www.tensorflow.org/lite/) model and make it [compatible](https://developers.google.com/ml-kit/custom-models#model-compatibility) with [ML Kit](https://developers.google.com/ml-kit).

**NOTE**: Before jumping into coding, make sure you read and understand the ML Kit's compatibility requirements for TensorFlow Lite models [here](https://developers.google.com/ml-kit/custom-models). 

You can run this tutorial in [Google Colab](https://colab.research.google.com/github/flutter-ml/mlkit-custom-model/blob/main/ml_kit_custom_model.ipynb).

<td>
    <a target="_blank" href="https://colab.research.google.com/github/flutter-ml/mlkit-custom-model/blob/main/ml_kit_custom_model.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
</td>

Or you can clone this repo and run this in your local terminal: 

```bash
python3 ml_kit_custom_model.py
```

That will generate these files:

* model.tflite
* model_with_metadata.tflite
* labels.txt

You will need `model_with_metadata.tflite` to test in your mobile app using [ML Kit](https://developers.google.com/ml-kit).

You can use and tweak our [demo app](https://github.com/flutter-ml/mlkit-custom-model/blob/main/flutter_demo_app) to test your tflite model using [google_mlkit_image_labeling](https://pub.dev/packages/google_mlkit_image_labeling) and [google_mlkit_object_detection](https://pub.dev/packages/google_mlkit_object_detection) in [Flutter](https://flutter.dev/). 
