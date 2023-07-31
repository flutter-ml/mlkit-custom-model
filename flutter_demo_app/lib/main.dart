import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_image_labeling/google_mlkit_image_labeling.dart';
import 'package:google_mlkit_object_detection/google_mlkit_object_detection.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;
  final _images = <String>[];
  ImageLabeler? _imageLabeler;
  ObjectDetector? _objectDetector;
  final _modelPath = 'assets/ml/object_labeler_flowers.tflite';

  @override
  void initState() {
    _getAssets();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Custom Model Demo app'),
      ),
      body: _images.isEmpty
          ? Container()
          : Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  Image.asset(_images[_counter]),
                  FutureBuilder<List<ImageLabel>>(
                      future: _labelImage(_images[_counter]),
                      builder: (context, snapshot) {
                        final labels = snapshot.data ?? [];
                        var str = '...';
                        if (labels.isNotEmpty) {
                          final label = labels.reduce((curr, next) =>
                              curr.confidence > next.confidence ? curr : next);
                          str =
                              'IMAGE LABELER:\nlabeler response: ${label.label}\nconfidence: ${label.confidence.toStringAsFixed(2)}';
                        }
                        return Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: Text(str),
                        );
                      }),
                  FutureBuilder<List<DetectedObject>>(
                      future: _detectObject(_images[_counter]),
                      builder: (context, snapshot) {
                        final detectedObjects = snapshot.data ?? [];
                        var str = '...';
                        if (detectedObjects.isNotEmpty) {
                          str =
                              'OBJECT DETECTOR:\n${detectedObjects.map((e) => e.labels.map((e) => '${e.text} [${e.confidence.toStringAsFixed(2)}]').toString()).toList()}';
                        }
                        return Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: Text(str),
                        );
                      }),
                ],
              ),
            ),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Next',
        child: const Icon(Icons.navigate_next),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }

  void _getAssets() async {
    final manifestContent = await rootBundle.loadString('AssetManifest.json');
    final Map<String, dynamic> manifestMap = json.decode(manifestContent);
    final assets = manifestMap.keys
        .where((String key) => key.contains('images/'))
        .where((String key) =>
            key.contains('.jpg') ||
            key.contains('.jpeg') ||
            key.contains('.png') ||
            key.contains('.webp'))
        .toList();
    setState(() {
      _images.clear();
      _images.addAll(assets);
    });
  }

  void _incrementCounter() {
    setState(() {
      _counter++;
      _counter = _counter % _images.length;
    });
  }

  Future<List<ImageLabel>> _labelImage(String asset) async {
    if (_imageLabeler == null) {
      // initialize labeler
      final modelPath = await _getAssetPath(_modelPath);
      final options = LocalLabelerOptions(modelPath: modelPath);
      _imageLabeler = ImageLabeler(options: options);
    }
    // get and process image
    final inputImage = InputImage.fromFilePath(await _getAssetPath(asset));
    return (await _imageLabeler?.processImage(inputImage)) ?? [];
  }

  Future<List<DetectedObject>> _detectObject(String asset) async {
    if (_objectDetector == null) {
      // initialize object detector
      final modelPath = await _getAssetPath(_modelPath);
      final options = LocalObjectDetectorOptions(
        modelPath: modelPath,
        mode: DetectionMode.single,
        classifyObjects: true,
        multipleObjects: true,
      );
      _objectDetector = ObjectDetector(options: options);
    }
    // get and process image
    final inputImage = InputImage.fromFilePath(await _getAssetPath(asset));
    return (await _objectDetector?.processImage(inputImage)) ?? [];
  }

  Future<String> _getAssetPath(String asset) async {
    final path = '${(await getApplicationSupportDirectory()).path}/$asset';
    await Directory(dirname(path)).create(recursive: true);
    final file = File(path);
    if (!await file.exists()) {
      final byteData = await rootBundle.load(asset);
      await file.writeAsBytes(byteData.buffer
          .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    }
    return file.path;
  }
}
