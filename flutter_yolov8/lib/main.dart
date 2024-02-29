import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_speed_dial/flutter_speed_dial.dart';
import 'dart:async';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_vision/flutter_vision.dart';

enum Options { none, imagev5 }

late List<CameraDescription> cameras;
main() async {
  WidgetsFlutterBinding.ensureInitialized();
  DartPluginRegistrant.ensureInitialized();
  runApp(
    const MaterialApp(
      home: MyApp(),
    ),
  );
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late FlutterVision vision;
  Options option = Options.none;
  @override
  void initState() {
    super.initState();
    vision = FlutterVision();
  }

  @override
  void dispose() async {
    super.dispose();
    await vision.closeTesseractModel();
    await vision.closeYoloModel();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: task(option),
      floatingActionButton: SpeedDial(
        //margin bottom
        icon: Icons.menu, //icon on Floating action button
        activeIcon: Icons.close, //icon when menu is expanded on button
        backgroundColor: Colors.black12, //background color of button
        foregroundColor: Colors.white, //font color, icon color in button
        activeBackgroundColor:
            Colors.deepPurpleAccent, //background color when menu is expanded
        activeForegroundColor: Colors.white,
        visible: true,
        closeManually: false,
        curve: Curves.bounceIn,
        overlayColor: Colors.black,
        overlayOpacity: 0.5,
        buttonSize: const Size(56.0, 56.0),
        children: [
          SpeedDialChild(
            child: const Icon(Icons.camera),
            backgroundColor: Colors.blue,
            foregroundColor: Colors.white,
            label: 'YoloV5 on Image',
            labelStyle: const TextStyle(fontSize: 18.0),
            onTap: () {
              setState(() {
                option = Options.imagev5;
              });
            },
          ),
        ],
      ),
    );
  }

  Widget task(Options option) {
    if (option == Options.imagev5) {
      return YoloImageV5(vision: vision);
    }
    return const Center(child: Text("Choose Task"));
  }
}

class YoloImageV5 extends StatefulWidget {
  final FlutterVision vision;
  const YoloImageV5({Key? key, required this.vision}) : super(key: key);

  @override
  State<YoloImageV5> createState() => _YoloImageV5State();
}

class _YoloImageV5State extends State<YoloImageV5> {
  late CameraController controller;
  late List<Map<String, dynamic>> yoloResults;
  File? imageFile;
  int imageHeight = 1;
  int imageWidth = 1;
  bool isLoaded = false;
  CameraImage? cameraImage;
  bool isDetecting = false;

  @override
  void initState() {
    super.initState();
    init();
    loadYoloModel().then((value) {
      setState(() {
        yoloResults = [];
        isLoaded = true;
      });
    });
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  void init() async {
    cameras = await availableCameras();
    controller = CameraController(cameras[0], ResolutionPreset.medium);
    await controller.initialize();
  }

  @override
  Widget build(BuildContext context) {
    final Size size = MediaQuery.of(context).size;
    if (!isLoaded) {
      return const Scaffold(
        body: Center(
          child: Text("Model not loaded, waiting for it"),
        ),
      );
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        imageFile != null ? Image.file(imageFile!) : const SizedBox(),
        Align(
          alignment: Alignment.bottomCenter,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              TextButton(
                onPressed: capturePhoto,
                child: const Text("Pick image"),
              ),
              ElevatedButton(
                onPressed: yoloOnImage,
                child: const Text("Detect"),
              )
            ],
          ),
        ),
        ...displayBoxesAroundRecognizedObjects(size),
      ],
    );
  }

  Future<void> loadYoloModel() async {
    await widget.vision.loadYoloModel(
        labels: 'assets/yolov5/labels.txt',
        modelPath: 'assets/yolov5/yolov5n_display.tflite',
        modelVersion: "yolov5",
        quantization: false,
        numThreads: 2,
        useGpu: true);
    setState(() {
      isLoaded = true;
    });
  }

  Future<void> yoloOnFrame(CameraImage cameraImage) async {
    final result = await widget.vision.yoloOnFrame(
        bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        iouThreshold: 0.4,
        confThreshold: 0.4,
        classThreshold: 0.5);

    if (result.isNotEmpty) {
      setState(() {
        yoloResults = result;
        print(result);
      });
    } else {
      print("model called result empty");
    }
  }

  Future<void> capturePhoto() async {
    try {
      // Ensure that the camera controller is initialized
      await controller.initialize();

      // Take a picture and get the file
      XFile photo = await controller.takePicture();

      setState(() {
        imageFile = File(photo.path);
      });
      yoloOnImage();
      capturePhoto();
    } catch (e) {
      print("Error capturing photo: $e");
    }
  }

  // void capturePhotoOnLoop() {
  //   // Create a periodic timer with a duration of 3 seconds
  //   Timer.periodic(const Duration(seconds: 3), (timer) async {
  //     // Call the capturePhoto function
  //     await capturePhoto();
  //   });
  // }

  yoloOnImage() async {
    yoloResults.clear();
    DateTime startTime = DateTime.now(); // Record start time
    Uint8List byte = await imageFile!.readAsBytes();
    final image = await decodeImageFromList(byte);
    imageHeight = image.height;
    imageWidth = image.width;
    final result = await widget.vision.yoloOnImage(
      bytesList: byte,
      imageHeight: image.height,
      imageWidth: image.width,
      iouThreshold: 0.8,
      confThreshold: 0.4,
      classThreshold: 0.5,
    );
    DateTime endTime = DateTime.now(); // Record end time
    Duration elapsedTime =
        endTime.difference(startTime); // Calculate elapsed time
    print(
        'Response time: ${elapsedTime.inMilliseconds} ms'); // Print elapsed time
    if (result.isNotEmpty) {
      print(result);
      writeToLogFile(result.toString());
      setState(() {
        yoloResults = result;
      });
    }
  }

  late File logFile;

// Initialize the log file in initState or wherever appropriate
  Future<bool> initLogFile() async {
    try {
      final directory = await getDownloadsDirectory();
      if (directory != null) {
        logFile = File('${directory.path}/app_log.txt');
        print("\n\n\n\n$logFile\n\n\n");
        if (!await logFile.exists()) {
          await logFile.create();
        }
        return true; // Return true if initialization is successful
      } else {
        print('Error: Downloads directory is null');
        return false; // Return false if initialization fails
      }
    } catch (e) {
      print('Error initializing log file: $e');
      return false; // Return false if initialization fails
    }
  }

// Function to write logs to the log file
  void writeToLogFile(String log) async {
    await initLogFile(); // Ensure log file is initialized before writing
    await logFile.writeAsString('$log\n', mode: FileMode.append);
  }

  List<Widget> displayBoxesAroundRecognizedObjects(Size screen) {
    if (yoloResults.isEmpty) return [];

    for (var result in yoloResults) {
      if (result['tag'] == 'moving_display') {
        return [];
      }
    }

    double factorX = screen.width / (imageWidth);
    double imgRatio = imageWidth / imageHeight;
    double newWidth = imageWidth * factorX;
    double newHeight = newWidth / imgRatio;
    double factorY = newHeight / (imageHeight);

    double pady = (screen.height - newHeight) / 2;

    Color colorPick = const Color.fromARGB(255, 50, 233, 30);
    return yoloResults.map((result) {
      return Positioned(
        left: result["box"][0] * factorX,
        top: result["box"][1] * factorY + pady,
        width: (result["box"][2] - result["box"][0]) * factorX,
        height: (result["box"][3] - result["box"][1]) * factorY,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: const BorderRadius.all(Radius.circular(10.0)),
            border: Border.all(color: Colors.pink, width: 2.0),
          ),
          child: Text(
            "${result['tag']} ${(result['box'][4] * 100).toStringAsFixed(0)}%",
            style: TextStyle(
              background: Paint()..color = colorPick,
              color: Colors.white,
              fontSize: 18.0,
            ),
          ),
        ),
      );
    }).toList();
  }
}

class PolygonPainter extends CustomPainter {
  final List<Map<String, double>> points;

  PolygonPainter({required this.points});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color.fromARGB(129, 255, 2, 124)
      ..strokeWidth = 2
      ..style = PaintingStyle.fill;

    final path = Path();
    if (points.isNotEmpty) {
      path.moveTo(points[0]['x']!, points[0]['y']!);
      for (var i = 1; i < points.length; i++) {
        path.lineTo(points[i]['x']!, points[i]['y']!);
      }
      path.close();
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return false;
  }
}
