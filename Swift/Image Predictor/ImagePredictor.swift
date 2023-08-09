/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Makes predictions from images using the MobileNet model.
*/

import Vision
import UIKit

/// A convenience class that makes image classification predictions.
///
/// The Image Predictor creates and reuses an instance of a Core ML image classifier inside a ``VNCoreMLRequest``.
/// Each time it makes a prediction, the class:
/// - Creates a `VNImageRequestHandler` with an image
/// - Starts an image classification request for that image
/// - Converts the prediction results in a completion handler
/// - Updates the delegate's `predictions` property
/// - Tag: ImagePredictor
class ImagePredictor {
    /// - Tag: name
    static func createImageClassifier() -> VNCoreMLModel {
        // Use a default model configuration.
        let defaultConfig = MLModelConfiguration()

        // Create an instance of the image classifier's wrapper class.
        let imageClassifierWrapper = try? Test2(configuration: defaultConfig)

        guard let imageClassifier = imageClassifierWrapper else {
            fatalError("App failed to create an image classifier model instance.")
        }

        // Get the underlying model instance.
        let imageClassifierModel = imageClassifier.model

        // Create a Vision instance using the image classifier's model instance.
        guard let imageClassifierVisionModel = try? VNCoreMLModel(for: imageClassifierModel) else {
            fatalError("App failed to create a `VNCoreMLModel` instance.")
        }

        return imageClassifierVisionModel
    }

    /// A common image classifier instance that all Image Predictor instances use to generate predictions.
    ///
    /// Share one ``VNCoreMLModel`` instance --- for each Core ML model file --- across the app,
    /// since each can be expensive in time and resources.
    var imageClassifier = createImageClassifier()
    
    func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
        let size = image.size
        let widthRatio = targetSize.width / size.width
        let heightRatio = targetSize.height / size.height
        let scaleFactor = max(widthRatio, heightRatio)
        let scaledSize = CGSize(width: size.width * scaleFactor, height: size.height * scaleFactor)
        let origin = CGPoint(x: (targetSize.width - scaledSize.width) / 2.0, y: (targetSize.height - scaledSize.height) / 2.0)
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 0.0)
        image.draw(in: CGRect(origin: origin, size: scaledSize))
        let scaledImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return scaledImage!
    }
    
    func classifyImageTest(iconImage: UIImage) {
        /**
        let buffer = pixelBuffer(from: iconImage!)
        print("buffer", buffer)
        let mlarray = try! MLMultiArray(shape: [1, 224, 224, 3], dataType: MLMultiArrayDataType.float32 )
        for i in 0 ..< 224 * 224 {
            mlarray[i] = buffer[i] as NSNumber
        }
        print("reached here", mlarray)
        */
        
        let orientation = CGImagePropertyOrientation(iconImage.imageOrientation)
        guard let photoImage = iconImage.cgImage else {
            fatalError("Photo doesn't have underlying CGImage.")
        }

        let handler = VNImageRequestHandler(cgImage: photoImage, orientation: orientation)
        print("handler", handler)
        
        let request = VNCoreMLRequest(model:imageClassifier)
        print("request", request)
        
        try! handler.perform([request])
        print("res", request.results)
        
        /**
        guard let results = request.results as? [VNClassificationObservation] else {fatalError("Photo doesn't have underlying CGImage.")}

        results.forEach({ (result) in
            print("\(result.identifier) \(result.confidence * 100)")
        })
         */
    }
}
