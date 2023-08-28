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
    
    static func createImageClassifier() -> VNCoreMLModel {
        // デフォルトのモデルのコンフィグを使用
        let defaultConfig = MLModelConfiguration()
        
        // 生成したモデルのクラスを呼び出し、モデルを呼び出すためのラッパーを取得
        let imageClassifierWrapper = try? Test(configuration: defaultConfig)
        
        guard let imageClassifier = imageClassifierWrapper else {
            fatalError("App failed to create an image classifier model instance.")
        }
        
        // モデルのインスタンスを取得
        let imageClassifierModel = imageClassifier.model
        
        // Visionフレームワークの画像分類インスタンスを取得
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
        
        let orientation = CGImagePropertyOrientation(iconImage.imageOrientation)
        guard let photoImage = iconImage.cgImage else {
            fatalError("Photo doesn't have underlying CGImage.")
        }
        
        let handler = VNImageRequestHandler(cgImage: photoImage, orientation: orientation)
        
        let request = VNCoreMLRequest(model:imageClassifier)
        
        // 以下で画像分類を行う
        try! handler.perform([request])
        
        // 分類結果を表示するコード
        print("res", request.results!)
    }
}
