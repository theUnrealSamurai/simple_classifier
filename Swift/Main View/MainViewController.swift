/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The view controller that selects an image and makes a prediction using Vision and Core ML.
*/

import UIKit
import Vision

class MainViewController: UIViewController {
    var firstRun = true

    /// A predictor instance that uses Vision and Core ML to generate prediction strings from a photo.
    let imagePredictor = ImagePredictor()

    /// The largest number of predictions the main view controller displays the user.
    let predictionsToShow = 2

    // MARK: Main storyboard outlets
    @IBOutlet weak var startupPrompts: UIStackView!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var predictionLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        let imagePath = Bundle.main.path(forResource: "dog", ofType: "png")
        let iconImage = UIImage(contentsOfFile:imagePath!)
        let targetSize = CGSize(width: 224, height: 224)
        let resizedImage = self.imagePredictor.resizeImage(image: iconImage!, targetSize: targetSize)
        
        self.imagePredictor.classifyImageTest(iconImage: resizedImage)
    }
}
