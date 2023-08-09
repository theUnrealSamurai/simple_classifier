/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Adds the camera picker support to the main view controller.
*/

import UIKit

extension MainViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    /// Creates a controller that gives the user a view they can use to take a photo with the device's camera.
    var cameraPicker: UIImagePickerController {
        let cameraPicker = UIImagePickerController()
        cameraPicker.delegate = self
        cameraPicker.sourceType = .camera
        return cameraPicker
    }


}
