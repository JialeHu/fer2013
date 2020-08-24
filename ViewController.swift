//
//  ViewController.swift
//  FER
//
//  Created by Jiale Hu on 7/13/20.
//  Copyright © 2020 Jiale Hu. All rights reserved.
//

import UIKit
import AVKit
import Vision
import CoreML

class ViewController: UIViewController {
    
    // MARK: Instance variables
    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var expresionLabel: UILabel!
    
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var rootLayer: CALayer?
    private var detectionOverlayLayer: CALayer?
    private var faceRectangleLayer: CAShapeLayer?
    private var textLayer: CATextLayer?
    
    private var captureSession: AVCaptureSession?
    private var captureDevice: AVCaptureDevice?
    private var resolution: CGSize?
    
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private var videoDataOutputQueue: DispatchQueue?
    
    private var trackingRequests: [VNTrackObjectRequest]?
    private var detectionRequests: [VNDetectFaceRectanglesRequest]?
    
    private var sequenceRequestHandler = VNSequenceRequestHandler()
    
    private let model = try! VNCoreMLModel(for: FER2013().model)
    private let modelInputSize = CGSize(width: 48, height: 48)
    
    // MARK: UIViewController overrides
    override func viewDidLoad() {
        super.viewDidLoad()
        
        do {
            try self.setupCaptureSession()
        } catch { print(error) }
        
        // Preview Layer
        let videoPreviewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession!)
        self.previewLayer = videoPreviewLayer
        videoPreviewLayer.name = "CameraPreview"
        videoPreviewLayer.backgroundColor = UIColor.black.cgColor
        videoPreviewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        if let previewRootLayer = self.previewView?.layer {
            self.rootLayer = previewRootLayer
            previewRootLayer.masksToBounds = true
            videoPreviewLayer.frame = previewRootLayer.bounds
            previewRootLayer.addSublayer(videoPreviewLayer)
        }
        
        // Drawing layers
        self.setupDrawingLayer()
        
        self.prepareVisionRequest()
        
        self.captureSession?.startRunning()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print("MemoryWarning")
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .portrait
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        return .portrait
    }
    
    // MARK: Setup Capture Session
    private func setupCaptureSession() throws {
        let captureSession = AVCaptureSession()
        
        // Input device
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front)
        guard let device = deviceDiscoverySession.devices.first else { throw NSError() }
        if let deviceInput = try? AVCaptureDeviceInput(device: device) {
            if captureSession.canAddInput(deviceInput) {
                captureSession.addInput(deviceInput)
            }
        }
        
        // Resolution
        var highestResolutionFormat: AVCaptureDevice.Format? = nil
        var highestResolutionDimensions = CMVideoDimensions(width: 0, height: 0)
        
        for format in device.formats {
            let deviceFormat = format as AVCaptureDevice.Format
            let deviceFormatDescription = deviceFormat.formatDescription
            if CMFormatDescriptionGetMediaSubType(deviceFormatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
                let candidateDimensions = CMVideoFormatDescriptionGetDimensions(deviceFormatDescription)
                if (highestResolutionFormat == nil) || (candidateDimensions.width > highestResolutionDimensions.width) {
                    highestResolutionFormat = deviceFormat
                    highestResolutionDimensions = candidateDimensions
                }
            }
        }
        let resolution = CGSize(width: CGFloat(highestResolutionDimensions.width), height: CGFloat(highestResolutionDimensions.height))
        
        // Output queue
        let videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        let videoDataOutputQueue = DispatchQueue(label: "FER")
        videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        videoDataOutput.connection(with: .video)?.isEnabled = true
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
        
        if let captureConnection = videoDataOutput.connection(with: AVMediaType.video) {
            if captureConnection.isCameraIntrinsicMatrixDeliverySupported {
                captureConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }
        
        // Instance variables
        self.captureSession = captureSession
        self.captureDevice = device
        self.resolution = resolution
            
        self.videoDataOutput = videoDataOutput
        self.videoDataOutputQueue = videoDataOutputQueue
    }
    
    // MARK: Setup Vision Request
    private func prepareVisionRequest() {
        
        var requests = [VNTrackObjectRequest]()
        
        // Face detection request handler
        let faceDetectionRequest = VNDetectFaceRectanglesRequest { (request, error) in
            if error != nil {
                print("FaceDetection error: \(String(describing: error)).")
            }
            // Get request and result
            guard let faceDetectionRequest = request as? VNDetectFaceRectanglesRequest,
                let results = faceDetectionRequest.results as? [VNFaceObservation] else {
                    return
            }
            DispatchQueue.main.async {
                // Add the observations to the tracking list
                for observation in results {
                    let faceTrackingRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
                    requests.append(faceTrackingRequest)
                }
                self.trackingRequests = requests
            }
        }
        self.detectionRequests = [faceDetectionRequest]
        self.sequenceRequestHandler = VNSequenceRequestHandler()
    }
    
    // MARK: Setup Drawing Layer
    private func setupDrawingLayer() {
        let captureDeviceResolution = self.resolution!
        let captureDeviceBounds = CGRect(x: 0,
                                         y: 0,
                                         width: captureDeviceResolution.width,
                                         height: captureDeviceResolution.height)
        let captureDeviceBoundsCenterPoint = CGPoint(x: captureDeviceBounds.midX,
                                                     y: captureDeviceBounds.midY)
        let normalizedCenterPoint = CGPoint(x: 0.5, y: 0.5)
        
        let faceTextLayer = CATextLayer()
        faceTextLayer.name = "FacialExpressionTextLayer"
        faceTextLayer.bounds = captureDeviceBounds
        faceTextLayer.anchorPoint = normalizedCenterPoint
        faceTextLayer.position = captureDeviceBoundsCenterPoint
        faceTextLayer.foregroundColor = UIColor.black.cgColor
        faceTextLayer.shadowOpacity = 0.7
        
        let faceRectangleShapeLayer = CAShapeLayer()
        faceRectangleShapeLayer.name = "RectangleOutlineLayer"
        faceRectangleShapeLayer.bounds = captureDeviceBounds
        faceRectangleShapeLayer.anchorPoint = normalizedCenterPoint
        faceRectangleShapeLayer.position = captureDeviceBoundsCenterPoint
        faceRectangleShapeLayer.fillColor = nil
        faceRectangleShapeLayer.strokeColor = UIColor.yellow.withAlphaComponent(0.7).cgColor
        faceRectangleShapeLayer.lineWidth = 5
        faceRectangleShapeLayer.shadowOpacity = 0.7
        faceRectangleShapeLayer.shadowRadius = 10
        faceRectangleShapeLayer.addSublayer(faceTextLayer)
        
        let rootLayer = self.rootLayer!
        let overlayLayer = CALayer()
        overlayLayer.name = "DetectionOverlay"
        overlayLayer.masksToBounds = true
        overlayLayer.anchorPoint = normalizedCenterPoint
        overlayLayer.bounds = captureDeviceBounds
        overlayLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        overlayLayer.addSublayer(faceRectangleShapeLayer)
        rootLayer.addSublayer(overlayLayer)
        
        self.detectionOverlayLayer = overlayLayer
        self.faceRectangleLayer = faceRectangleShapeLayer
        self.textLayer = faceTextLayer
    }
    
    // MARK: Update Drawing Layer
    fileprivate func updateDrawing(face: VNFaceObservation) {
        CATransaction.begin()
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let faceRectangleLayer = self.faceRectangleLayer!
        let previewLayer = self.previewLayer!
        let overlayLayer = self.detectionOverlayLayer!
        let rootLayer = self.rootLayer!
        
        let deviceResolution = self.resolution!
        let faceBounds = VNImageRectForNormalizedRect(face.boundingBox, Int(deviceResolution.width), Int(deviceResolution.height))
        let faceRectanglePath = CGMutablePath()
        faceRectanglePath.addRect(faceBounds)
        faceRectangleLayer.path = faceRectanglePath
        
        let videoPreviewRect = previewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        // Rotate the layer into screen orientation.
        var rotation: CGFloat
        var scaleX: CGFloat
        var scaleY: CGFloat
        switch UIDevice.current.orientation {
        case .portraitUpsideDown:
            rotation = 180
            scaleX = videoPreviewRect.width / deviceResolution.width
            scaleY = videoPreviewRect.height / deviceResolution.height
        case .landscapeLeft:
            rotation = 90
            scaleX = videoPreviewRect.height / deviceResolution.width
            scaleY = scaleX
        case .landscapeRight:
            rotation = -90
            scaleX = videoPreviewRect.height / deviceResolution.width
            scaleY = scaleX
        default:
            rotation = 0
            scaleX = videoPreviewRect.width / deviceResolution.width
            scaleY = videoPreviewRect.height / deviceResolution.height
        }
        
        // Scale and mirror the image to ensure upright presentation.
        let affineTransform = CGAffineTransform(rotationAngle: self.radiansForDegrees(rotation))
            .scaledBy(x: scaleX, y: -scaleY)
        overlayLayer.setAffineTransform(affineTransform)
        
        let rootLayerBounds = rootLayer.bounds
        overlayLayer.position = CGPoint(x: rootLayerBounds.midX, y: rootLayerBounds.midY)
        
        CATransaction.commit()
    }
    
    fileprivate func clearDrawing() {
        CATransaction.begin()
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let faceRectangleLayer = self.faceRectangleLayer!
        
        faceRectangleLayer.path = nil
        
        CATransaction.commit()
    }
    
    // MARK: Device Orientation
    private func exifOrientationForDeviceOrientation(_ deviceOrientation: UIDeviceOrientation) -> CGImagePropertyOrientation {
        switch deviceOrientation {
        case .portraitUpsideDown:
            return .rightMirrored
        case .landscapeLeft:
            return .downMirrored
        case .landscapeRight:
            return .upMirrored
        default:
            return .leftMirrored
        }
    }
    
    // MARK: Conversions
    private func bufferToImage(pixelBuffer: CVImageBuffer, exifOrientation: CGImagePropertyOrientation, face: VNFaceObservation) -> CIImage {
        let image = CIImage.init(cvPixelBuffer: pixelBuffer).oriented(exifOrientation)
                        
        let width = image.extent.width * face.boundingBox.width
        let height = image.extent.height * face.boundingBox.height
        let x = image.extent.width * face.boundingBox.origin.x
        let y = image.extent.height * face.boundingBox.origin.y
        
        let faceCropRect = CGRect(x: x, y: y, width: width, height: height)
//                let percentage: CGFloat = 0.6
//                let rect = CGRect(x: x, y: y, width: width, height: height)
//                let increasedRect = rect.insetBy(dx: width * -percentage, dy: height * -percentage)
        let cropped = image.cropped(to: faceCropRect)
        
//        let grayFilter = CIFilter(name: "CIPhotoEffectMono")
//        grayFilter!.setValue(cropped, forKey: kCIInputImageKey)
//        let grayscaled = grayFilter!.outputImage!
//
//        let targetSize = self.modelInputSize
//        let currentSize = grayscaled.extent.size
//        let scaleX = targetSize.width / currentSize.width
//        let scaleY = targetSize.height / currentSize.height
//        let scaleTransform = CGAffineTransform(scaleX: scaleX, y: scaleY)
//        let scaled = grayscaled.transformed(by: scaleTransform, highQualityDownsample: true)
        
        return cropped
    }
    
    private func radiansForDegrees(_ degrees: CGFloat) -> CGFloat {
        return CGFloat(Double(degrees) * Double.pi / 180.0)
    }
    
}

// MARK: Output Delegate
extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Get request handler options
        var requestHandlerOptions: [VNImageOption: AnyObject] = [:]
        let cameraIntrinsicData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil)
        if cameraIntrinsicData != nil {
            requestHandlerOptions[VNImageOption.cameraIntrinsics] = cameraIntrinsicData
        }
        
        // Get pixal buffer
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to obtain a CVPixelBuffer for the current output frame.")
            return
        }
        
        // Get tracking requests
        let exifOrientation = exifOrientationForDeviceOrientation(UIDevice.current.orientation)
        guard let requests = self.trackingRequests, !requests.isEmpty else { // No Face detected
            // Clear Drawing
            DispatchQueue.main.async {
                self.clearDrawing()
            }
            // Perform initial detection
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                            orientation: exifOrientation,
                                                            options: requestHandlerOptions)
            do {
                guard let detectRequests = self.detectionRequests else { return }
                try imageRequestHandler.perform(detectRequests)
            } catch let error as NSError {
                print(error)
            }
            return
        }
        
        // Sequence request handler
        do {
            try self.sequenceRequestHandler.perform(requests,
                                                     on: pixelBuffer,
                                                     orientation: exifOrientation)
        } catch let error as NSError {
            print(error)
        }
        
        // Setup the next round of tracking.
        var newTrackingRequests = [VNTrackObjectRequest]()
        for trackingRequest in requests {
            guard let results = trackingRequest.results else {
                return
            }
            guard let observation = results[0] as? VNDetectedObjectObservation else {
                return
            }
            if !trackingRequest.isLastFrame {
                if observation.confidence > 0.3 {
                    trackingRequest.inputObservation = observation
                } else {
                    trackingRequest.isLastFrame = true
                }
                newTrackingRequests.append(trackingRequest)
            }
        }
        self.trackingRequests = newTrackingRequests
        if newTrackingRequests.isEmpty {
            // Nothing to track, so abort.
            return
        }
        
        // Perform face landmark tracking on detected faces.
        var faceLandmarkRequests = [VNDetectFaceLandmarksRequest]()
        
        // MARK: After detection
        // Perform landmark detection on tracked faces.
        for trackingRequest in newTrackingRequests {
            
            // Face landmark request completion handler
            let faceLandmarksRequest = VNDetectFaceLandmarksRequest { (request, error) in
                if error != nil {
                    print("FaceLandmarks error: \(String(describing: error)).")
                }
                
                guard let landmarksRequest = request as? VNDetectFaceLandmarksRequest,
                    let results = landmarksRequest.results as? [VNFaceObservation] else {
                        print("Failed to get landmark results")
                        return
                }
                
                // Get Face image
                DispatchQueue.main.async {
                    
                    let face = results[0]
                    self.updateDrawing(face: face)
                    
                }
                
                // MARK: CoreML
                let face = results[0]
                
                let mlRequest = VNCoreMLRequest(model: self.model) { (finishedReq, err) in
                    
                    if err != nil {
                        print("FaceML error: \(String(describing: err)).")
                    }
                    guard let results = finishedReq.results as? [VNClassificationObservation] else { return }
                    print(results.first!.identifier.description + " " +  results.first!.confidence.description)
//                    guard let results = finishedReq.results as? [MLMultiArray] else { return }
//                    print(results)
                    
                    // Update Text Layer
                    DispatchQueue.main.async {
                        
                        self.expresionLabel.text = results.first!.identifier.description + " " +  results.first!.confidence.description
                        
                    }
                }
                
                // Request coreML with processed image
                let faceImage = self.bufferToImage(pixelBuffer: pixelBuffer, exifOrientation: exifOrientation, face: face)
                print(faceImage)
                // Image Preview
//                DispatchQueue.main.async {
//                    let ciContext = CIContext()
//                    if let cgImage = ciContext.createCGImage(faceImage, from: faceImage.extent) {
//                        self.imageView.image = UIImage(cgImage: cgImage)
//                    }
//                }
//                try? VNImageRequestHandler(ciImage: faceImage, options: [:]).perform([mlRequest])
                
                
                var facePixelBuffer: CVPixelBuffer? = nil
                CVPixelBufferCreate(kCFAllocatorDefault, Int(self.modelInputSize.width), Int(self.modelInputSize.height), kCVPixelFormatType_OneComponent8, nil, &facePixelBuffer)
                CVPixelBufferLockBaseAddress(facePixelBuffer!, CVPixelBufferLockFlags(rawValue:0))
                
                let colorspace = CGColorSpaceCreateDeviceGray()
                let bitmapContext = CGContext(data: CVPixelBufferGetBaseAddress(facePixelBuffer!), width: Int(self.modelInputSize.width), height: Int(self.modelInputSize.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(facePixelBuffer!), space: colorspace, bitmapInfo: 0)!
                bitmapContext.draw(CIContext().createCGImage(faceImage, from: faceImage.extent)!, in: CGRect(x: 0, y: 0, width: self.modelInputSize.width, height: self.modelInputSize.height))
                
//                let context = CIContext()
//                context.render(faceImage, to: facePixelBuffer!)
                try? VNImageRequestHandler(cvPixelBuffer: facePixelBuffer!, options: [:]).perform([mlRequest])
                
                
                
            }
            
            // Next tracking observation
            guard let trackingResults = trackingRequest.results else {
                return
            }
            guard let observation = trackingResults[0] as? VNDetectedObjectObservation else {
                return
            }
            let faceObservation = VNFaceObservation(boundingBox: observation.boundingBox)
            faceLandmarksRequest.inputFaceObservations = [faceObservation]
            
            // Continue to track detected facial landmarks.
            faceLandmarkRequests.append(faceLandmarksRequest)
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                            orientation: exifOrientation,
                                                            options: requestHandlerOptions)
            do {
                try imageRequestHandler.perform(faceLandmarkRequests)
            } catch let error as NSError {
                print(error)
            }
            
        }
        
    }
    
}
