import AVFoundation
import UIKit
import CoreImage
import Combine

/// High-performance camera session manager for LensCoach 2.0
/// Feeds camera sample buffers to the PerceptionEngine (CoreML Neural Engine)
/// and triggers the GemmaCoachEngine on-device advice loop.
public class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate, AVCapturePhotoCaptureDelegate {
    @Published public var currentFrame: CVPixelBuffer?
    @Published public var capturedImage: UIImage?
    
    // Core 2.0 AI Engines
    @Published public var perceptionEngine = PerceptionEngine.shared
    @Published public var gemmaCoach = GemmaCoachEngine.shared
    
    // Grid & HUD State
    @Published public var isGridEnabled: Bool = true
    @Published public var isHorizonLevelGuideEnabled: Bool = true
    
    public let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let photoOutput = AVCapturePhotoOutput()
    private let sessionQueue = DispatchQueue(label: "com.lenscoach.sessionQueue")
    private let videoOutputQueue = DispatchQueue(label: "com.lenscoach.videoOutputQueue", qos: .userInteractive)
    
    private var frameCounter: Int = 0
    private var lastCoachEvaluationTime: Date = .distantPast
    
    public override init() {
        super.init()
        setupSession()
    }
    
    public func setupSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.session.beginConfiguration()
            
            // 1. Setup Wide Angle Input
            if let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
               let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice),
               self.session.canAddInput(videoDeviceInput) {
                self.session.addInput(videoDeviceInput)
            } else {
                print("CameraManager: Could not configure back camera input")
            }
            
            // 2. Setup Video Frame Output (30fps BGRA for CoreML/Metal pipeline)
            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
                self.videoOutput.alwaysDiscardsLateVideoFrames = true
                self.videoOutput.setSampleBufferDelegate(self, queue: self.videoOutputQueue)
                self.videoOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
                ]
            }
            
            // 3. Setup High-Res Photo Output
            if self.session.canAddOutput(self.photoOutput) {
                self.session.addOutput(self.photoOutput)
                self.photoOutput.isHighResolutionCaptureEnabled = true
            }
            
            self.session.sessionPreset = .photo
            self.session.commitConfiguration()
        }
    }
    
    public func start() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }
    
    public func stop() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }
    
    public func capturePhoto() {
        let settings = AVCapturePhotoSettings()
        settings.isHighResolutionPhotoEnabled = true
        photoOutput.capturePhoto(with: settings, delegate: self)
        
        // Haptic feedback on capture
        DispatchQueue.main.async {
            let impact = UIImpactFeedbackGenerator(style: .medium)
            impact.impactOccurred()
        }
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        // Update viewfinder frame buffer on main thread
        DispatchQueue.main.async {
            self.currentFrame = pixelBuffer
            DiagnosticsManager.shared.reportFrame()
        }
        
        // Process Neural Engine FrameScore on every 2nd frame (15-20 FPS)
        frameCounter += 1
        guard frameCounter % 2 == 0 else { return }
        
        perceptionEngine.processFrame(pixelBuffer: pixelBuffer)
        
        // Trigger Gemma Coach evaluation check
        let now = Date()
        if now.timeIntervalSince(lastCoachEvaluationTime) >= 1.5 {
            lastCoachEvaluationTime = now
            gemmaCoach.evaluateViewfinder(telemetry: perceptionEngine.currentTelemetry)
        }
    }
    
    // MARK: - AVCapturePhotoCaptureDelegate
    
    public func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("CameraManager photo error: \(error)")
            return
        }
        
        guard let data = photo.fileDataRepresentation(),
              let image = UIImage(data: data) else { return }
        
        // Save to Gallery with real-time aesthetic telemetry
        GalleryManager.shared.savePhoto(
            image,
            aestheticScore: perceptionEngine.currentTelemetry.aestheticScore,
            attributes: perceptionEngine.currentTelemetry.attributes
        )
        
        DispatchQueue.main.async {
            self.capturedImage = image
        }
        
        // Trigger on-device deep critique
        Task {
            _ = await self.gemmaCoach.analyzeCapturedPhoto(
                image: image,
                telemetry: self.perceptionEngine.currentTelemetry
            )
        }
    }
}
