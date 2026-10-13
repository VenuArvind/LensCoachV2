import Foundation
import CoreML
import Vision
import CoreMotion
import UIKit
import Combine

/// Telemetry metrics representing the live viewfinder scene.
public struct SceneTelemetry: Equatable {
    public var aestheticScore: Float = 0.0
    public var attributes: [String: Float] = [:]
    public var horizonTiltDegrees: Double = 0.0
    public var isHorizonLevel: Bool { abs(horizonTiltDegrees) <= 0.6 }
    public var subjectRect: CGRect? = nil
    public var isCameraSteady: Bool = true
    
    public init(
        aestheticScore: Float = 0.0,
        attributes: [String: Float] = [:],
        horizonTiltDegrees: Double = 0.0,
        subjectRect: CGRect? = nil,
        isCameraSteady: Bool = true
    ) {
        self.aestheticScore = aestheticScore
        self.attributes = attributes
        self.horizonTiltDegrees = horizonTiltDegrees
        self.subjectRect = subjectRect
        self.isCameraSteady = isCameraSteady
    }
}

/// Fast real-time perception engine (30 FPS on Apple Neural Engine)
/// Computes multi-attribute aesthetic scores, horizon roll tilt via CoreMotion,
/// and salient subject regions via Apple Vision.
public class PerceptionEngine: ObservableObject {
    public static let shared = PerceptionEngine()
    
    @Published public var currentTelemetry = SceneTelemetry()
    @Published public var smoothedScore: Float = 0.0
    @Published public var smoothedTilt: Double = 0.0
    
    // CoreML FrameScore model
    private var vnModel: VNCoreMLModel?
    private let alpha: Float = 0.35 // EMA filter factor for aesthetic score
    private var lastAttributes: [Float] = Array(repeating: 0.0, count: 12)
    
    // Motion Manager for true physical horizon leveling
    private let motionManager = CMMotionManager()
    
    // Saliency / Object request
    private lazy var saliencyRequest: VNGenerateAttentionBasedSaliencyImageRequest = {
        let req = VNGenerateAttentionBasedSaliencyImageRequest()
        return req
    }()
    
    public let attributeNames = [
        "Score", "Balance", "Color", "Content", "DoF",
        "Light", "Blur", "Object", "Repetition", "RuleOfThirds",
        "Symmetry", "Vivid"
    ]
    
    public init() {
        setupCoreMLModel()
        startMotionTracking()
    }
    
    deinit {
        motionManager.stopDeviceMotionUpdates()
    }
    
    private func setupCoreMLModel() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all // Prioritize Apple Neural Engine
            
            guard let modelURL = Bundle.main.url(forResource: "FrameScore", withExtension: "mlmodelc") ??
                                Bundle.module.url(forResource: "FrameScore", withExtension: "mlmodelc") else {
                print("PerceptionEngine: FrameScore.mlmodelc not found in bundle")
                return
            }
            
            let coreMLModel = try MLModel(contentsOf: modelURL, configuration: config)
            self.vnModel = try VNCoreMLModel(for: coreMLModel)
            print("PerceptionEngine: Neural Engine FrameScore loaded successfully")
        } catch {
            print("PerceptionEngine: Failed to load FrameScore CoreML model: \(error)")
        }
    }
    
    private func startMotionTracking() {
        guard motionManager.isDeviceMotionAvailable else { return }
        motionManager.deviceMotionUpdateInterval = 1.0 / 30.0
        motionManager.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self = self, let motion = motion else { return }
            
            // Calculate tilt angle in degrees relative to landscape/portrait
            let gravity = motion.gravity
            let tiltRadians = atan2(gravity.x, -gravity.y)
            var tiltDegrees = tiltRadians * (180.0 / .pi)
            
            // Normalize tilt relative to the closest 90-degree orientation
            if abs(tiltDegrees) > 45 && abs(tiltDegrees) < 135 {
                tiltDegrees = tiltDegrees > 0 ? tiltDegrees - 90 : tiltDegrees + 90
            } else if abs(tiltDegrees) >= 135 {
                tiltDegrees = tiltDegrees > 0 ? tiltDegrees - 180 : tiltDegrees + 180
            }
            
            // EMA smooth horizon tilt
            let smoothed = (0.25 * tiltDegrees) + (0.75 * self.smoothedTilt)
            self.smoothedTilt = smoothed
            
            // Check stability (low acceleration variance)
            let userAcc = motion.userAcceleration
            let isSteady = (abs(userAcc.x) + abs(userAcc.y) + abs(userAcc.z)) < 0.18
            
            self.currentTelemetry.horizonTiltDegrees = smoothed
            self.currentTelemetry.isCameraSteady = isSteady
        }
    }
    
    /// Process a live camera pixel buffer at 30 FPS
    public func processFrame(pixelBuffer: CVPixelBuffer) {
        guard let vnModel = vnModel else { return }
        
        let request = VNCoreMLRequest(model: vnModel) { [weak self] req, error in
            guard let self = self,
                  let results = req.results as? [VNCoreMLFeatureValueObservation],
                  let multiArray = results.first?.featureValue.multiArrayValue else { return }
            
            self.handleScores(multiArray)
        }
        request.imageCropAndScaleOption = .centerCrop
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        let startTime = CFAbsoluteTimeGetCurrent()
        do {
            try handler.perform([request, saliencyRequest])
            let latencyMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            DiagnosticsManager.shared.reportInferenceLatency(latencyMs)
            
            // Extract dominant salient region if available
            if let saliencyObs = saliencyRequest.results?.first as? VNSaliencyImageObservation,
               let salientObjects = saliencyObs.salientObjects,
               let primarySubject = salientObjects.first {
                DispatchQueue.main.async {
                    self.currentTelemetry.subjectRect = primarySubject.boundingBox
                }
            }
        } catch {
            print("PerceptionEngine inference error: \(error)")
        }
    }
    
    private func handleScores(_ multiArray: MLMultiArray) {
        var rawScores = [Float]()
        for i in 0..<min(12, multiArray.count) {
            rawScores.append(multiArray[i].floatValue)
        }
        while rawScores.count < 12 { rawScores.append(0.5) }
        
        // Exponential moving average filter
        var smoothed = [Float]()
        for i in 0..<12 {
            let val = (alpha * rawScores[i]) + ((1.0 - alpha) * lastAttributes[i])
            smoothed.append(val)
            lastAttributes[i] = val
        }
        
        DispatchQueue.main.async {
            self.smoothedScore = smoothed[0]
            var map: [String: Float] = [:]
            for i in 1..<12 {
                map[self.attributeNames[i]] = smoothed[i]
            }
            map["Score"] = smoothed[0]
            
            self.currentTelemetry.aestheticScore = smoothed[0]
            self.currentTelemetry.attributes = map
        }
    }
    
    public func reset() {
        lastAttributes = Array(repeating: 0.0, count: 12)
        smoothedScore = 0.0
        smoothedTilt = 0.0
    }
}
