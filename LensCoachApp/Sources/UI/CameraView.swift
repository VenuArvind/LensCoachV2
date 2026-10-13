import SwiftUI
import AVFoundation

public struct CameraPreview: UIViewRepresentable {
    @ObservedObject public var cameraManager: CameraManager
    
    public init(cameraManager: CameraManager) {
        self.cameraManager = cameraManager
    }
    
    public func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let previewLayer = AVCaptureVideoPreviewLayer(session: cameraManager.session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        context.coordinator.previewLayer = previewLayer
        return view
    }
    
    public func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.previewLayer?.frame = uiView.bounds
    }
    
    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    public class Coordinator: NSObject {
        var previewLayer: AVCaptureVideoPreviewLayer?
    }
}

public struct CameraView: View {
    @StateObject public var cameraManager = CameraManager()
    @State private var showCritiqueSheet: Bool = false
    @State private var showGallerySheet: Bool = false
    @State private var showDiagnostics: Bool = false
    
    public init() {}
    
    public var body: some View {
        ZStack {
            // 1. Camera Viewfinder
            CameraPreview(cameraManager: cameraManager)
                .edgesIgnoringSafeArea(.all)
                .onAppear {
                    cameraManager.start()
                }
                .onDisappear {
                    cameraManager.stop()
                }
            
            // 2. Rule of Thirds Overlay
            if cameraManager.isGridEnabled {
                GridOverlay()
                    .edgesIgnoringSafeArea(.all)
                    .allowsHitTesting(false)
            }
            
            // 3. Dynamic Horizon Level Guide (Aerospace style)
            if cameraManager.isHorizonLevelGuideEnabled {
                HorizonLevelReticle(
                    tiltDegrees: cameraManager.perceptionEngine.currentTelemetry.horizonTiltDegrees,
                    isLevel: cameraManager.perceptionEngine.currentTelemetry.isHorizonLevel
                )
                .allowsHitTesting(false)
            }
            
            // 4. Salient Subject Bounding Box
            if let subject = cameraManager.perceptionEngine.currentTelemetry.subjectRect {
                SubjectFocusBox(normalizedRect: subject)
                    .allowsHitTesting(false)
            }
            
            // 5. HUD Controls & Telemetry
            VStack(spacing: 0) {
                // Top Telemetry & Control Bar
                HStack(spacing: 12) {
                    // Aesthetic Score Pill (CoreML ANE)
                    AestheticScorePill(
                        score: cameraManager.perceptionEngine.currentTelemetry.aestheticScore
                    )
                    
                    Spacer()
                    
                    // Quick Action Buttons
                    HStack(spacing: 10) {
                        Button(action: {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                cameraManager.isGridEnabled.toggle()
                            }
                        }) {
                            Image(systemName: cameraManager.isGridEnabled ? "grid" : "grid.slash")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundColor(cameraManager.isGridEnabled ? .white : .white.opacity(0.4))
                                .frame(width: 36, height: 36)
                                .background(BlurView(style: .systemUltraThinMaterialDark).clipShape(Circle()))
                        }
                        
                        Button(action: {
                            withAnimation(.easeInOut(duration: 0.2)) {
                                cameraManager.isHorizonLevelGuideEnabled.toggle()
                            }
                        }) {
                            Image(systemName: "gyroscope")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundColor(cameraManager.isHorizonLevelGuideEnabled ? .emerald : .white.opacity(0.4))
                                .frame(width: 36, height: 36)
                                .background(BlurView(style: .systemUltraThinMaterialDark).clipShape(Circle()))
                        }
                        
                        Button(action: { showGallerySheet = true }) {
                            Image(systemName: "photo.stack")
                                .font(.system(size: 15, weight: .bold))
                                .foregroundColor(.white)
                                .frame(width: 36, height: 36)
                                .background(BlurView(style: .systemUltraThinMaterialDark).clipShape(Circle()))
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 50)
                
                Spacer()
                
                // Gemma 4 Live Coach Pill (Floating Viewfinder Advice)
                LiveCoachCapsule(tip: cameraManager.gemmaCoach.activeTip)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 20)
                
                // Bottom Controls Bar with Shutter Button
                HStack(alignment: .center) {
                    // Engine Status Pill
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color.emerald)
                            .frame(width: 6, height: 6)
                        Text("GEMMA 4 EDGE")
                            .font(.system(size: 10, weight: .black, design: .monospaced))
                            .foregroundColor(.white.opacity(0.8))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(BlurView(style: .systemUltraThinMaterialDark).clipShape(Capsule()))
                    .frame(width: 100, alignment: .leading)
                    
                    Spacer()
                    
                    // Modern Shutter Button
                    Button(action: {
                        cameraManager.capturePhoto()
                        showCritiqueSheet = true
                    }) {
                        ZStack {
                            Circle()
                                .stroke(Color.white.opacity(0.3), lineWidth: 4)
                                .frame(width: 78, height: 78)
                            
                            Circle()
                                .fill(Color.white)
                                .frame(width: 64, height: 64)
                        }
                    }
                    
                    Spacer()
                    
                    // Empty balancer
                    Color.clear.frame(width: 100, height: 1)
                }
                .padding(.horizontal, 24)
                .padding(.bottom, 40)
            }
        }
        .sheet(isPresented: $showCritiqueSheet) {
            if let image = cameraManager.capturedImage {
                CritiqueView(
                    coachEngine: cameraManager.gemmaCoach,
                    image: image,
                    onDismiss: { showCritiqueSheet = false }
                )
            }
        }
        .sheet(isPresented: $showGallerySheet) {
            GalleryView()
        }
    }
}

// MARK: - Subviews & Overlays

/// Aerospace-style Horizon Level Reticle
struct HorizonLevelReticle: View {
    let tiltDegrees: Double
    let isLevel: Bool
    
    var body: some View {
        HStack(spacing: 24) {
            // Left Horizon Hash
            Rectangle()
                .fill(isLevel ? Color.emerald : Color.white.opacity(0.5))
                .frame(width: 44, height: isLevel ? 3 : 1.5)
            
            // Center Reticle
            ZStack {
                Circle()
                    .stroke(isLevel ? Color.emerald : Color.white.opacity(0.4), lineWidth: 1.5)
                    .frame(width: 14, height: 14)
                
                Circle()
                    .fill(isLevel ? Color.emerald : Color.white)
                    .frame(width: 4, height: 4)
            }
            
            // Right Horizon Hash
            Rectangle()
                .fill(isLevel ? Color.emerald : Color.white.opacity(0.5))
                .frame(width: 44, height: isLevel ? 3 : 1.5)
        }
        .rotationEffect(.degrees(tiltDegrees))
        .animation(.interactiveSpring(response: 0.15, dampingFraction: 0.7), value: tiltDegrees)
        .overlay(
            // Degree Indicator
            VStack {
                Spacer().frame(height: 36)
                Text(String(format: "%+.1f°", tiltDegrees))
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(isLevel ? .emerald : .white.opacity(0.6))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(BlurView(style: .systemUltraThinMaterialDark).clipShape(Capsule()))
            }
        )
    }
}

/// Floating Live Coach Capsule
struct LiveCoachCapsule: View {
    let tip: LiveTip
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: tip.action.iconName)
                .font(.system(size: 18, weight: .bold))
                .foregroundColor(.emerald)
                .frame(width: 28)
            
            VStack(alignment: .leading, spacing: 2) {
                Text("COACH TIP")
                    .font(.system(size: 9, weight: .black, design: .monospaced))
                    .foregroundColor(.emerald)
                
                Text(tip.text)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.white)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            BlurView(style: .systemUltraThinMaterialDark)
                .cornerRadius(18)
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(Color.white.opacity(0.12), lineWidth: 1)
                )
        )
        .shadow(color: .black.opacity(0.4), radius: 10, x: 0, y: 4)
        .animation(.spring(), value: tip.text)
    }
}

/// Aesthetic Score Pill (CoreML Neural Engine)
struct AestheticScorePill: View {
    let score: Float
    
    var color: Color {
        if score >= 0.70 { return .emerald }
        if score >= 0.45 { return .yellow }
        return .orange
    }
    
    var body: some View {
        HStack(spacing: 8) {
            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.2), lineWidth: 2.5)
                    .frame(width: 22, height: 22)
                
                Circle()
                    .trim(from: 0, to: CGFloat(min(1.0, max(0.0, score))))
                    .stroke(color, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                    .frame(width: 22, height: 22)
                    .rotationEffect(.degrees(-90))
            }
            
            VStack(alignment: .leading, spacing: 0) {
                Text(String(format: "%.1f", score * 10))
                    .font(.system(size: 14, weight: .black, design: .rounded))
                    .foregroundColor(.white)
                Text("SCORE")
                    .font(.system(size: 7, weight: .black))
                    .foregroundColor(.white.opacity(0.5))
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(BlurView(style: .systemUltraThinMaterialDark).clipShape(Capsule()))
    }
}

/// Dynamic Salient Subject Box
struct SubjectFocusBox: View {
    let normalizedRect: CGRect
    
    var body: some View {
        GeometryReader { geo in
            let frame = CGRect(
                x: normalizedRect.origin.x * geo.size.width,
                y: (1.0 - normalizedRect.origin.y - normalizedRect.size.height) * geo.size.height,
                width: normalizedRect.size.width * geo.size.width,
                height: normalizedRect.size.height * geo.size.height
            )
            
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.white.opacity(0.4), style: StrokeStyle(lineWidth: 1.5, dash: [6, 4]))
                .frame(width: max(40, frame.width), height: max(40, frame.height))
                .position(x: frame.midX, y: frame.midY)
                .animation(.interactiveSpring(), value: normalizedRect)
        }
    }
}

/// Rule of Thirds Viewfinder Grid
struct GridOverlay: View {
    var body: some View {
        GeometryReader { geo in
            Path { path in
                let w = geo.size.width
                let h = geo.size.height
                
                // Vertical grid lines
                path.move(to: CGPoint(x: w / 3.0, y: 0))
                path.addLine(to: CGPoint(x: w / 3.0, y: h))
                path.move(to: CGPoint(x: 2 * w / 3.0, y: 0))
                path.addLine(to: CGPoint(x: 2 * w / 3.0, y: h))
                
                // Horizontal grid lines
                path.move(to: CGPoint(x: 0, y: h / 3.0))
                path.addLine(to: CGPoint(x: w, y: h / 3.0))
                path.move(to: CGPoint(x: 0, y: 2 * h / 3.0))
                path.addLine(to: CGPoint(x: w, y: 2 * h / 3.0))
            }
            .stroke(Color.white.opacity(0.25), lineWidth: 0.75)
        }
    }
}

/// Reusable Glassmorphism Background
public struct BlurView: UIViewRepresentable {
    public var style: UIBlurEffect.Style
    
    public init(style: UIBlurEffect.Style) {
        self.style = style
    }
    
    public func makeUIView(context: Context) -> UIVisualEffectView {
        UIVisualEffectView(effect: UIBlurEffect(style: style))
    }
    
    public func updateUIView(_ uiView: UIVisualEffectView, context: Context) {
        uiView.effect = UIBlurEffect(style: style)
    }
}
