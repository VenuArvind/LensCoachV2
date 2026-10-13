import SwiftUI

/// On-Device Gemma 4 E2B Shot Review & Photographic Analysis
public struct CritiqueView: View {
    @ObservedObject var coachEngine: GemmaCoachEngine
    let originalImage: UIImage
    var onDismiss: () -> Void
    
    @State private var showCropPreview: Bool = false
    @State private var showFilterEnhance: Bool = false
    @State private var enhancedImage: UIImage? = nil
    
    public init(
        coachEngine: GemmaCoachEngine = .shared,
        image: UIImage,
        onDismiss: @escaping () -> Void
    ) {
        self.coachEngine = coachEngine
        self.originalImage = image
        self.onDismiss = onDismiss
    }
    
    public var body: some View {
        ZStack {
            Color.black.edgesIgnoringSafeArea(.all)
            
            VStack(spacing: 0) {
                // Header Bar
                HStack {
                    HStack(spacing: 8) {
                        Image(systemName: "cpu.fill")
                            .foregroundColor(.emerald)
                        Text("GEMMA 4 ON-DEVICE REVIEW")
                            .font(.system(size: 13, weight: .black, design: .monospaced))
                            .foregroundColor(.white)
                    }
                    
                    Spacer()
                    
                    Button(action: onDismiss) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 26))
                            .foregroundColor(.white.opacity(0.6))
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 12)
                
                ScrollView(showsIndicators: false) {
                    VStack(spacing: 20) {
                        // Image Preview Card with interactive Crop overlay
                        ZStack {
                            let displayImage = (showFilterEnhance && enhancedImage != nil) ? enhancedImage! : originalImage
                            
                            Image(uiImage: displayImage)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: 260)
                                .cornerRadius(18)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 18)
                                        .stroke(Color.white.opacity(0.15), lineWidth: 1)
                                )
                                .animation(.easeInOut(duration: 0.25), value: showFilterEnhance)
                            
                            // Crop Recommendation Box
                            if showCropPreview, let crop = coachEngine.lastCritique?.cropSuggestion {
                                GeometryReader { geo in
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.yellow, lineWidth: 2)
                                        .background(Color.yellow.opacity(0.1))
                                        .frame(
                                            width: geo.size.width * crop.width,
                                            height: geo.size.height * crop.height
                                        )
                                        .position(
                                            x: geo.size.width * crop.midX,
                                            y: geo.size.height * crop.midY
                                        )
                                        .overlay(
                                            Text("SUGGESTED CROP")
                                                .font(.system(size: 9, weight: .black, design: .monospaced))
                                                .foregroundColor(.black)
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 3)
                                                .background(Color.yellow)
                                                .cornerRadius(4)
                                                .position(
                                                    x: geo.size.width * crop.midX,
                                                    y: (geo.size.height * crop.minY) - 10
                                                )
                                        )
                                }
                                .padding()
                            }
                        }
                        .padding(.horizontal, 16)
                        
                        // Analysis State Indicator
                        if coachEngine.isAnalyzingPhoto {
                            HStack(spacing: 12) {
                                ProgressView()
                                    .tint(.emerald)
                                Text("Gemma 4 Edge is analyzing composition...")
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(.white.opacity(0.8))
                            }
                            .padding(.vertical, 30)
                        } else if let critique = coachEngine.lastCritique {
                            // Overall Score & Radar Card
                            VStack(spacing: 16) {
                                HStack(alignment: .center, spacing: 20) {
                                    // Big Score Ring
                                    ZStack {
                                        Circle()
                                            .stroke(Color.white.opacity(0.1), lineWidth: 8)
                                            .frame(width: 86, height: 86)
                                        
                                        Circle()
                                            .trim(from: 0, to: CGFloat(critique.overallScore / 10.0))
                                            .stroke(
                                                LinearGradient(
                                                    colors: [.emerald, .cyan],
                                                    startPoint: .topLeading,
                                                    endPoint: .bottomTrailing
                                                ),
                                                style: StrokeStyle(lineWidth: 8, lineCap: .round)
                                            )
                                            .frame(width: 86, height: 86)
                                            .rotationEffect(.degrees(-90))
                                        
                                        VStack(spacing: 0) {
                                            Text(String(format: "%.1f", critique.overallScore))
                                                .font(.system(size: 26, weight: .black, design: .rounded))
                                                .foregroundColor(.white)
                                            Text("OUT OF 10")
                                                .font(.system(size: 8, weight: .bold))
                                                .foregroundColor(.white.opacity(0.5))
                                        }
                                    }
                                    
                                    // 5 Core Sub-Scores (lenscoach_v2_plan.md Section 7)
                                    VStack(alignment: .leading, spacing: 6) {
                                        ScoreBar(label: "Composition", score: critique.compositionScore, color: .cyan)
                                        ScoreBar(label: "Lighting", score: critique.lightingScore, color: .yellow)
                                        ScoreBar(label: "Color Palette", score: critique.colorScore, color: .pink)
                                        ScoreBar(label: "Sharpness", score: critique.focusScore, color: .emerald)
                                        ScoreBar(label: "Subject", score: critique.subjectScore, color: .purple)
                                    }
                                }
                                
                                Text(critique.summary)
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(.white.opacity(0.9))
                                    .lineSpacing(4)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .padding(18)
                            .background(Color.white.opacity(0.06))
                            .cornerRadius(20)
                            .padding(.horizontal, 16)
                            
                            // Interactive Fixes & Crop Controls
                            HStack(spacing: 12) {
                                Button(action: { withAnimation { showCropPreview.toggle() } }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: showCropPreview ? "crop" : "crop.rotate")
                                        Text(showCropPreview ? "Hide Crop" : "Preview Crop")
                                    }
                                    .font(.system(size: 13, weight: .bold))
                                    .foregroundColor(showCropPreview ? .black : .white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(showCropPreview ? Color.yellow : Color.white.opacity(0.1))
                                    .cornerRadius(12)
                                }
                                
                                Button(action: {
                                    if enhancedImage == nil {
                                        enhancedImage = CoreImageEditService.shared.applyCoachEdits(
                                            to: originalImage,
                                            edits: critique.edits
                                        )
                                    }
                                    withAnimation { showFilterEnhance.toggle() }
                                }) {
                                    HStack(spacing: 6) {
                                        Image(systemName: showFilterEnhance ? "wand.and.stars.inverse" : "wand.and.stars")
                                        Text(showFilterEnhance ? "Original" : "Coach Edits")
                                    }
                                    .font(.system(size: 13, weight: .bold))
                                    .foregroundColor(showFilterEnhance ? .black : .white)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(showFilterEnhance ? Color.emerald : Color.white.opacity(0.1))
                                    .cornerRadius(12)
                                }
                            }
                            .padding(.horizontal, 16)
                            
                            // Strengths Section
                            if !critique.strengths.isEmpty {
                                VStack(alignment: .leading, spacing: 10) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "checkmark.seal.fill")
                                            .foregroundColor(.emerald)
                                        Text("WHAT WORKED WELL")
                                            .font(.system(size: 11, weight: .black, design: .monospaced))
                                            .foregroundColor(.emerald)
                                    }
                                    
                                    ForEach(critique.strengths, id: \.self) { item in
                                        HStack(alignment: .top, spacing: 8) {
                                            Circle()
                                                .fill(Color.emerald)
                                                .frame(width: 5, height: 5)
                                                .padding(.top, 6)
                                            Text(item)
                                                .font(.system(size: 13, weight: .medium))
                                                .foregroundColor(.white.opacity(0.85))
                                        }
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(16)
                                .background(Color.emerald.opacity(0.08))
                                .cornerRadius(16)
                                .padding(.horizontal, 16)
                            }
                            
                            // Structured Issues Section (lenscoach_v2_plan.md Section 7)
                            if !critique.issues.isEmpty {
                                VStack(alignment: .leading, spacing: 12) {
                                    HStack(spacing: 6) {
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .foregroundColor(.yellow)
                                        Text("ACTIONABLE OPPORTUNITIES")
                                            .font(.system(size: 11, weight: .black, design: .monospaced))
                                            .foregroundColor(.yellow)
                                    }
                                    
                                    ForEach(critique.issues) { issue in
                                        VStack(alignment: .leading, spacing: 4) {
                                            HStack {
                                                Text(issue.aspect.uppercased())
                                                    .font(.system(size: 9, weight: .black, design: .monospaced))
                                                    .foregroundColor(.black)
                                                    .padding(.horizontal, 6)
                                                    .padding(.vertical, 2)
                                                    .background(Color.yellow)
                                                    .cornerRadius(4)
                                                
                                                Text(issue.problem)
                                                    .font(.system(size: 12, weight: .semibold))
                                                    .foregroundColor(.white.opacity(0.9))
                                            }
                                            
                                            HStack(alignment: .top, spacing: 6) {
                                                Image(systemName: "arrow.turn.down.right")
                                                    .font(.system(size: 10, weight: .bold))
                                                    .foregroundColor(.yellow.opacity(0.8))
                                                Text(issue.fix)
                                                    .font(.system(size: 12, weight: .medium))
                                                    .foregroundColor(.white.opacity(0.75))
                                            }
                                            .padding(.leading, 4)
                                        }
                                        .padding(10)
                                        .background(Color.white.opacity(0.04))
                                        .cornerRadius(10)
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(16)
                                .background(Color.yellow.opacity(0.08))
                                .cornerRadius(16)
                                .padding(.horizontal, 16)
                            }
                            
                            // Next Shot Idea (lenscoach_v2_plan.md Section 7)
                            VStack(alignment: .leading, spacing: 8) {
                                HStack(spacing: 6) {
                                    Image(systemName: "camera.viewfinder")
                                        .foregroundColor(.cyan)
                                    Text("NEXT SHOT RECOMMENDATION")
                                        .font(.system(size: 11, weight: .black, design: .monospaced))
                                        .foregroundColor(.cyan)
                                }
                                
                                Text(critique.nextShot)
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(.white.opacity(0.9))
                                    .lineSpacing(3)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(16)
                            .background(Color.cyan.opacity(0.08))
                            .cornerRadius(16)
                            .padding(.horizontal, 16)
                        }
                    }
                    .padding(.bottom, 30)
                }
            }
        }
    }
}

// Sub-Score Progress Bar
struct ScoreBar: View {
    let label: String
    let score: Int
    let color: Color
    
    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.white.opacity(0.7))
                .frame(width: 80, alignment: .leading)
            
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.white.opacity(0.1))
                        .frame(height: 6)
                    
                    Capsule()
                        .fill(color)
                        .frame(width: geo.size.width * CGFloat(Double(score) / 10.0), height: 6)
                }
            }
            .frame(height: 6)
            
            Text("\(score)")
                .font(.system(size: 11, weight: .black, design: .monospaced))
                .foregroundColor(color)
                .frame(width: 20, alignment: .trailing)
        }
    }
}

// Custom Emerald Palette
extension Color {
    static let emerald = Color(red: 0.18, green: 0.80, blue: 0.44)
}
