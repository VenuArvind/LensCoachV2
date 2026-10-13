import SwiftUI
import Charts

public struct GalleryView: View {
    @ObservedObject var gallery = GalleryManager.shared
    @Environment(\.dismiss) var dismiss
    
    @State private var selectedEntry: PhotoEntry? = nil
    
    public init() {}
    
    public var body: some View {
        NavigationView {
            ZStack {
                Color.black.edgesIgnoringSafeArea(.all)
                
                ScrollView {
                    VStack(spacing: 24) {
                        // Analytics Section
                        if gallery.photos.count > 1 {
                            VStack(alignment: .leading, spacing: 12) {
                                Text("AESTHETIC SCORE TRENDS")
                                    .font(.system(size: 13, weight: .black, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.6))
                                
                                ChartView(photos: gallery.photos)
                                    .frame(height: 200)
                                    .padding()
                                    .background(Color.white.opacity(0.05))
                                    .cornerRadius(16)
                            }
                            .padding(.horizontal)
                        }
                        
                        // Gallery Grid
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                            ForEach(gallery.photos) { entry in
                                Button(action: {
                                    selectedEntry = entry
                                }) {
                                    GalleryItemView(entry: entry)
                                }
                                .buttonStyle(PlainButtonStyle())
                            }
                        }
                        .padding(.horizontal)
                    }
                    .padding(.vertical)
                }
            }
            .navigationTitle("Your Portfolio")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                        .foregroundColor(.blue)
                }
            }
            .sheet(item: $selectedEntry) { entry in
                if let image = GalleryManager.shared.loadImage(for: entry) {
                    let coach = GemmaCoachEngine.shared
                    let _ = {
                        if let critique = entry.critique {
                            coach.lastCritique = critique
                        }
                    }()
                    CritiqueView(
                        coachEngine: coach,
                        image: image,
                        onDismiss: { selectedEntry = nil }
                    )
                }
            }
        }
    }
}

struct ChartView: View {
    var photos: [PhotoEntry]
    
    var body: some View {
        Chart {
            ForEach(photos.reversed()) { entry in
                LineMark(
                    x: .value("Date", entry.date),
                    y: .value("Score", entry.aestheticScore)
                )
                .interpolationMethod(.catmullRom)
                .foregroundStyle(LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing))
                
                AreaMark(
                    x: .value("Date", entry.date),
                    y: .value("Score", entry.aestheticScore)
                )
                .interpolationMethod(.catmullRom)
                .foregroundStyle(LinearGradient(colors: [.blue.opacity(0.3), .clear], startPoint: .top, endPoint: .bottom))
            }
        }
        .chartYScale(domain: 0...1)
        .chartXAxis {
            AxisMarks(values: .stride(by: .day)) { _ in
                AxisGridLine().foregroundStyle(.white.opacity(0.1))
                AxisTick().foregroundStyle(.white.opacity(0.1))
                AxisValueLabel().foregroundStyle(.white.opacity(0.5))
            }
        }
        .chartYAxis {
            AxisMarks { value in
                AxisGridLine().foregroundStyle(.white.opacity(0.1))
                AxisValueLabel {
                    if let score = value.as(Double.self) {
                        Text("\(Int(score * 100))%")
                            .foregroundColor(.white.opacity(0.5))
                    }
                }
            }
        }
    }
}

struct GalleryItemView: View {
    let entry: PhotoEntry
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let image = GalleryManager.shared.loadImage(for: entry) {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: (UIScreen.main.bounds.width - 48) / 2, height: 200)
                    .clipped()
                    .cornerRadius(12)
                    .overlay(
                        VStack {
                            Spacer()
                            HStack {
                                if entry.critique != nil {
                                    HStack(spacing: 3) {
                                        Image(systemName: "cpu.fill")
                                            .font(.system(size: 8))
                                        Text("REVIEWED")
                                            .font(.system(size: 8, weight: .black, design: .monospaced))
                                    }
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(Color.black.opacity(0.65))
                                    .foregroundColor(.emerald)
                                    .cornerRadius(6)
                                    .padding(8)
                                }
                                
                                Spacer()
                                
                                Text("\(Int(entry.aestheticScore * 100))%")
                                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(scoreColor(entry.aestheticScore).opacity(0.85))
                                    .cornerRadius(8)
                                    .padding(8)
                            }
                        }
                    )
            }
            
            Text(entry.date, style: .date)
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(.white.opacity(0.5))
        }
    }
    
    func scoreColor(_ score: Float) -> Color {
        if score > 0.7 { return .emerald }
        if score > 0.45 { return .yellow }
        return .orange
    }
}
