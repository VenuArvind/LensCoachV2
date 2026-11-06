#!/usr/bin/env swift
import Foundation
import CoreML
import Vision
import AppKit

// LensCoach 2.0 - Comprehensive On-Device Benchmark Suite
// Evaluates:
// 1. CoreML FrameScore on Apple Neural Engine (ANE)
// 2. Full Perception Pipeline (FrameScore + Apple Vision Face Detection + Saliency)
// 3. Spearman Rank Correlation Coefficient (SRCC) vs AADB Ground Truth
// 4. Resident Memory Footprint (mach_task_basic_info)

struct BenchmarkRecord {
    let filename: String
    let groundTruthScore: Double
    let predictedScore: Double
    let aneLatencyMs: Double
    let fullPerceptionLatencyMs: Double
    let facesDetected: Int
}

func getMemoryUsageMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kerr == KERN_SUCCESS ? Double(info.resident_size) / (1024.0 * 1024.0) : 0.0
}

func computeSRCC(x: [Double], y: [Double]) -> Double {
    let n = Double(x.count)
    guard n > 1 else { return 0.0 }
    
    func getRanks(_ arr: [Double]) -> [Double] {
        let indexed = arr.enumerated().sorted { $0.element < $1.element }
        var ranks = [Double](repeating: 0.0, count: arr.count)
        for (rank, item) in indexed.enumerated() {
            ranks[item.offset] = Double(rank + 1)
        }
        return ranks
    }
    
    let rx = getRanks(x)
    let ry = getRanks(y)
    
    var dSq: Double = 0.0
    for i in 0..<x.count {
        let diff = rx[i] - ry[i]
        dSq += diff * diff
    }
    
    return 1.0 - (6.0 * dSq) / (n * (n * n - 1.0))
}

func percentile(_ sortedArr: [Double], _ p: Double) -> Double {
    guard !sortedArr.isEmpty else { return 0.0 }
    let idx = min(Double(sortedArr.count - 1), max(0.0, p * Double(sortedArr.count - 1)))
    let lower = Int(floor(idx))
    let upper = Int(ceil(idx))
    if lower == upper { return sortedArr[lower] }
    let weight = idx - Double(lower)
    return sortedArr[lower] * (1.0 - weight) + sortedArr[upper] * weight
}

let sep = String(repeating: "=", count: 80)
print(sep)
print("LensCoach 2.0 - On-Device Perception & FrameScore Benchmark")
print(sep)

let projectRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let modelPath = projectRoot.appendingPathComponent("LensCoachApp/Sources/Resources/FrameScore.mlmodelc")
let testCsvPath = projectRoot.appendingPathComponent("Training/data/aadb-imagedatabase/Dataset_test.csv")
let imagesDir = projectRoot.appendingPathComponent("Training/data/aadb-imagedatabase/datasetImages_warp256 2")

print("Initializing CoreML FrameScore model on Apple Silicon...")
let config = MLModelConfiguration()
config.computeUnits = .all
guard let mlModel = try? MLModel(contentsOf: modelPath, configuration: config),
      let vnModel = try? VNCoreMLModel(for: mlModel) else {
    print("❌ Failed to load CoreML model at \(modelPath.path)")
    exit(1)
}

print("✅ FrameScore loaded successfully (Apple Neural Engine + GPU enabled).")

// Parse Dataset_test.csv
guard let csvData = try? String(contentsOf: testCsvPath, encoding: .utf8) else {
    print("❌ Failed to read \(testCsvPath.path)")
    exit(1)
}

let rows = csvData.components(separatedBy: .newlines).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
print("Loaded \(rows.count - 1) test image entries from Dataset_test.csv.")

var testSamples: [(filename: String, gtScore: Double)] = []
for (i, line) in rows.enumerated() {
    if i == 0 { continue } // skip header
    let cols = line.components(separatedBy: ",")
    if cols.count >= 13, let score = Double(cols[12]) {
        testSamples.append((filename: cols[0], gtScore: score))
    }
}

// Select test batch (first 60 valid images)
let sampleLimit = min(60, testSamples.count)
print("Benchmarking across \(sampleLimit) real photographic frames...")

// Warm up Neural Engine pipeline
if let firstSample = testSamples.first {
    let firstUrl = imagesDir.appendingPathComponent(firstSample.filename)
    if let img = NSImage(contentsOf: firstUrl),
       let cgImg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {
        let req = VNCoreMLRequest(model: vnModel)
        let handler = VNImageRequestHandler(cgImage: cgImg, options: [:])
        _ = try? handler.perform([req])
    }
}

var benchmarkRecords: [BenchmarkRecord] = []
var initialMemory = getMemoryUsageMB()

for (idx, sample) in testSamples.prefix(sampleLimit).enumerated() {
    let imgUrl = imagesDir.appendingPathComponent(sample.filename)
    guard let img = NSImage(contentsOf: imgUrl),
          let cgImg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        continue
    }
    
    // 1. Measure CoreML FrameScore alone on ANE
    let fsReq = VNCoreMLRequest(model: vnModel)
    fsReq.imageCropAndScaleOption = .centerCrop
    let fsHandler = VNImageRequestHandler(cgImage: cgImg, options: [:])
    
    let t0 = CFAbsoluteTimeGetCurrent()
    _ = try? fsHandler.perform([fsReq])
    let aneLatency = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
    
    var predScore = 0.5
    if let obs = fsReq.results?.first as? VNCoreMLFeatureValueObservation,
       let arr = obs.featureValue.multiArrayValue {
        predScore = Double(arr[0].floatValue)
    }
    
    // 2. Measure Full Perception Loop (FrameScore + Face Detection + Saliency)
    let fullReq1 = VNCoreMLRequest(model: vnModel)
    fullReq1.imageCropAndScaleOption = .centerCrop
    let faceReq = VNDetectFaceRectanglesRequest()
    let salReq = VNGenerateAttentionBasedSaliencyImageRequest()
    let fullHandler = VNImageRequestHandler(cgImage: cgImg, options: [:])
    
    let t1 = CFAbsoluteTimeGetCurrent()
    _ = try? fullHandler.perform([fullReq1, faceReq, salReq])
    let fullLatency = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0
    
    let faces = faceReq.results?.count ?? 0
    
    let record = BenchmarkRecord(
        filename: sample.filename,
        groundTruthScore: sample.gtScore,
        predictedScore: predScore,
        aneLatencyMs: aneLatency,
        fullPerceptionLatencyMs: fullLatency,
        facesDetected: faces
    )
    benchmarkRecords.append(record)
    
    if (idx + 1) % 15 == 0 || idx == sampleLimit - 1 {
        print(String(format: "  [%02d/%02d] ANE: %.2f ms | Full Loop: %.2f ms | Pred: %.2f (GT: %.2f)",
                     idx + 1, sampleLimit, aneLatency, fullLatency, predScore, sample.gtScore))
    }
}

let finalMemory = getMemoryUsageMB()

// Compute statistics
let aneLatencies = benchmarkRecords.map { $0.aneLatencyMs }.sorted()
let fullLatencies = benchmarkRecords.map { $0.fullPerceptionLatencyMs }.sorted()
let gtScores = benchmarkRecords.map { $0.groundTruthScore }
let predScores = benchmarkRecords.map { $0.predictedScore }

let srcc = computeSRCC(x: gtScores, y: predScores)
var mse: Double = 0.0
for i in 0..<gtScores.count {
    let diff = gtScores[i] - predScores[i]
    mse += diff * diff
}
mse /= Double(max(1, gtScores.count))

let avgAne = aneLatencies.reduce(0, +) / Double(max(1, aneLatencies.count))
let p50Ane = percentile(aneLatencies, 0.50)
let p95Ane = percentile(aneLatencies, 0.95)

let avgFull = fullLatencies.reduce(0, +) / Double(max(1, fullLatencies.count))
let p50Full = percentile(fullLatencies, 0.50)
let p95Full = percentile(fullLatencies, 0.95)

print("\n" + sep)
print("LENSCOACH 2.0 EMPIRICAL BENCHMARK SUMMARY")
print(sep)
print("Frames Evaluated:              \(benchmarkRecords.count)")
print(String(format: "CoreML ANE Latency:            P50: %.2f ms | P95: %.2f ms | Mean: %.2f ms", p50Ane, p95Ane, avgAne))
print(String(format: "Full Perception Pipeline:      P50: %.2f ms | P95: %.2f ms | Mean: %.2f ms", p50Full, p95Full, avgFull))
print(String(format: "Aesthetic Scoring Correlation: SRCC = %.4f (MSE: %.4f)", srcc, mse))
print(String(format: "Resident Process Memory (RSS): Initial: %.1f MB | Peak/Final: %.1f MB", initialMemory, finalMemory))
print(String(format: "Theoretical Viewfinder Cadence: %.1f FPS (Perception loop budget)", 1000.0 / avgFull))
print(sep)

// Save JSON Report
let outputDir = projectRoot.appendingPathComponent("benchmarks")
try? FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
let jsonReportUrl = outputDir.appendingPathComponent("v2_benchmark_results.json")

var jsonOutput: [String: Any] = [:]
jsonOutput["evaluation_version"] = "2.0"
jsonOutput["timestamp"] = Date().timeIntervalSince1970
jsonOutput["frames_evaluated"] = benchmarkRecords.count
jsonOutput["hardware_acceleration"] = "Apple Silicon Neural Engine (ANE) + Apple Vision"

var aneDict: [String: Any] = [:]
aneDict["mean"] = round(avgAne * 100) / 100
aneDict["p50"] = round(p50Ane * 100) / 100
aneDict["p95"] = round(p95Ane * 100) / 100
aneDict["min"] = round((aneLatencies.first ?? 0) * 100) / 100
aneDict["max"] = round((aneLatencies.last ?? 0) * 100) / 100
jsonOutput["ane_latency_ms"] = aneDict

var fullDict: [String: Any] = [:]
fullDict["mean"] = round(avgFull * 100) / 100
fullDict["p50"] = round(p50Full * 100) / 100
fullDict["p95"] = round(p95Full * 100) / 100
fullDict["min"] = round((fullLatencies.first ?? 0) * 100) / 100
fullDict["max"] = round((fullLatencies.last ?? 0) * 100) / 100
jsonOutput["full_perception_latency_ms"] = fullDict

var corrDict: [String: Any] = [:]
corrDict["srcc"] = round(srcc * 10000) / 10000
corrDict["mse"] = round(mse * 10000) / 10000
jsonOutput["aesthetic_correlation"] = corrDict

var memDict: [String: Any] = [:]
memDict["initial"] = round(initialMemory * 10) / 10
memDict["final"] = round(finalMemory * 10) / 10
jsonOutput["resident_memory_mb"] = memDict

var sampleList: [[String: Any]] = []
for r in benchmarkRecords {
    var item: [String: Any] = [:]
    item["file"] = r.filename
    item["gt_score"] = r.groundTruthScore
    item["pred_score"] = round(r.predictedScore * 1000) / 1000
    item["ane_ms"] = round(r.aneLatencyMs * 100) / 100
    item["full_ms"] = round(r.fullPerceptionLatencyMs * 100) / 100
    item["faces"] = r.facesDetected
    sampleList.append(item)
}
jsonOutput["samples"] = sampleList

if let jsonData = try? JSONSerialization.data(withJSONObject: jsonOutput, options: .prettyPrinted) {
    try? jsonData.write(to: jsonReportUrl)
    print("✅ Saved raw benchmark telemetry to \(jsonReportUrl.path)")
}

// Generate v2_dashboard.html
let dashboardUrl = projectRoot.appendingPathComponent("v2_dashboard.html")
let htmlContent = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>LensCoach 2.0 Diagnostics & Benchmark Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; margin: 0; }
h1, h2 { color: #58a6ff; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 30px; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px; }
.stat-box { background: #21262d; border-radius: 6px; padding: 15px; text-align: center; }
.stat-val { font-size: 28px; font-weight: bold; color: #7ee787; margin-top: 5px; }
.stat-lbl { font-size: 13px; color: #8b949e; text-transform: uppercase; }
.chart-container { position: relative; height: 320px; margin-top: 15px; }
</style>
</head>
<body>
<h1>LensCoach 2.0 Empirical Performance Dashboard</h1>
<p style="color: #8b949e;">Measured live on Apple Silicon (Apple Neural Engine + Vision Framework) across \(benchmarkRecords.count) benchmark frames.</p>

<div class="stats-grid">
  <div class="stat-box"><div class="stat-lbl">ANE FrameScore Latency</div><div class="stat-val">\(String(format: "%.2f", p50Ane)) ms</div></div>
  <div class="stat-box"><div class="stat-lbl">Full Perception P95</div><div class="stat-val">\(String(format: "%.2f", p95Full)) ms</div></div>
  <div class="stat-box"><div class="stat-lbl">Aesthetic Score SRCC</div><div class="stat-val">\(String(format: "%.4f", srcc))</div></div>
  <div class="stat-box"><div class="stat-lbl">Resident Memory (RSS)</div><div class="stat-val">\(String(format: "%.1f", finalMemory)) MB</div></div>
</div>

<div class="card">
  <h2>Perception Latency Distribution (ms)</h2>
  <div class="chart-container"><canvas id="latChart"></canvas></div>
</div>

<div class="card">
  <h2>FrameScore Aesthetic Correlation (Predicted vs Ground Truth)</h2>
  <div class="chart-container"><canvas id="corrChart"></canvas></div>
</div>

<script>
const samples = \(String(data: try! JSONSerialization.data(withJSONObject: benchmarkRecords.map { ["ane": $0.aneLatencyMs, "full": $0.fullPerceptionLatencyMs, "gt": $0.groundTruthScore, "pred": $0.predictedScore] }), encoding: .utf8)!);
const labels = samples.map((_, i) => 'Frame ' + (i+1));

new Chart(document.getElementById('latChart'), {
  type: 'line',
  data: {
    labels: labels,
    datasets: [
      { label: 'Full Perception Loop (ms)', data: samples.map(s => s.full), borderColor: '#f78166', backgroundColor: 'rgba(247,129,102,0.1)', fill: true, tension: 0.3 },
      { label: 'CoreML ANE Latency (ms)', data: samples.map(s => s.ane), borderColor: '#7ee787', backgroundColor: 'rgba(126,231,135,0.1)', fill: true, tension: 0.3 }
    ]
  },
  options: { responsive: true, maintainAspectRatio: false, scales: { y: { title: { display: true, text: 'Milliseconds (ms)', color: '#8b949e' } } } }
});

new Chart(document.getElementById('corrChart'), {
  type: 'scatter',
  data: {
    datasets: [{
      label: 'Frame Samples (SRCC: \(String(format: "%.3f", srcc)))',
      data: samples.map(s => ({ x: s.gt, y: s.pred })),
      backgroundColor: '#58a6ff'
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { title: { display: true, text: 'Human Consensus Aesthetic Rating (Ground Truth)', color: '#8b949e' }, min: 0, max: 1 },
      y: { title: { display: true, text: 'FrameScore Predicted Score', color: '#8b949e' }, min: 0, max: 1 }
    }
  }
});
</script>
</body>
</html>
"""

try? htmlContent.write(to: dashboardUrl, atomically: true, encoding: .utf8)
print("✅ Generated v2 interactive dashboard: \(dashboardUrl.path)")
