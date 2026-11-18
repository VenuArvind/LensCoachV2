import UIKit
import os

public class DiagnosticsManager {
    public static let shared = DiagnosticsManager()
    private var timer: Timer?
    
    // Performance metrics
    public var lastInferenceLatency: Double = 0.0
    private var frameCount: Int = 0
    private var currentFPS: Int = 0
    private var vlmCallCount: Int = 0
    private var lastVlmTtftMs: Double = 0.0
    private var lastVlmTokensPerSec: Double = 0.0
    private var lastVlmTotalMs: Double = 0.0
    
    // Returns memory used by the app in Megabytes (RSS) and Jetsam phys_footprint
    private func getMemoryUsageMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / (1024 * 1024)
        } else {
            return 0.0
        }
    }

    private func getPhysFootprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return Double(info.phys_footprint) / (1024 * 1024)
        }
        return 0.0
    }
    
    // Public hook for CameraManager to report a frame was processed
    public func reportFrame() {
        frameCount += 1
    }
    
    // Public hook for AestheticAnalyzer to report latency
    public func reportInferenceLatency(_ ms: Double) {
        lastInferenceLatency = ms
    }

    // Public hook for GemmaCoachEngine to report VLM call count, TTFT, throughput, and total latency
    public func reportVLMCall(ttftMs: Double, tokensPerSec: Double, totalMs: Double) {
        DispatchQueue.main.async {
            self.vlmCallCount += 1
            self.lastVlmTtftMs = ttftMs
            self.lastVlmTokensPerSec = tokensPerSec
            self.lastVlmTotalMs = totalMs
        }
    }
    
    // Starts logging diagnostics to the console and a local CSV file
    public func startLogging() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        
        let fileManager = FileManager.default
        guard let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let logURL = documentsURL.appendingPathComponent("diagnostics.csv")
        
        // Ensure file exists with header
        if !fileManager.fileExists(atPath: logURL.path) {
            let header = "Timestamp,FPS,Latency_ms,Memory_MB,PhysFootprint_MB,ThermalState,Battery_Percent,VLM_Calls,VLM_TTFT_ms,VLM_Tok_s,VLM_Total_ms\n"
            try? header.write(to: logURL, atomically: true, encoding: .utf8)
        }
        
        self.timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            let memoryMB = self.getMemoryUsageMB()
            let physMB = self.getPhysFootprintMB()
            let thermalState = ProcessInfo.processInfo.thermalState.rawValue
            let batteryLevel = UIDevice.current.batteryLevel * 100
            
            // Calculate FPS
            self.currentFPS = self.frameCount
            self.frameCount = 0
            
            let timestamp = Date().timeIntervalSince1970
            let logLine = "\(timestamp),\(self.currentFPS),\(String(format: "%.1f", self.lastInferenceLatency)),\(String(format: "%.1f", memoryMB)),\(String(format: "%.1f", physMB)),\(thermalState),\(Int(batteryLevel)),\(self.vlmCallCount),\(String(format: "%.1f", self.lastVlmTtftMs)),\(String(format: "%.1f", self.lastVlmTokensPerSec)),\(String(format: "%.1f", self.lastVlmTotalMs))\n"
            
            // Print to console for immediate visibility
            print("DIAGNOSTICS | \(logLine.trimmingCharacters(in: .newlines))")
            
            // Append to File
            if let fileHandle = try? FileHandle(forWritingTo: logURL) {
                fileHandle.seekToEndOfFile()
                if let data = logLine.data(using: .utf8) {
                    fileHandle.write(data)
                }
                fileHandle.closeFile()
            }
        }
    }
    
    public func stopLogging() {
        self.timer?.invalidate()
        self.timer = nil
    }
}
