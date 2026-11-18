// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LensCoachApp",
    platforms: [
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "LensCoachApp",
            targets: ["LensCoachApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/google-ai-edge/LiteRT-LM", from: "0.9.0")
    ],
    targets: [
        .target(
            name: "LensCoachApp",
            dependencies: [
                .product(name: "LiteRTLM", package: "LiteRT-LM")
            ],
            path: "Sources",
            resources: [
                .copy("Resources/FrameScore.mlmodelc"),
                .copy("Resources/gemma-4-E2B-it.litertlm")
            ])
    ]
)
