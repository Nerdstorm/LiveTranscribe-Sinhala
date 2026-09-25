// swift-tools-version: 6.0
// pseudo-label: labels recordings with Qwen3-ASR on a Mac, the way LiveTranscribe runs it, for the
// replay set (docs/replay.md). Build it with xcodebuild, which compiles MLX's Metal shaders:
//
//   xcodebuild build -scheme PseudoLabel -configuration Release -destination platform=macOS,arch=arm64 \
//     -derivedDataPath .build/xcode -skipPackagePluginValidation CLANG_COVERAGE_MAPPING=NO
//
// Without -skipPackagePluginValidation the build stops to ask about mlx-swift's CUDA build
// plug-in, which a Mac build doesn't run. Without CLANG_COVERAGE_MAPPING=NO, Xcode builds a
// package's tools with code coverage, and every run leaves a default.profraw where it ran.
import PackageDescription

let package = Package(
    name: "PseudoLabel",
    platforms: [.macOS(.v14)],
    dependencies: [
        // The revision LiveTranscribe uses, so the labels are what the app's model writes.
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", revision: "d302a5c6080d2bb97bae38c7418f82abb76013b6"),
    ],
    targets: [
        .target(name: "LabelFile"),
        .executableTarget(
            name: "pseudo-label",
            dependencies: [
                "LabelFile",
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
            ]
        ),
        .testTarget(name: "LabelFileTests", dependencies: ["LabelFile"]),
    ]
)
