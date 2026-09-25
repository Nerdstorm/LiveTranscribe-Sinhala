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
        // Upstream main as of 2026-09-18, past LiveTranscribe's v0.1.3 pin, for its fix to
        // Qwen3-ASR's audio features (Blaizzy/mlx-audio-swift#247: the Slaney mel scale and a
        // periodic Hann window, as Qwen's own feature extractor computes them). The fine-tune
        // trains on these features, and the app must move to this revision to run it as trained.
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", revision: "01dec7c9bdce3088a6b6b7ab9f2e403458195efb"),
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
