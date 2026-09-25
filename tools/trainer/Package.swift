// swift-tools-version: 6.0
// trainer: fine-tunes Qwen3-ASR 0.6B for Sinhala on a Mac with MLX, on the code LiveTranscribe
// runs (mlx-audio-swift's Qwen3ASRModel), and exports the result as a model folder the app loads
// (docs/training.md). Build it with xcodebuild, which compiles MLX's Metal shaders:
//
//   xcodebuild build -scheme Trainer -configuration Release -destination platform=macOS,arch=arm64 \
//     -derivedDataPath .build/xcode -skipPackagePluginValidation CLANG_COVERAGE_MAPPING=NO
//
// and test it with `xcodebuild test` and the same options (-configuration Debug is fine). The
// plug-in check would otherwise stop the build to ask about mlx-swift's CUDA build plug-in, which
// a Mac build doesn't run.
// Xcode otherwise builds a package's tools with code coverage, and every run leaves a
// default.profraw in the folder it ran in.
import PackageDescription

let package = Package(
    name: "Trainer",
    platforms: [.macOS(.v14)],
    dependencies: [
        // The versions LiveTranscribe pins, so training runs the code the app runs, except
        // mlx-audio-swift: upstream main as of 2026-09-18, past the app's v0.1.3 pin, for its fix
        // to Qwen3-ASR's audio features (Blaizzy/mlx-audio-swift#247: the Slaney mel scale and a
        // periodic Hann window, as Qwen's own feature extractor computes them). The app must move
        // to this revision to run the fine-tuned model as it was trained. mlx-audio-swift must be
        // pinned by revision: its manifest uses unsafeFlags.
        .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", revision: "01dec7c9bdce3088a6b6b7ab9f2e403458195efb"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.3.4"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", exact: "0.11.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", exact: "1.8.2"),
    ],
    targets: [
        .target(
            name: "TrainerKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "trainer",
            dependencies: [
                "TrainerKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "TrainerKitTests",
            dependencies: ["TrainerKit", .product(name: "MLX", package: "mlx-swift")]
        ),
    ]
)
