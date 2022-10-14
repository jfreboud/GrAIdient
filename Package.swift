// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MAKit",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "MAKit",
            targets: ["MAKit", "MAKitTestsUtils"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/icanzilb/Retry",
            branch: "main"
        ),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "MAKit",
            dependencies: [],
            resources: [
                .copy("Metal/Kernel")
            ]
        ),
        .target(
            name: "MAKitTestsUtils",
            dependencies: ["MAKit"]
        ),
        .testTarget(
            name: "MAKitTests",
            dependencies: ["MAKit", "MAKitTestsUtils", "Retry"]
        ),
    ]
)
