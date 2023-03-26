//
// GrAITorchTests.swift
// GrAITorchTests
//
// Created by Jean-François Reboud on 19/10/2022.
//

import XCTest
import GrAIdient

/// Compare models created by GrAIdient and PyTorch.
final class GrAITorchTests: XCTestCase
{
    /// Size of one image (height and width are the same).
    let _size = 32
    /// Kernel split size of one image (height and width are the same).
    let _patch = 8
    
    /// Initialize test.
    override func setUp()
    {
        setPythonLib()
        _ = MetalKernel.get
        GrAI.Opti.GPU = true
    }
    
    ///
    /// Compute the gradient norm on the first layer of the model.
    ///
    /// - Parameter model: The model we want to evalulate the gradient norm on.
    /// - Returns: The gradient norm on the first layer.
    ///
    func _getGradientNorm(_ model: Model) -> Double
    {
        // Create the context to build a graph of layers
        // that come after the layers inside `model`.
        let context = ModelContext(name: "ModelTest", models: [model])
        let params = GrAI.Model.Params(context: context)
        
        // Append a loss layer.
        let lastLayer = MSE1D(
            layerPrev: model.layers.last! as! Layer1D,
            params: params
        )
        lastLayer.coeff = 1.0 / 2.0
        
        // Initialize the finalModel with the links (`layerPrev` updated).
        let finalModel = Model(model: context.model, modelsPrev: [model])
        
        // Initialize for inference.
        finalModel.initKernel(phase: .Inference)
        // The final model contains the layers of `model` and the loss layer.
        finalModel.layers = model.layers + context.model.layers
        
        let optimizerParams = getOptimizerParams(nbLoops: 1)
        finalModel.setupOptimizers(params: optimizerParams)
        
        let groundTruth: [[Double]] = [[0.0]]
        let firstLayer: Input2D = finalModel.layers.first as! Input2D
        
        // Update internal batch size.
        finalModel.updateKernel(batchSize: 1)
        
        // Forward.
        try! finalModel.forward()
        
        // Apply loss derivative.
        try! lastLayer.lossDerivativeGPU(groundTruth)
        
        // Backward.
        try! finalModel.backward()
        
        // Get the gradient norm on the first layer.
        let gradNormOutput: Double =
            try! finalModel.getGradientNorm(layers: [firstLayer])
        return gradNormOutput
    }
    
    /// Test that modelConv1 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelConv1()
    {
        // Build model.
        let model = ModelTestConv1.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConv1GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelConv2 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelConv2()
    {
        // Build model.
        let model = ModelTestConv2.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeConv2GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelFFT backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelFFT()
    {
        // Build model.
        let model = ModelTestFFT.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        let secondLayer: FTFrequences2D = model.layers[1] as! FTFrequences2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getComplexData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        try! secondLayer.setDataGPU(batchSize: 1)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeFFTGradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelDeConv1 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelDeConv1()
    {
        // Build model.
        let model = ModelTestDeConv1.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConv1GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelDeConv2 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelDeConv2()
    {
        // Build model.
        let model = ModelTestDeConv2.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConv2GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelDeConv3 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelDeConv3()
    {
        // Build model.
        let model = ModelTestDeConv3.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConv3GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelDeConv4 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelDeConv4()
    {
        // Build model.
        let model = ModelTestDeConv4.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeDeConv4GradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelCat backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelCat()
    {
        // Build model.
        let model = ModelTestCat.build(_size)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeCatGradNorm(_size))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelPatchConv backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelPatchConv()
    {
        // Build model.
        let model = ModelTestPatchConv.build(size: _size, patch: _patch)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computePatchConvGradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelAttention1 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelAttention1()
    {
        // Build model.
        let model = ModelTestAttention1.build(size: _size, patch: _patch)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeAttention1GradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    /// Test that modelAttention2 backward pass returns the same gradient norm in GrAIdient and PyTorch.
    func testModelAttention2()
    {
        // Build model.
        let model = ModelTestAttention2.build(size: _size, patch: _patch)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeAttention2GradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
    
    ///
    /// Test that modelLayerNorm backward pass returns the same gradient norm
    /// in GrAIdient and PyTorch.
    ///
    func testModelLayerNorm()
    {
        // Build model.
        let model = ModelTestLayerNorm.build(size: _size, patch: _patch)
        
        // Initialize for inference.
        model.initKernel(phase: .Inference)
        // Avoid the compute of every gradients of weights.
        model.computeDeltaWeights = false
        
        let firstLayer: Input2D = model.layers.first as! Input2D
        // Allow backward pass go through the first layer.
        firstLayer.computeDelta = true
        // Allow to compute the gradients of weights for the first layer.
        firstLayer.computeDeltaWeights = true
        
        // Set data.
        let data: [Float] = getInputData(_size)
        try! firstLayer.setDataGPU(data, batchSize: 1, format: .RGB)
        
        // Get the gradient norm on the first layer.
        let expectedNorm: Double = Double(computeLayerNormGradNorm(
            size: _size, patch: _patch
        ))
        let gradNormOutput: Double = _getGradientNorm(model)
        
        // Compare difference.
        let diffPercent =
            abs(gradNormOutput - expectedNorm) / expectedNorm * 100.0
        XCTAssert(diffPercent < 1.0)
    }
}
