//
// Input2DMSE2DCase.swift
// GrAITests
//
// Created by Jean-François Reboud on 05/03/2023.
//

import XCTest
import GrAIdient
import GrAITestsUtils

///
/// A class that will test a model with a structural hypothesis:
/// the model last layer is a MSE2D layer, the model first layer is a Input2D.
///
class Input2DMSE2DCase: XCTestCase, Input2DCase, IOCase
{
    /// Height of the Input2D layer.
    var height = 6
    /// Width of the Input2D layer.
    var width = 6
    
    /// Batch size of data.
    var batchSize: Int = -1
    /// Optimizer parameters.
    var optimizerParams = GrAI.Optimizer.Params()
    
    /// Systematic call before test begins.
    override func setUp()
    {
        batchSize = 5
        _ = MetalKernel.get
        
        GrAI.Opti.GPU = true
        GrAI.Precision.float = true
        
        setOptimizerParams(params: &optimizerParams)
        optimizerParams.nbLoops = 2
    }
    
    ///
    /// A function to create/set ground truth to the model.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth to set.
    ///     - model: The model.
    /// - Returns: The ground truth.
    ///
    func setLoss(_ groundTruth: [[Double]]?, _ model: Model) -> [[Double]]
    {
        let lastLayer = model.layers.last as! MSE2D
        let gt: [[Double]]
        if let groundTruthTmp = groundTruth
        {
            gt = groundTruthTmp
        }
        else
        {
            gt = buildData(dim1: getBatchSize(model), dim2: height * width)
        }
        
        if GrAI.Opti.GPU
        {
            try! lastLayer.lossDerivativeGPU(
                gt.reduce([], +),
                batchSize: gt.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        else
        {
            try! lastLayer.lossDerivativeCPU(
                gt.reduce([], +),
                batchSize: gt.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        return gt
    }
    
    ///
    /// A function to get loss of a model.
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth to set.
    ///     - model: The model.
    /// - Returns: The loss value.
    ///
    func getLoss(_ groundTruth: [[Double]], _ model: Model) -> Double
    {
        let lastLayer = model.layers.last as! MSE2D
        if GrAI.Opti.GPU
        {
            return Double(try! lastLayer.getLossGPU(
                groundTruth.reduce([], +),
                batchSize: groundTruth.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            ))
        }
        else
        {
            return try! lastLayer.getLossCPU(
                groundTruth.reduce([], +),
                batchSize: groundTruth.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
    }
    
    ///
    /// A function to get the gradients of weights approximations..
    ///
    /// - Parameters:
    ///     - groundTruth: The ground truth.
    ///     - model: The model.
    /// - Returns: The gradients of weights approximations.
    ///
    func getGradientsApprox(
        _ groundTruth: [[Double]],
        _ model: Model) -> [Double]
    {
        let lastLayer = model.layers.last as! MSE2D
        return try! lastLayer.collectGradientsApprox(
            groundTruth.reduce([], +),
            batchSize: groundTruth.count,
            nbChannels: 1, height: height, width: width,
            format: .Neuron
        )
    }
    
    ///
    /// A function to create/set data to the model.
    ///
    /// - Parameters:
    ///     - inputs: The data to set.
    ///     - model: The model.
    /// - Returns: (The data, the batch size).
    ///
    func setData(_ inputs: [[Double]]?, _ model: Model) -> ([[Double]], Int)
    {
        let firstLayer = model.layers.first as! Input2D
        let ins: [[Double]]
        if let insTmp = inputs
        {
            ins = insTmp
        }
        else
        {
            ins = buildData(dim1: getBatchSize(model), dim2: height * width)
        }
        
        if GrAI.Opti.GPU
        {
            try! firstLayer.setDataGPU(
                ins.reduce([], +),
                batchSize: ins.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        else
        {
            try! firstLayer.setDataCPU(
                ins.reduce([], +),
                batchSize: ins.count,
                nbChannels: 1, height: height, width: width,
                format: .Neuron
            )
        }
        return (ins, ins.count)
    }
}
