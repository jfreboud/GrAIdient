//
// FullyConnectedSeq.swift
// GrAIdient
//
// Created by Jean-François Reboud on 08/03/2023.
//

import Foundation

/// Layer with a sequential shape neural structure, weights and biases and an activation function.
public class FullyConnectedSeq: ActivationSeq,
                                LayerWithActivation, LayerWeightInit
{
    ///
    /// Grid of weights.
    /// Shape ~ (nbNeurons, nbNeuronsPrev).
    ///
    var _wArrays: WeightGrids! = nil
    ///
    /// Array of biases.
    /// Shape ~ (nbNeurons,).
    ///
    var _bArrays: WeightArrays! = nil
    
    ///
    /// Buffer of weights.
    /// Shape ~ (nbNeurons, nbNeuronsPrev).
    ///
    var _wBuffers: IWeightBuffers! = nil
    ///
    /// Buffer of biases.
    /// Shape ~ (nbNeurons,).
    ///
    var _bBuffers: IWeightBuffers! = nil
    
    ///
    /// Buffer of gradients per sample for weights.
    /// Shape ~ (batch, nbNeurons, nbNeuronsPrev).
    ///
    var _wDeltaWeights: FloatBuffer! = nil
    ///
    /// Buffer of gradients per sample.
    /// Shape ~ (batch, nbNeurons).
    ///
    var _bDeltaWeights: FloatBuffer! = nil
    
    /// Whether to compute weights' gradients or not.
    public var computeDeltaWeights: Bool = true
    
    /// Whether gradients of weights must be accumulated or not.
    public var accumulateDeltaWeights: Bool = false
    
    /// Height of the weight's grid shape.
    public let weightHeight: Int
    /// Width of the weight's grid shape.
    public let weightWidth: Int
    
    /// Whether to update biases or not.
    var _updateBiases: Bool = true
    
    /// Cache for weights before calling `initKernel` API.
    var _weightsList = [Float]()
    
    /// Weights in the CPU execution context.
    public var weightsCPU: [Float]
    {
        get {
            if _wArrays == nil
            {
                return _weightsList
            }
            
            var weightsTmp = [Float]()
            for i in 0..<weightHeight {
            for j in 0..<weightWidth
            {
                weightsTmp.append(Float(_wArrays.w(i, j)))
            }}
            
            if _updateBiases {
            for depth in 0..<weightHeight
            {
                weightsTmp.append(Float(_bArrays.w[depth]))
            }}
            return weightsTmp
        }
        set {
            _weightsList = newValue
        }
    }
    
    /// Weights in the GPU execution context.
    public var weightsGPU: [Float]
    {
        get {
            if _wBuffers == nil
            {
                return _weightsList
            }
            
            var weightsTmp = _wBuffers.w.download()
            if _updateBiases
            {
                weightsTmp += _bBuffers.w.download()
            }
            return weightsTmp
        }
        set {
            _weightsList = newValue
        }
    }
    
    /// Method used to initialize weights values.
    public var weightInitClass: WeightInitClass = .XavierUniform
    
    /// Get the number of input and output connections.
    public var connectivityIO: (Int, Int)
    {
        get {
            return (weightWidth, weightHeight)
        }
    }
    
    /// Number of new weights due to this layer, estimated during the Gradient Checking.
    var nbLearnedGC: Int
    {
        get {
            if !_updateBiases
            {
                return nbNeurons * weightWidth
            }
            else
            {
                return nbNeurons * (weightWidth + 1)
            }
        }
    }
    
    private enum Keys: String, CodingKey
    {
        case weightWidth
        case weightHeight
        case weights
        case updateBiases
    }
    
    ///
    /// Create a layer with a sequential shape neural structure.
    ///
    /// - Parameters:
    ///     - layerPrev: Previous layer that has been queued to the model.
    ///     - nbNeurons: Number of neurons.
    ///     - activation: The activation function.
    ///     - biases: Whether to update biases or not.
    ///     - params: Contextual parameters linking to the model.
    ///
    public init(layerPrev: LayerSeq,
                nbNeurons: Int, activation: String?, biases: Bool,
                params: GrAI.Model.Params)
    {
        weightWidth = layerPrev.nbNeurons
        weightHeight = nbNeurons
        _updateBiases = biases
        
        super.init(layerPrev: layerPrev,
                   sequence: layerPrev.sequence,
                   nbNeurons: nbNeurons,
                   activation: activation,
                   params: params)
    }
    
    ///
    /// Decode from the disk.
    ///
    /// Throw an error if reading from the decoder fails, or
    /// if the data read is corrupted or otherwise invalid.
    ///
    /// - Parameter decoder: The decoder to read data from.
    ///
    public required init(from decoder: Decoder) throws
    {
        let values = try decoder.container(keyedBy: Keys.self)
        _updateBiases = try values.decode(Bool.self, forKey: .updateBiases)
        weightWidth = try values.decode(Int.self, forKey: .weightWidth)
        weightHeight = try values.decode(Int.self, forKey: .weightHeight)
        
        try super.init(from: decoder)
        
        let weightsList = try values.decode([Float].self, forKey: .weights)
        self.weightsCPU = weightsList
    }
    
    ///
    /// Encode to the disk.
    ///
    /// If the value fails to encode anything, `encoder` will encode an empty
    /// keyed container in its place.
    ///
    /// Throw an error if any values are invalid for the given
    /// encoder's format.
    ///
    /// - Parameter encoder: The encoder to write data to.
    ///
    public override func encode(to encoder: Encoder) throws
    {
        var container = encoder.container(keyedBy: Keys.self)
        
        try container.encode(_updateBiases, forKey: .updateBiases)
        try container.encode(weightWidth, forKey: .weightWidth)
        try container.encode(weightHeight, forKey: .weightHeight)
        
        let weightsList: [Float]
        if GrAI.Opti.GPU
        {
            weightsList = self.weightsGPU
        }
        else
        {
            weightsList = self.weightsCPU
        }
        try container.encode(weightsList, forKey: .weights)
        
        try super.encode(to: encoder)
    }
    
    ///
    /// Create a layer with same values as this.
    ///
    /// - Parameters:
    ///     - mapping: Dictionary allowing to find the layer associated to some id.
    ///     This dictionary is particularly useful when the different layers cannot access
    ///     their `layerPrev`.
    ///     - inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new layer. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public override func copy(
        mapping: Dictionary<Int, Layer>,
        inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = mapping[idPrev] as! LayerSeq
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
            
        let layer = FullyConnectedSeq(
            layerPrev: layerPrev,
            nbNeurons: nbNeurons,
            activation: _activation?.name,
            biases: _updateBiases,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._bArrays = _bArrays
            layer._wBuffers = _wBuffers
            layer._bBuffers = _bBuffers
        }
        else
        {
            if GrAI.Opti.GPU
            {
                layer.weightsGPU = weightsGPU
            }
            else
            {
                layer.weightsCPU = weightsCPU
            }
        }
        return layer
    }
    
    ///
    /// Extract main operation of this layer without the activation part.
    ///
    /// This API will create a new layer in the same context as this.
    ///
    /// - Parameter inPlace: Whether hard resources should be copied as is.
    ///
    /// - Returns: A new instance of `Layer`. When `inPlace` is false, `initKernel` is
    /// necessary in order to recreate hard resources.
    ///
    public func removeActivation(inPlace: Bool) -> Layer
    {
        let context = ModelContext(name: "", curID: 0)
        let layerPrev = self.layerPrev as! LayerSeq
        
        let params = GrAI.Model.Params(context: context)
        params.context.curID = id
        
        let layer = FullyConnectedSeq(
            layerPrev: layerPrev,
            nbNeurons: nbNeurons,
            activation: nil,
            biases: _updateBiases,
            params: params
        )
        if inPlace
        {
            layer._wArrays = _wArrays
            layer._bArrays = _bArrays
            layer._wBuffers = _wBuffers
            layer._bBuffers = _bBuffers
        }
        else
        {
            if GrAI.Opti.GPU
            {
                layer.weightsGPU = weightsGPU
            }
            else
            {
                layer.weightsCPU = weightsCPU
            }
        }
        return layer
    }
    
    ///
    /// Extract main operation of this layer without the activation part.
    ///
    /// - Parameter params: Contextual parameters linking to the model.
    ///
    /// - Returns: A new layer.
    ///
    public func removeActivation(params: GrAI.Model.Params) -> Layer
    {
        let layerPrev = self.layerPrev as! LayerSeq
        let layer = FullyConnectedSeq(
            layerPrev: layerPrev,
            nbNeurons: nbNeurons,
            activation: nil,
            biases: _updateBiases,
            params: params
        )
        if GrAI.Opti.GPU
        {
            layer.weightsGPU = weightsGPU
        }
        else
        {
            layer.weightsCPU = weightsCPU
        }
        return layer
    }
    
    ///
    /// Clean state resources in the CPU execution context.
    ///
    /// We first clean the neurons' state (forward and backward).
    /// We do not clean weights and biases but must reset their delta (dependent on batch size) and
    /// momentum state.
    ///
    public override func resetKernelCPU()
    {
        super.resetKernelCPU()
        
        _wArrays?.reset()
        _bArrays?.reset()
    }
    
    ///
    /// Clean state resources in the GPU execution context.
    ///
    /// We first clean the neurons' state (forward and backward).
    /// We do not clean weights and biases but must reset their delta (dependent on batch size) and
    /// momentum state.
    ///
    public override func resetKernelGPU()
    {
        super.resetKernelGPU()
        
        _wDeltaWeights = nil
        _bDeltaWeights = nil
        
        _wBuffers?.reset()
        _bBuffers?.reset()
    }
    
    ///
    /// Initialize weights in the CPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    public func initWeightsCPU()
    {
        if _weightsList.count == 0
        {
            _weightsList = generateWeightsList()
            _weightsList += [Float](repeating: 0.0, count: weightHeight)
        }
        
        _wArrays = WeightGrids(width: weightWidth, height: weightHeight)
        _bArrays = WeightArrays(weightHeight)
        
        for i in 0..<weightHeight {
        for j in 0..<weightWidth
        {
            let offset = j + weightWidth * i
            _wArrays.w(i, j, Double(_weightsList[offset]))
        }}
    
        // In both cases, biases may have been set by caller or by ourselves.
        if _updateBiases
        {
            let offset = weightHeight * weightWidth
            for depth in 0..<weightHeight
            {
                _bArrays.w[depth] = Double(_weightsList[offset + depth])
            }
        }
        else
        {
            for depth in 0..<weightHeight
            {
                _bArrays.w[depth] = 0.0
            }
        }
        _weightsList = []
    }
    
    ///
    /// Initialize weights in the GPU execution context.
    ///
    /// Their momentum and delta state are also reset.
    ///
    public func initWeightsGPU()
    {
        _wBuffers = WeightBuffers(
            nbElems: weightHeight * weightWidth,
            deviceID: deviceID
        )
        _bBuffers = WeightBuffers(
            nbElems: weightHeight,
            deviceID: deviceID
        )
        
        _bBuffers.w.initialize()
        if _weightsList.count == 0
        {
            generateWeightsList(out: _wBuffers.w, deviceID: deviceID)
        }
        else
        {
            _wBuffers.w.initialize(array: &_weightsList)
            if _updateBiases
            {
                _bBuffers.w.initialize(
                    array: &_weightsList,
                    start: weightHeight * weightWidth
                )
            }
        }
        
        _weightsList = []
        _wDeltaWeights = nil
        _bDeltaWeights = nil
    }
    
    ///
    /// Initialize state resources in the GPU execution context.
    ///
    /// We initialize the neurons' forward state.
    /// We initialize the weights and biases' delta.
    ///
    public override func checkStateForwardGPU(batchSize: Int) throws
    {
        try super.checkStateForwardGPU(batchSize: batchSize)
        
        if computeDeltaWeights &&
           GrAI.Gradient.sample && _wDeltaWeights == nil
        {
            _wDeltaWeights = FloatBuffer(nbElems: 
                batchSize * sequence * nbNeurons * weightWidth,
                deviceID: deviceID
            )
            
            if _updateBiases
            {
                _bDeltaWeights = FloatBuffer(nbElems: 
                    batchSize * sequence * nbNeurons, deviceID: deviceID
                )
            }
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCCPU() throws
    {
        try _forwardGCCPU()
        _activation?.forwardGC(self)
    }
    
    private func _forwardGCCPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                neurons.get(seq, depth)!.initGC(
                    batchSize: batchSize, nbGC: newGC
                )
            }}
            
            let neuronsPrev = layerPrev.neurons!
            let nbNeuronsPrev = layerPrev.nbNeurons
            
            for elem in 0..<nbGC {
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                for depth in 0..<nbNeurons
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbNeuronsPrev
                    {
                        let w = _wArrays.w(depth, depthPrev)
                        let outPrev = neuronsPrev
                            .get(seq, depthPrev)!.gc[batch][elem].out
                        tmp += w * outPrev
                    }
                    neurons.get(seq, depth)!.gc[batch][elem].out = tmp
                }
            }}}
            
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                for DEPTH in 0..<nbNeurons {
                for DEPTHPREV in 0..<weightWidth {
                for elem in 0...1
                {
                    for depth in 0..<nbNeurons
                    {
                        var tmp: Double = _bArrays.w[depth]
                        for depthPrev in 0..<nbNeuronsPrev
                        {
                            var w = _wArrays.w(depth, depthPrev)
                            if depth == DEPTH && depthPrev == DEPTHPREV
                            {
                                if elem % 2 == 0
                                {
                                    w += Ɛ
                                }
                                else
                                {
                                    w -= Ɛ
                                }
                            }
                            
                            let outPrev = neuronsPrev
                                .get(seq, depthPrev)!.v[batch].out
                            tmp += w * outPrev
                        }
                        
                        let offset = nbGC +
                            2 * (DEPTHPREV + weightWidth * DEPTH) + elem
                        neurons.get(seq, depth)!.gc[batch][offset].out = tmp
                    }
                }}}
            }}
            
            if _updateBiases {
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                for DEPTH in 0..<nbNeurons {
                for elem in 0...1
                {
                    for depth in 0..<nbNeurons
                    {
                        var tmp: Double = _bArrays.w[depth]
                        if depth == DEPTH
                        {
                            if elem % 2 == 0
                            {
                                tmp += Ɛ
                            }
                            else
                            {
                                tmp -= Ɛ
                            }
                        }
                        
                        for depthPrev in 0..<nbNeuronsPrev
                        {
                            let w = _wArrays.w(depth, depthPrev)
                            let outPrev = neuronsPrev
                                .get(seq, depthPrev)!.v[batch].out
                            tmp += w * outPrev
                        }
                        
                        let offset = nbGC +
                            2 * nbNeurons * weightWidth +
                            2 * DEPTH + elem
                        neurons.get(seq, depth)!.gc[batch][offset].out = tmp
                    }
                }}
            }}}
        }
    }
    
    ///
    /// Apply the forward pass of the Gradient Checking in GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGCGPU() throws
    {
        try _forwardGCGPU()
        _activation?.forwardGC(self)
    }
    
    private func _forwardGCGPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let nbGC = layerPrev.nbGC
            let newGC = nbGC + 2 * nbLearnedGC
            
            for seq in 0..<sequence {
            for depth in 0..<nbNeurons
            {
                neurons.get(seq, depth)!.initGC(
                    batchSize: batchSize, nbGC: newGC
                )
            }}
            
            let weightsPtr = _wBuffers.w.download()
            let biasesPtr = _bBuffers.w.download()
            
            let neuronsPrev = layerPrev.neurons!
            let nbNeuronsPrev = layerPrev.nbNeurons
            
            for elem in 0..<nbGC {
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                for depth in 0..<nbNeurons
                {
                    var tmp: Double = Double(biasesPtr[depth])
                    for depthPrev in 0..<nbNeuronsPrev
                    {
                        let offsetWeights = depthPrev + weightWidth * depth
                        
                        let w = Double(weightsPtr[offsetWeights])
                        let outPrev = neuronsPrev
                            .get(seq, depthPrev)!.gc[batch][elem].out
                        tmp += w * outPrev
                    }
                    neurons.get(seq, depth)!.gc[batch][elem].out = tmp
                }
            }}}
            
            let outsPrevPtr = layerPrev.outs.download()
            
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                for DEPTH in 0..<nbNeurons {
                for DEPTHPREV in 0..<weightWidth {
                for elem in 0...1
                {
                    for depth in 0..<nbNeurons
                    {
                        var tmp: Double = Double(biasesPtr[depth])
                        for depthPrev in 0..<nbNeuronsPrev
                        {
                            let offsetWeights = depthPrev + weightWidth * depth
                            let offsetPrev = depthPrev + nbNeuronsPrev * seq +
                                sequence * nbNeuronsPrev * batch
                
                            var w = Double(weightsPtr[offsetWeights])
                            if depth == DEPTH && depthPrev == DEPTHPREV
                            {
                                if elem % 2 == 0
                                {
                                    w += Ɛ
                                }
                                else
                                {
                                    w -= Ɛ
                                }
                            }
                            
                            let outPrev = Double(outsPrevPtr[offsetPrev])
                            tmp += w * outPrev
                        }
                        
                        let offset = nbGC +
                            2 * (DEPTHPREV + weightWidth * DEPTH) + elem
                        neurons.get(seq, depth)!.gc[batch][offset].out = tmp
                    }
                }}}
            }}
            
            if _updateBiases {
            for batch in 0..<batchSize {
            for seq in 0..<sequence
            {
                for DEPTH in 0..<nbNeurons {
                for elem in 0...1
                {
                    for depth in 0..<nbNeurons
                    {
                        var tmp: Double = Double(biasesPtr[depth])
                        if depth == DEPTH
                        {
                            if elem % 2 == 0
                            {
                                tmp += Ɛ
                            }
                            else
                            {
                                tmp -= Ɛ
                            }
                        }
                        
                        for depthPrev in 0..<nbNeuronsPrev
                        {
                            let offsetWeights = depthPrev + weightWidth * depth
                            let offsetPrev = depthPrev + nbNeuronsPrev * seq +
                                sequence * nbNeuronsPrev * batch
                            
                            let w = Double(weightsPtr[offsetWeights])
                            let outPrev = Double(outsPrevPtr[offsetPrev])
                            tmp += w * outPrev
                        }
                        
                        let offset = nbGC +
                            2 * nbNeurons * weightWidth +
                            2 * DEPTH + elem
                        neurons.get(seq, depth)!.gc[batch][offset].out = tmp
                    }
                }}
            }}}
        }
    }
    
    ///
    /// Apply the forward pass in the CPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardCPU() throws
    {
        try _forwardCPU()
        _activation?.forwardCPU(self)
    }
    
    private func _forwardCPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateCPU(batchSize: batchSize)
            
            let neuronsPrev = layerPrev.neurons!
            let nbNeuronsPrev = layerPrev.nbNeurons
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                for depth in 0..<nbNeurons
                {
                    var tmp: Double = _bArrays.w[depth]
                    for depthPrev in 0..<nbNeuronsPrev
                    {
                        let w = _wArrays.w(depth, depthPrev)
                        let outPrev = neuronsPrev
                            .get(seq, depthPrev)!.v[elem].out
                        tmp += w * outPrev
                    }
                    
                    neurons.get(seq, depth)!.v[elem].out = tmp
                }
            }}
        }
    }
    
    ///
    /// Apply the forward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func forwardGPU() throws
    {
        try _forwardGPU()
        _activation?.forwardGPU(self)
    }
    
    private func _forwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq
        {
            try checkStateForwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(layerPrev.nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            
            let kernel: String
            let coeff: Int
            if layerPrev.nbNeurons % 4 == 0 && batchSize % 8 == 0
            {
                kernel = "flSeq48Forward"
                coeff = 8
            }
            else if layerPrev.nbNeurons % 4 == 0
            {
                kernel = "flSeq4Forward"
                coeff = 1
            }
            else
            {
                kernel = "flSeqForward"
                coeff = 1
            }
            
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(layerPrev.outs.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBuffer(_bBuffers.w.metal, atIndex: 2)
            command.setBytes(pNbNeurons, atIndex: 3)
            command.setBytes(pNbNeuronsPrev, atIndex: 4)
            command.setBytes(pNbBatch, atIndex: 5)
            command.setBytes(pSequence, atIndex: 6)
            command.setBuffer(outs.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: nbNeurons,
                height: (batchSize / coeff) * sequence
            )
            command.enqueue()
        }
    }
    
    /// Apply the backward pass in the CPU execution context.
    public override func backwardCPU()
    {
        _activation?.backwardCPU(self)
        
        _backwardCPU()
        _backwardWeightsCPU()
    }
    
    private func _backwardCPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            let nbNeuronsPrev = layerPrev.nbNeurons
            let neuronsPrev = layerPrev.neurons!
            
            for elem in 0..<batchSize {
            for seq in 0..<sequence
            {
                for depthPrev in 0..<nbNeuronsPrev
                {
                    var tmp: Double = 0.0
                    for depth in 0..<nbNeurons
                    {
                        let w = _wArrays.w(depth, depthPrev)
                        let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                        tmp += w * deltaCur
                    }
                    
                    if layerPrev.dirty
                    {
                        neuronsPrev.get(seq, depthPrev)!.v[elem].delta = tmp
                    }
                    else
                    {
                        neuronsPrev.get(seq, depthPrev)!.v[elem].delta += tmp
                    }
                }
            }}
            propagateDirty()
        }
    }
    
    private func _backwardWeightsCPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, computeDeltaWeights
        {
            let nbNeuronsPrev = layerPrev.nbNeurons
            let neuronsPrev = layerPrev.neurons!
            
            // -----------------------------------------------------------------
            // Compute Gradients per batch
            // -----------------------------------------------------------------
            for depth in 0..<nbNeurons {
            for depthPrev in 0..<nbNeuronsPrev
            {
                var tmp: Double = 0.0
                for elem in 0..<batchSize {
                for seq in 0..<sequence
                {
                    let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                    let outPrev = neuronsPrev.get(seq, depthPrev)!.v[elem].out
                    tmp += outPrev * deltaCur
                }}
                
                if accumulateDeltaWeights
                {
                    tmp += _wArrays.g(depth, depthPrev)
                }
                _wArrays.g(depth, depthPrev, tmp)
            }}
            
            if _updateBiases
            {
                for depth in 0..<nbNeurons
                {
                    var tmp: Double = 0.0
                    for elem in 0..<batchSize {
                    for seq in 0..<sequence
                    {
                        let deltaCur = neurons.get(seq, depth)!.v[elem].delta
                        tmp += deltaCur
                    }}
                    
                    if accumulateDeltaWeights
                    {
                        tmp += _bArrays.g[depth]
                    }
                    _bArrays.g[depth] = tmp
                }
            }
        }
    }
    
    ///
    /// Apply the backward pass in the GPU execution context.
    ///
    /// Throw an error if batch size is greater than the first batch size.
    ///
    public override func backwardGPU() throws
    {
        _activation?.backwardGPU(self)
        
        try _backwardGPU()
        _backwardWeightsGPU()
    }
    
    private func _backwardGPU() throws
    {
        if let layerPrev = self.layerPrev as? LayerSeq, mustComputeBackward
        {
            try layerPrev.checkStateBackwardGPU(batchSize: batchSize)
            
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(layerPrev.nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pDirty: [UInt32] = layerPrev.dirty ? [1] : [0]
            
            let kernel: String
            let coeff1: Int
            let coeff2: Int
            if layerPrev.nbNeurons % 4 == 0 && batchSize % 8 == 0
            {
                kernel = "flSeq48Backward"
                coeff1 = 4
                coeff2 = 8
            }
            else if layerPrev.nbNeurons % 4 == 0
            {
                kernel = "flSeq4Backward"
                coeff1 = 4
                coeff2 = 1
            }
            else
            {
                kernel = "flSeqBackward"
                coeff1 = 1
                coeff2 = 1
            }
            
            let command = MetalKernel.get.createCommand(
                kernel, deviceID: deviceID
            )
            command.setBuffer(delta.metal, atIndex: 0)
            command.setBuffer(_wBuffers.w.metal, atIndex: 1)
            command.setBytes(pNbNeurons, atIndex: 2)
            command.setBytes(pNbNeuronsPrev, atIndex: 3)
            command.setBytes(pNbBatch, atIndex: 4)
            command.setBytes(pSequence, atIndex: 5)
            command.setBytes(pDirty, atIndex: 6)
            command.setBuffer(layerPrev.delta.metal, atIndex: 7)
            
            command.dispatchThreads(
                width: weightWidth / coeff1,
                height: (batchSize / coeff2) * sequence
            )
            command.enqueue()
            
            propagateDirty()
        }
    }
    
    private func _backwardWeightsGPU()
    {
        if let layerPrev = self.layerPrev as? LayerSeq, computeDeltaWeights
        {
            let pNbNeurons: [UInt32] = [UInt32(nbNeurons)]
            let pNbNeuronsPrev: [UInt32] = [UInt32(layerPrev.nbNeurons)]
            let pNbBatch: [UInt32] = [UInt32(batchSize)]
            let pSequence: [UInt32] = [UInt32(sequence)]
            let pAccumulate: [UInt32] = accumulateDeltaWeights ? [1] : [0]
            
            var command: MetalCommand
            if GrAI.Gradient.batch
            {
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                let kernel = layerPrev.nbNeurons % 4 == 0 ?
                    "flSeqBatch4DerWeights" : "flSeqBatchDerWeights"
                let coeff = layerPrev.nbNeurons % 4 == 0 ? 4 : 1
                command = MetalKernel.get.createCommand(
                    kernel, deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(delta.metal, atIndex: 1)
                command.setBytes(pNbNeurons, atIndex: 2)
                command.setBytes(pNbNeuronsPrev, atIndex: 3)
                command.setBytes(pNbBatch, atIndex: 4)
                command.setBytes(pSequence, atIndex: 5)
                command.setBytes(pAccumulate, atIndex: 6)
                command.setBuffer(_wBuffers.g.metal, atIndex: 7)
                
                command.dispatchThreads(
                    width: nbNeurons,
                    height: weightWidth / coeff
                )
                command.enqueue()
                
                if _updateBiases
                {
                    let kernel = layerPrev.nbNeurons % 4 == 0 ?
                        "flPatchBatch4DerBiases" : "flPatchBatchDerBiases"
                    command = MetalKernel.get.createCommand(
                        kernel, deviceID: deviceID
                    )
                    command.setBuffer(delta.metal, atIndex: 0)
                    command.setBytes(pNbNeurons, atIndex: 1)
                    command.setBytes(pNbBatch, atIndex: 2)
                    command.setBytes(pSequence, atIndex: 3)
                    command.setBytes(pAccumulate, atIndex: 4)
                    command.setBuffer(_bBuffers.g.metal, atIndex: 5)
                    
                    command.dispatchThreads(nbNeurons / coeff)
                    command.enqueue()
                }
            }
            else
            {
                // -------------------------------------------------------------
                // Compute Gradients per sample
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "flSeqDerWeights", deviceID: deviceID
                )
                command.setBuffer(layerPrev.outs.metal, atIndex: 0)
                command.setBuffer(delta.metal, atIndex: 1)
                command.setBytes(pNbNeurons, atIndex: 2)
                command.setBytes(pNbNeuronsPrev, atIndex: 3)
                command.setBytes(pNbBatch, atIndex: 4)
                command.setBytes(pSequence, atIndex: 5)
                command.setBuffer(_wDeltaWeights.metal, atIndex: 6)
                
                command.dispatchThreads(
                    width: nbNeurons * batchSize,
                    height: weightWidth
                )
                command.enqueue()
                
                if _updateBiases
                {
                    command = MetalKernel.get.createCommand(
                        "flPatchDerBiases", deviceID: deviceID
                    )
                    command.setBuffer(delta.metal, atIndex: 0)
                    command.setBytes(pNbNeurons, atIndex: 1)
                    command.setBytes(pNbBatch, atIndex: 2)
                    command.setBytes(pSequence, atIndex: 3)
                    command.setBuffer(_bDeltaWeights.metal, atIndex: 4)
                    
                    command.dispatchThreads(
                        width: nbNeurons,
                        height: batchSize
                    )
                    command.enqueue()
                }
                
                // -------------------------------------------------------------
                // Compute Gradients per batch
                // -------------------------------------------------------------
                command = MetalKernel.get.createCommand(
                    "flSeqReduceWeights", deviceID: deviceID
                )
                command.setBuffer(_wDeltaWeights.metal, atIndex: 0)
                command.setBytes(pNbNeurons, atIndex: 1)
                command.setBytes(pNbNeuronsPrev, atIndex: 2)
                command.setBytes(pNbBatch, atIndex: 3)
                command.setBytes(pAccumulate, atIndex: 4)
                command.setBuffer(_wBuffers.g.metal, atIndex: 5)
                
                command.dispatchThreads(
                    width: nbNeurons,
                    height: weightWidth
                )
                command.enqueue()
                
                if _updateBiases
                {
                    command = MetalKernel.get.createCommand(
                        "reduceBiases", deviceID: deviceID
                    )
                    command.setBuffer(_bDeltaWeights.metal, atIndex: 0)
                    command.setBytes(pNbNeurons, atIndex: 1)
                    command.setBytes(pNbBatch, atIndex: 2)
                    command.setBytes(pAccumulate, atIndex: 3)
                    command.setBuffer(_bBuffers.g.metal, atIndex: 4)
                    
                    command.dispatchThreads(nbNeurons)
                    command.enqueue()
                }
            }
        }
    }
    
    /// Get the weights in the CPU execution context.
    public func collectWeightsCPU() -> [IWeightArrays]
    {
        var weights = [WeightArrays]()
        weights.append(_wArrays)
        if _updateBiases
        {
            weights.append(_bArrays)
        }
        return weights
    }
    
    /// Get the weights in the GPU execution context.
    public func collectWeightsGPU() -> [IWeightBuffers]
    {
        var weights = [IWeightBuffers]()
        weights.append(_wBuffers)
        if _updateBiases
        {
            weights.append(_bBuffers)
        }
        return weights
    }
    
    ///
    /// Get the weights' gradients in the GPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass or
    /// when gradients per sample have not been computed.
    ///
    /// - Parameter elem: The batch element to retrieve the outputs from.
    ///
    public func getDeltaWeightsGPU<T: BinaryFloatingPoint>(elem: Int) throws
        -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        if !GrAI.Gradient.sample
        {
            throw UpdateError.PerSample
        }
        
        var deltaWeights = [T]()
        var deltaWeightsPtr = _wDeltaWeights.download()
        
        let offsetStart = elem * nbNeurons * weightWidth
        for depth in 0..<nbNeurons {
        for depthPrev in 0..<weightWidth
        {
            let offset = offsetStart + depthPrev + weightWidth * depth
            
            deltaWeights.append(T(
                deltaWeightsPtr[offset]
            ))
        }}
        
        if _updateBiases
        {
            deltaWeightsPtr = _bDeltaWeights.download()
            
            for depth in 0..<nbNeurons
            {
                let offset = depth + nbNeurons * elem
                
                deltaWeights.append(T(
                    deltaWeightsPtr[offset]
                ))
            }
        }
        return deltaWeights
    }
    
    ///
    /// Get the weights' gradients in the CPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass.
    ///
    public func getDeltaWeightsCPU<T: BinaryFloatingPoint>() throws -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        
        var deltaWeights = [T]()
        for depth in 0..<nbNeurons {
        for depthPrev in 0..<weightWidth
        {
            deltaWeights.append(T(_wArrays.g(depth, depthPrev)))
        }}
        if _updateBiases
        {
            for depth in 0..<nbNeurons
            {
                deltaWeights.append(T(_bArrays.g[depth]))
            }
        }
        return deltaWeights
    }
    
    ///
    /// Get the weights' gradients in the GPU execution context.
    ///
    /// Throw an error when layer has not been updated through backward pass.
    ///
    public func getDeltaWeightsGPU<T: BinaryFloatingPoint>() throws -> [T]
    {
        if dirty
        {
            throw UpdateError.Dirty
        }
        
        var deltaWeights = [T]()
        var deltaWeightsPtr = _wBuffers.g.download()
        
        for i in 0..<_wBuffers.nbElems
        {
            deltaWeights.append(T(deltaWeightsPtr[i]))
        }
        if _updateBiases
        {
            deltaWeightsPtr = _bBuffers.g.download()
            
            for i in 0..<_bBuffers.nbElems
            {
                deltaWeights.append(T(deltaWeightsPtr[i]))
            }
        }
        return deltaWeights
    }
}
