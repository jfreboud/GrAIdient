//
// Biases.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void reduceBiasesHalf(
    const device half * deltaWeights,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pAccumulate,
    device half * grads,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    uint accumulate;
    
    if (pNbNeurons && pNbBatch && pAccumulate && deltaWeights && grads)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        accumulate = *pAccumulate;
    }
    else
        return ;
    
    uint depth = id;
    if (depth >= nbNeurons)
    {
        return ;
    }
    
    half tmp = 0.0;
    for (uint elem=0; elem<nbBatch; elem++)
    {
        uint offset = depth + elem * nbNeurons;
        tmp += deltaWeights[offset];
    }
    
    if (accumulate)
    {
        grads[depth] += tmp;
    }
    else
    {
        grads[depth] = tmp;
    }
}
