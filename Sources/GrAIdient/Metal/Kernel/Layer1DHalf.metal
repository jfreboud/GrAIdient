//
// Layer1D.metal
// GrAIdient
//
// Created by Jean-François Reboud on 14/10/2022.
//

#include <metal_stdlib>
using namespace metal;

kernel void MSE1DLossHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device half * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && outs && groundTruth && losses)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id;
    if (elem >= nbBatch)
    {
        return ;
    }
    
    half tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * elem;
    
        half gt = groundTruth[offset];
        half out = outs[offset];
        half diff = out - gt;
        
        tmp += diff * diff;
    }
    
    losses[elem] = tmp;
}

kernel void MSE1DLossDerivativeHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbNeurons,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    half coeff;
    uint nbBatch;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pCoeff && pDirty &&
        outs && groundTruth && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;

    half gt = groundTruth[offset];
    half out = outs[offset];
    half diff = out - gt;
    
    if (dirty)
    {
        deltaPrev[offset] = 2 * coeff * diff / half(nbNeurons * nbBatch);
    }
    else
    {
        deltaPrev[offset] += 2 * coeff * diff / half(nbNeurons * nbBatch);
    }
}

kernel void linearErrorLossHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device half * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && outs && groundTruth && losses)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id;
    if (elem >= nbBatch)
    {
        return ;
    }
    
    half tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * elem;
    
        half gt = groundTruth[offset];
        half out = outs[offset];
        half diff = out - gt;
        
        tmp += diff;
    }
    
    losses[elem] = tmp;
}

kernel void linearErrorLossDerivativeHalf(
    const device half * outs,
    constant uint * pNbNeurons,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    half coeff;
    uint nbBatch;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pCoeff && pDirty && outs && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    
    if (dirty)
    {
        deltaPrev[offset] = coeff / half(nbNeurons * nbBatch);
    }
    else
    {
        deltaPrev[offset] += coeff / half(nbNeurons * nbBatch);
    }
}

kernel void selectNeurons1DForwardHalf(
    const device half * outsPrev,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNeurons,
    constant float * pCoeffs,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    
    if (pNbNeurons && pNbNeuronsPrev && pNeurons && pCoeffs && pNbBatch &&
        outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    uint offsetPrev = pNeurons[depth] + nbNeuronsPrev * elem;
    outs[offset] = pCoeffs[depth] * outsPrev[offsetPrev];
}

kernel void selectNeurons1DBackwardHalf(
    const device half * delta,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNeurons,
    constant float * pCoeffs,
    constant uint * pNbBatch,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    
    if (pNbNeurons && pNbNeuronsPrev && pNeurons && pCoeffs && pNbBatch &&
        deltaPrev && delta)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    uint offsetPrev = pNeurons[depth] + nbNeuronsPrev * elem;
    deltaPrev[offsetPrev] += pCoeffs[depth] * delta[offset];
}

kernel void concat1DForwardHalf(
    const device half * outsPrev,
    constant uint * pGlobalOffset,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint globalOffset;
    
    if (pGlobalOffset && pNbNeurons && pNbNeuronsPrev && pNbBatch &&
        outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        globalOffset = *pGlobalOffset;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeuronsPrev || elem >= nbBatch)
    {
        return ;
    }
    
    uint offsetPrev = depth + nbNeuronsPrev * elem;
    uint offset = globalOffset+depth + nbNeurons * elem;
    
    outs[offset] = outsPrev[offsetPrev];
}

kernel void concat1DBackwardHalf(
    const device half * delta,
    constant uint * pGlobalOffset,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint globalOffset;
    uint dirty;
    
    if (pGlobalOffset && pNbNeurons && pNbNeuronsPrev && pNbBatch && pDirty &&
        deltaPrev && delta)
    {
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        globalOffset = *pGlobalOffset;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeuronsPrev || elem >= nbBatch)
    {
        return ;
    }
    
    uint offsetPrev = depth + nbNeuronsPrev * elem;
    uint offset = globalOffset+depth + nbNeurons * elem;
    
    if (dirty)
    {
        deltaPrev[offsetPrev] = delta[offset];
    }
    else
    {
        deltaPrev[offsetPrev] += delta[offset];
    }
}

kernel void softmax1DForwardHalf(
    const device half * outsPrev,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint size;
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbHeads && pNbNeurons && pNbBatch && outsPrev && outs)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        size = nbNeurons / nbHeads;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    uint head = depth / size;
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    half cMax = outsPrev[0+head*size + nbNeurons * elem];
    for (uint j=0; j<size; j++)
    {
        uint offset1 = j+head*size + nbNeurons * elem;
        half outPrev = outsPrev[offset1];
        
        if (outPrev > cMax)
        {
            cMax = outPrev;
        }
    }
    
    half sum1 = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint offset1 = j+head*size + nbNeurons * elem;
        half outPrev = outsPrev[offset1];
        sum1 += exp(outPrev - cMax);
    }
    
    uint offset = depth + nbNeurons * elem;
    half outPrev = outsPrev[offset];
    outs[offset] = exp(outPrev - cMax) / sum1;
}

kernel void softmax1DBackwardHalf(
    const device half * outs,
    const device half * delta,
    constant uint * pNbHeads,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbHeads;
    uint size;
    uint nbNeurons;
    uint nbBatch;
    uint dirty;
    
    if (pNbHeads && pNbNeurons && pNbBatch && pDirty &&
        deltaPrev && outs && delta)
    {
        nbHeads = *pNbHeads;
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
        size = nbNeurons / nbHeads;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    uint head = depth / size;
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    half outCur = outs[offset];
    half deltaCur = delta[offset];
    
    half sum1 = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint offset1 = j+head*size + nbNeurons * elem;
        half outCur1 = outs[offset1];
        half deltaCur1 = delta[offset1];
        sum1 += outCur1 * deltaCur1;
    }
    
    if (dirty)
    {
        deltaPrev[offset] = outCur * (deltaCur - sum1);
    }
    else
    {
        deltaPrev[offset] += outCur * (deltaCur - sum1);
    }
}

kernel void dotProduct1DForwardHalf(
    const device half * outsPrev1,
    const device half * outsPrev2,
    constant int * pSize,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    
    if (pSize && pNbNeurons && pNbNeuronsPrev && pNbBatch &&
        outsPrev1 && outsPrev2 && outs)
    {
        size = *pSize;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    half sum = 0.0;
    for (uint j=0; j<size; j++)
    {
        uint offset = j+depth*size + nbNeuronsPrev * elem;
        half outPrev1 = outsPrev1[offset];
        half outPrev2 = outsPrev2[offset];
        sum += outPrev1 * outPrev2;
    }
    
    uint offset = depth + nbNeurons * elem;
    outs[offset] = sum;
}

kernel void dotProduct1DBackwardHalf(
    const device half * outsPrev,
    const device half * delta,
    constant int * pSize,
    constant uint * pNbNeurons,
    constant uint * pNbNeuronsPrev,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint size;
    uint nbNeurons;
    uint nbNeuronsPrev;
    uint nbBatch;
    uint dirty;
    
    if (pSize && pNbNeurons && pNbNeuronsPrev && pNbBatch && pDirty &&
        outsPrev && deltaPrev && delta)
    {
        size = *pSize;
        nbNeurons = *pNbNeurons;
        nbNeuronsPrev = *pNbNeuronsPrev;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    for (uint j=0; j<size; j++)
    {
        uint offsetPrev = j+depth*size + nbNeuronsPrev * elem;
        uint offset = depth + nbNeurons * elem;
        
        half outPrev = outsPrev[offsetPrev];
        half deltaCur = delta[offset];
        if (dirty)
        {
            deltaPrev[offsetPrev] = outPrev * deltaCur;
        }
        else
        {
            deltaPrev[offsetPrev] += outPrev * deltaCur;
        }
    }
}

kernel void constant1DForwardHalf(
    const device half * weights,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && weights && outs)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    outs[offset] = weights[depth];
}

kernel void BCE1DLossHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device half * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && outs && groundTruth && losses)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id;
    if (elem >= nbBatch)
    {
        return ;
    }
    
    half tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * elem;
    
        half gt = groundTruth[offset];
        half out = outs[offset];
        half tmp1 = log(out);
        half tmp2 = log(1 - out);
        
        tmp -= (gt * tmp1 + (1 - gt) * tmp2);
    }
    
    losses[elem] = tmp;
}

kernel void BCE1DLossDerivativeHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbNeurons,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    half coeff;
    uint nbBatch;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pCoeff && pDirty &&
        outs && groundTruth && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;

    half gt = groundTruth[offset];
    half out = outs[offset];
    half derivative = 0.0;
    
    if (gt == 1.0)
    {
        derivative = -1 / out;
    }
    else if (gt == 0.0)
    {
        derivative = 1 / (1 - out);
    }
    
    if (dirty)
    {
        deltaPrev[offset] = coeff * derivative / half(nbNeurons * nbBatch);
    }
    else
    {
        deltaPrev[offset] += coeff * derivative / half(nbNeurons * nbBatch);
    }
}

kernel void BCESigmoid1DLossHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    device half * losses,
    uint id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    
    if (pNbNeurons && pNbBatch && outs && groundTruth && losses)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
    }
    else
        return ;
    
    uint elem = id;
    if (elem >= nbBatch)
    {
        return ;
    }
    
    half tmp = 0.0;
    for (uint depth=0; depth<nbNeurons; depth++)
    {
        uint offset = depth + nbNeurons * elem;
    
        half gt = groundTruth[offset];
        half out = outs[offset];
        half value;
        
        if (out > 0)
        {
            value = (1 - gt) * out;
            value += log(1 + exp(-out));
        }
        else
        {
            value = -out * gt;
            value += log(exp(out) + 1);
        }
        
        tmp += value;
    }
    
    losses[elem] = tmp;
}

kernel void BCESigmoid1DLossDerivativeHalf(
    const device half * outs,
    const device half * groundTruth,
    constant uint * pNbNeurons,
    constant float * pCoeff,
    constant uint * pNbBatch,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    half coeff;
    uint nbBatch;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pCoeff && pDirty &&
        outs && groundTruth && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        coeff = *pCoeff;
        nbBatch = *pNbBatch;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;

    half gt = groundTruth[offset];
    half out = outs[offset];
    half value;
    
    if (out >= 0)
    {
        value = 1.0 / (1.0 + exp(-out));
    }
    else
    {
        value = exp(out) / (1.0 + exp(out));
    }
    
    if (dirty)
    {
        deltaPrev[offset] = coeff * (value - gt) / half(nbNeurons * nbBatch);
    }
    else
    {
        deltaPrev[offset] += coeff * (value - gt) / half(nbNeurons * nbBatch);
    }
}

kernel void dropout1DForwardHalf(
    const device half * outsPrev,
    const device bool * dropout,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant bool * pApplyDropout,
    constant float * pCoeff,
    device half * outs,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    bool applyDropout;
    half coeff;
    
    if (pNbNeurons && pNbBatch && pApplyDropout && pCoeff &&
        dropout && outsPrev && outs)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        applyDropout = *pApplyDropout;
        coeff = *pCoeff;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    uint offset = depth + nbNeurons * elem;
    if (applyDropout && !dropout[offset])
    {
        outs[offset] = 1.0 / (1.0 - coeff) * outsPrev[offset];
    }
    else if (applyDropout)
    {
        outs[offset] = 0.0;
    }
    else
    {
        outs[offset] = outsPrev[offset];
    }
}

kernel void dropout1DBackwardHalf(
    const device half * delta,
    const device bool * dropout,
    constant uint * pNbNeurons,
    constant uint * pNbBatch,
    constant bool * pApplyDropout,
    constant float * pCoeff,
    constant uint * pDirty,
    device half * deltaPrev,
    uint2 id [[ thread_position_in_grid ]])
{
    uint nbNeurons;
    uint nbBatch;
    bool applyDropout;
    half coeff;
    uint dirty;
    
    if (pNbNeurons && pNbBatch && pApplyDropout && pCoeff &&
        dropout && delta && deltaPrev)
    {
        nbNeurons = *pNbNeurons;
        nbBatch = *pNbBatch;
        applyDropout = *pApplyDropout;
        coeff = *pCoeff;
        dirty = *pDirty;
    }
    else
        return ;
    
    uint depth = id[0];
    uint elem = id[1];
    
    if (depth >= nbNeurons || elem >= nbBatch)
    {
        return ;
    }
    
    half newValue = 0.0;
    uint offset = depth + nbNeurons * elem;
    if (applyDropout && !dropout[offset])
    {
        newValue = 1.0 / (1.0 - coeff) * delta[offset];
    }
    else if (applyDropout)
    {
        newValue = 0.0;
    }
    else
    {
        newValue = delta[offset];
    }
    
    if (dirty)
    {
        deltaPrev[offset] = newValue;
    }
    else
    {
        deltaPrev[offset] += newValue;
    }
}
