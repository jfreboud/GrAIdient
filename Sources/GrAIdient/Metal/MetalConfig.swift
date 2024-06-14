//
// MetalConfig.swift
// GrAIdient
//
// Created by Jean-François Reboud on 18/05/2023.
//

let CONFIG_KERNELS =
[
    "ActivationFloat": [
        "forwardReLUFloat",
        "backwardReLUFloat",
        "forwardLeakyReLUFloat",
        "backwardLeakyReLUFloat",
        "forwardSoftReLUFloat",
        "backwardSoftReLUFloat",
        "forwardSigmoidFloat",
        "backwardSigmoidFloat",
        "forwardGELUApproxFloat",
        "backwardGELUApproxFloat",
        "forwardGELUFloat",
        "backwardGELUFloat",
    ],
    "ActivationHalf": [
        "forwardReLUHalf",
        "backwardReLUHalf",
        "forwardLeakyReLUHalf",
        "backwardLeakyReLUHalf",
        "forwardSoftReLUHalf",
        "backwardSoftReLUHalf",
        "forwardSigmoidHalf",
        "backwardSigmoidHalf",
        "forwardGELUApproxHalf",
        "backwardGELUApproxHalf",
        "forwardGELUHalf",
        "backwardGELUHalf",
    ],
    "BiasesFloat": [
        "reduceBiasesFloat",
    ],
    "BiasesHalf": [
        "reduceBiasesHalf",
    ],
    "BatchNormFloat": [
        "computeBNConvμFloat",
        "computeBNConvσ2Float",
        "forwardBNConvTrainingFloat",
        "forwardBNConvInferenceFloat",
        "backwardWeightsBNConvFloat",
        "backwardBNConvTrainingFloat",
        "backwardBNConvInferenceFloat",
    ],
    "BatchNormHalf": [
        "computeBNConvμHalf",
        "computeBNConvσ2Half",
        "forwardBNConvTrainingHalf",
        "forwardBNConvInferenceHalf",
        "backwardWeightsBNConvHalf",
        "backwardBNConvTrainingHalf",
        "backwardBNConvInferenceHalf",
    ],
    "ConvolutionFloat": [
        "convForwardFloat",
        "conv16ForwardFloat",
        "convBackwardFloat",
        "conv16BackwardFloat",
        "convBatchDerWeightsFloat",
        "conv34BatchDerWeightsFloat",
        "convBatchDerBiasesFloat",
        "convDerWeightsFloat",
        "convDerBiasesFloat",
        "convReduceWeightsFloat",
    ],
    "ConvolutionHalf": [
        "convForwardHalf",
        "conv16ForwardHalf",
        "convBackwardHalf",
        "conv16BackwardHalf",
        "convBatchDerWeightsHalf",
        "conv34BatchDerWeightsHalf",
        "convBatchDerBiasesHalf",
        "convDerWeightsHalf",
        "convDerBiasesHalf",
        "convReduceWeightsHalf",
    ],
    "DeconvolutionFloat": [
        "deconvForwardFloat",
        "deconvBackwardFloat",
        "deconvBatchDerWeightsFloat",
        "deconvDerWeightsFloat",
    ],
    "DeconvolutionHalf": [
        "deconvForwardHalf",
        "deconvBackwardHalf",
        "deconvBatchDerWeightsHalf",
        "deconvDerWeightsHalf",
    ],
    "EmbeddingSeqFloat": [
        "embeddingSeqForwardFloat",
        "embeddingSeqBatchDerWeightsFloat",
        "embeddingSeqDerWeightsFloat",
    ],
    "EmbeddingSeqHalf": [
        "embeddingSeqForwardHalf",
        "embeddingSeqBatchDerWeightsHalf",
        "embeddingSeqDerWeightsHalf",
    ],
    "FullyConnectedFloat": [
        "flForwardFloat",
        "flBackwardFloat",
        "flBatchDerWeightsFloat",
        "flBatchDerBiasesFloat",
        "flDerWeightsFloat",
        "flDerBiasesFloat",
        "flReduceWeightsFloat",
    ],
    "FullyConnectedHalf": [
        "flForwardHalf",
        "flBackwardHalf",
        "flBatchDerWeightsHalf",
        "flBatchDerBiasesHalf",
        "flDerWeightsHalf",
        "flDerBiasesHalf",
        "flReduceWeightsHalf",
    ],
    "FullyConnectedPatchFloat": [
        "flPatchForwardFloat",
        "flPatchBackwardFloat",
        "flPatchBatchDerWeightsFloat",
        "flPatchBatchDerBiasesFloat",
        "flPatchBatch4DerBiasesFloat",
        "flPatchDerWeightsFloat",
        "flPatchDerBiasesFloat",
        "flPatchReduceWeightsFloat",
    ],
    "FullyConnectedPatchHalf": [
        "flPatchForwardHalf",
        "flPatchBackwardHalf",
        "flPatchBatchDerWeightsHalf",
        "flPatchBatchDerBiasesHalf",
        "flPatchBatch4DerBiasesHalf",
        "flPatchDerWeightsHalf",
        "flPatchDerBiasesHalf",
        "flPatchReduceWeightsHalf",
    ],
    "FullyConnectedSeqFloat": [
        "flSeqForwardFloat",
        "flSeq48ForwardFloat",
        "flSeq4ForwardFloat",
        "flSeqBackwardFloat",
        "flSeq48BackwardFloat",
        "flSeq4BackwardFloat",
        "flSeqBatchDerWeightsFloat",
        "flSeqBatch4DerWeightsFloat",
        "flSeqDerWeightsFloat",
        "flSeqReduceWeightsFloat",
    ],
    "FullyConnectedSeqHalf": [
        "flSeqForwardHalf",
        "flSeq48ForwardHalf",
        "flSeq4ForwardHalf",
        "flSeqBackwardHalf",
        "flSeq48BackwardHalf",
        "flSeq4BackwardHalf",
        "flSeqBatchDerWeightsHalf",
        "flSeqBatch4DerWeightsHalf",
        "flSeqDerWeightsHalf",
        "flSeqReduceWeightsHalf",
    ],
    "InstanceNormFloat": [
        "computeInstanceNormConvμFloat",
        "computeInstanceNormConvσ2Float",
        "forwardInstanceNormConvFloat",
        "forwardAdaINFloat",
        "backwardWeightsInstanceNormConvFloat",
        "backward2AdaINFloat",
        "backwardInstanceNormConvFloat",
        "backward1AdaINFloat",
    ],
    "InstanceNormHalf": [
        "computeInstanceNormConvμHalf",
        "computeInstanceNormConvσ2Half",
        "forwardInstanceNormConvHalf",
        "forwardAdaINHalf",
        "backwardWeightsInstanceNormConvHalf",
        "backward2AdaINHalf",
        "backwardInstanceNormConvHalf",
        "backward1AdaINHalf",
    ],
    "Layer1DFloat": [
        "MSE1DLossFloat",
        "MSE1DLossDerivativeFloat",
        "linearErrorLossFloat",
        "linearErrorLossDerivativeFloat",
        "selectNeurons1DForwardFloat",
        "selectNeurons1DBackwardFloat",
        "concat1DForwardFloat",
        "concat1DBackwardFloat",
        "softmax1DForwardFloat",
        "softmax1DBackwardFloat",
        "dotProduct1DForwardFloat",
        "dotProduct1DBackwardFloat",
        "constant1DForwardFloat",
        "BCE1DLossFloat",
        "BCE1DLossDerivativeFloat",
        "BCESigmoid1DLossFloat",
        "BCESigmoid1DLossDerivativeFloat",
        "dropout1DForwardFloat",
        "dropout1DBackwardFloat",
    ],
    "Layer1DHalf": [
        "MSE1DLossHalf",
        "MSE1DLossDerivativeHalf",
        "linearErrorLossHalf",
        "linearErrorLossDerivativeHalf",
        "selectNeurons1DForwardHalf",
        "selectNeurons1DBackwardHalf",
        "concat1DForwardHalf",
        "concat1DBackwardHalf",
        "softmax1DForwardHalf",
        "softmax1DBackwardHalf",
        "dotProduct1DForwardHalf",
        "dotProduct1DBackwardHalf",
        "constant1DForwardHalf",
        "BCE1DLossHalf",
        "BCE1DLossDerivativeHalf",
        "BCESigmoid1DLossHalf",
        "BCESigmoid1DLossDerivativeHalf",
        "dropout1DForwardHalf",
        "dropout1DBackwardHalf",
    ],
    "Layer2DFloat": [
        "avgPoolForwardFloat",
        "avgPoolBackwardFloat",
        "maxPoolForwardFloat",
        "maxPoolBackwardFloat",
        "adaptiveAvgPoolForward1Float",
        "adaptiveAvgPoolForward2Float",
        "adaptiveAvgPoolBackward1Float",
        "adaptiveAvgPoolBackward2Float",
        "selectNeurons2DForwardFloat",
        "selectNeurons2DBackwardFloat",
        "IRDFT2RGBForwardFloat",
        "IRDFT2RGBBackwardFloat",
        "decorrelateRGBForwardFloat",
        "decorrelateRGBBackwardFloat",
        "linearScale2DForwardFloat",
        "linearScale2DBackwardFloat",
        "setDataFTFrequences2DFloat",
        "pad2DForwardFloat",
        "pad2DBackwardFloat",
        "crop2DForwardFloat",
        "crop2DBackwardFloat",
        "resizeBilinearPadForwardFloat",
        "resizeBilinearPadBackwardFloat",
        "rotate2DForwardFloat",
        "rotate2DBackwardFloat",
        "resizeBilinearCropForwardFloat",
        "resizeBilinearCropBackwardFloat",
        "concat02DForwardFloat",
        "concat02DBackwardFloat",
        "concat12DForwardFloat",
        "concat12DBackwardFloat",
        "constant2DForwardFloat",
        "MSE2DLossFloat",
        "MSE2DLossDerivativeFloat",
        "selfCorrelate2DForwardFloat",
        "selfCorrelate2DBackwardFloat",
        "normalize12DForwardFloat",
        "normalize12DBackwardFloat",
        "computeSquaredNorm122DFloat",
        "normalize122DForwardFloat",
        "computeDeltaTmp122DFloat",
        "normalize122DBackwardFloat",
        "similarBatchError2DLossFloat",
        "similarBatchError2DLossDerivativeFloat",
        "similarError2DLossDerivativeFloat",
        "flipHorizontal2DForwardFloat",
        "flipHorizontal2DBackwardFloat",
        "flipVertical2DForwardFloat",
        "flipVertical2DBackwardFloat",
        "colorJitterHSVForwardFloat",
        "BCE2DLossFloat",
        "BCE2DLossDerivativeFloat",
        "BCESigmoid2DLossFloat",
        "BCESigmoid2DLossDerivativeFloat",
        "layerCAM2DForwardFloat",
    ],
    "Layer2DHalf": [
        "avgPoolForwardHalf",
        "avgPoolBackwardHalf",
        "maxPoolForwardHalf",
        "maxPoolBackwardHalf",
        "adaptiveAvgPoolForward1Half",
        "adaptiveAvgPoolForward2Half",
        "adaptiveAvgPoolBackward1Half",
        "adaptiveAvgPoolBackward2Half",
        "selectNeurons2DForwardHalf",
        "selectNeurons2DBackwardHalf",
        "IRDFT2RGBForwardHalf",
        "IRDFT2RGBBackwardHalf",
        "decorrelateRGBForwardHalf",
        "decorrelateRGBBackwardHalf",
        "linearScale2DForwardHalf",
        "linearScale2DBackwardHalf",
        "setDataFTFrequences2DHalf",
        "pad2DForwardHalf",
        "pad2DBackwardHalf",
        "crop2DForwardHalf",
        "crop2DBackwardHalf",
        "resizeBilinearPadForwardHalf",
        "resizeBilinearPadBackwardHalf",
        "rotate2DForwardHalf",
        "rotate2DBackwardHalf",
        "resizeBilinearCropForwardHalf",
        "resizeBilinearCropBackwardHalf",
        "concat02DForwardHalf",
        "concat02DBackwardHalf",
        "concat12DForwardHalf",
        "concat12DBackwardHalf",
        "constant2DForwardHalf",
        "MSE2DLossHalf",
        "MSE2DLossDerivativeHalf",
        "selfCorrelate2DForwardHalf",
        "selfCorrelate2DBackwardHalf",
        "normalize12DForwardHalf",
        "normalize12DBackwardHalf",
        "computeSquaredNorm122DHalf",
        "normalize122DForwardHalf",
        "computeDeltaTmp122DHalf",
        "normalize122DBackwardHalf",
        "similarBatchError2DLossHalf",
        "similarBatchError2DLossDerivativeHalf",
        "similarError2DLossDerivativeHalf",
        "flipHorizontal2DForwardHalf",
        "flipHorizontal2DBackwardHalf",
        "flipVertical2DForwardHalf",
        "flipVertical2DBackwardHalf",
        "colorJitterHSVForwardHalf",
        "BCE2DLossHalf",
        "BCE2DLossDerivativeHalf",
        "BCESigmoid2DLossHalf",
        "BCESigmoid2DLossDerivativeHalf",
        "layerCAM2DForwardHalf",
    ],
    "LayerMergeFloat": [
        "sum1Float",
        "sum14Float",
        "sum2Float",
        "sum24Float",
        "multiplyForwardFloat",
        "multiplyBackwardFloat",
    ],
    "LayerMergeHalf": [
        "sum1Half",
        "sum14Half",
        "sum2Half",
        "sum24Half",
        "multiplyForwardHalf",
        "multiplyBackwardHalf",
    ],
    "LayerNormFloat": [
        "computeLayerNormSeqμFloat",
        "computeLayerNormSeqμ4Float",
        "computeLayerNormSeqσ2Float",
        "computeLayerNormSeqσ24Float",
        "forwardLayerNormSeqFloat",
        "forwardLayerNormSeq4Float",
        "backwardWeights1LayerNormSeqFloat",
        "backwardWeights1LayerNormSeq4Float",
        "backwardWeights2LayerNormSeqFloat",
        "backwardWeights2LayerNormSeq4Float",
        "backwardLayerNormSeqFloat",
        "backwardLayerNormSeq4Float",
    ],
    "LayerNormHalf": [
        "computeLayerNormSeqμHalf",
        "computeLayerNormSeqμ4Half",
        "computeLayerNormSeqσ2Half",
        "computeLayerNormSeqσ24Half",
        "forwardLayerNormSeqHalf",
        "forwardLayerNormSeq4Half",
        "backwardWeights1LayerNormSeqHalf",
        "backwardWeights1LayerNormSeq4Half",
        "backwardWeights2LayerNormSeqHalf",
        "backwardWeights2LayerNormSeq4Half",
        "backwardLayerNormSeqHalf",
        "backwardLayerNormSeq4Half",
    ],
    "LayerSeqFloat": [
        "avgPoolSeqForwardFloat",
        "avgPoolSeqBackwardFloat",
        "concat1SeqForwardFloat",
        "concat1Seq4ForwardFloat",
        "concat1SeqBackwardFloat",
        "concat1Seq4BackwardFloat",
        "concat2SeqForwardFloat",
        "concat2SeqBackwardFloat",
        "constant12SeqForwardFloat",
        "constant12Seq4ForwardFloat",
        "constant12SeqBackwardFloat",
        "constant12Seq4BackwardFloat",
        "constant2SeqForwardFloat",
        "constant2Seq4ForwardFloat",
        "querySeqForwardFloat",
        "querySeq4ForwardFloat",
        "queryQuerySeqBackwardFloat",
        "queryQuerySeq4BackwardFloat",
        "queryKeySeqBackwardFloat",
        "queryKeySeq4BackwardFloat",
        "querySelfSeqForwardFloat",
        "querySelfSeq4ForwardFloat",
        "querySelfQuerySeqBackwardFloat",
        "querySelfQuerySeq4BackwardFloat",
        "querySelfKeySeqBackwardFloat",
        "querySelfKeySeq4BackwardFloat",
        "softmaxSeqForwardFloat",
        "softmaxSeq4ForwardFloat",
        "softmaxSeqBackwardFloat",
        "softmaxSeq4BackwardFloat",
        "valueSeqForwardFloat",
        "valueSeq4ForwardFloat",
        "valueValueSeqBackwardFloat",
        "valueValueSeq4BackwardFloat",
        "valueScoreSeqBackwardFloat",
        "valueScoreSeq4BackwardFloat",
        "valueSelfSeqForwardFloat",
        "valueSelfSeq4ForwardFloat",
        "valueSelfValueSeqBackwardFloat",
        "valueSelfValueSeq4BackwardFloat",
        "valueSelfScoreSeqBackwardFloat",
        "valueSelfScoreSeq4BackwardFloat",
        "selectSeqForwardFloat",
        "selectSeqBackwardFloat",
        "layerCAMSeqForwardFloat",
    ],
    "LayerSeqHalf": [
        "avgPoolSeqForwardHalf",
        "avgPoolSeqBackwardHalf",
        "concat1SeqForwardHalf",
        "concat1Seq4ForwardHalf",
        "concat1SeqBackwardHalf",
        "concat1Seq4BackwardHalf",
        "concat2SeqForwardHalf",
        "concat2SeqBackwardHalf",
        "constant12SeqForwardHalf",
        "constant12Seq4ForwardHalf",
        "constant12SeqBackwardHalf",
        "constant12Seq4BackwardHalf",
        "constant2SeqForwardHalf",
        "constant2Seq4ForwardHalf",
        "querySeqForwardHalf",
        "querySeq4ForwardHalf",
        "queryQuerySeqBackwardHalf",
        "queryQuerySeq4BackwardHalf",
        "queryKeySeqBackwardHalf",
        "queryKeySeq4BackwardHalf",
        "querySelfSeqForwardHalf",
        "querySelfSeq4ForwardHalf",
        "querySelfQuerySeqBackwardHalf",
        "querySelfQuerySeq4BackwardHalf",
        "querySelfKeySeqBackwardHalf",
        "querySelfKeySeq4BackwardHalf",
        "softmaxSeqForwardHalf",
        "softmaxSeq4ForwardHalf",
        "softmaxSeqBackwardHalf",
        "softmaxSeq4BackwardHalf",
        "valueSeqForwardHalf",
        "valueSeq4ForwardHalf",
        "valueValueSeqBackwardHalf",
        "valueValueSeq4BackwardHalf",
        "valueScoreSeqBackwardHalf",
        "valueScoreSeq4BackwardHalf",
        "valueSelfSeqForwardHalf",
        "valueSelfSeq4ForwardHalf",
        "valueSelfValueSeqBackwardHalf",
        "valueSelfValueSeq4BackwardHalf",
        "valueSelfScoreSeqBackwardHalf",
        "valueSelfScoreSeq4BackwardHalf",
        "selectSeqForwardHalf",
        "selectSeqBackwardHalf",
        "layerCAMSeqForwardHalf",
    ],
    "OptimizerFloat": [
        "clipGradientsFloat",
        "multiplyGradientsFloat",
        "weightsSGDFloat",
        "weightsMomentumFloat",
        "weightsAdamFloat",
        "weightsAMSGradFloat",
        "weightsAdamRectifiedFloat",
        "weightsAdaBoundFloat",
        "weightsAMSBoundFloat",
    ],
    "OptimizerHalf": [
        "clipGradientsHalf",
        "multiplyGradientsHalf",
        "weightsSGDHalf",
        "weightsMomentumHalf",
        "weightsAdamHalf",
        "weightsAMSGradHalf",
        "weightsAdamRectifiedHalf",
        "weightsAdaBoundHalf",
        "weightsAMSBoundHalf",
    ],
    "ReduceFloat": [
        "reduceSum64Float",
        "reduceSumFloat",
        "reduceMax64Float",
        "reduceMaxFloat",
    ],
    "ReduceHalf": [
        "reduceSum64Half",
        "reduceSumHalf",
        "reduceMax64Half",
        "reduceMaxHalf",
    ],
    "ResetFloat": [
        "resetFloat",
    ],
    "ResetHalf": [
        "resetHalf",
        "convertFloat2Half",
        "convertHalf2Float",
    ],
    "VQ2DFloat": [
        "vq2DForwardFloat",
        "vq2DBackwardFloat",
        "vq2DBatchDerWeightsFloat",
        "vq2DDerWeightsFloat",
        "vq2DReduceWeightsFloat",
        "vq2DLossFloat",
        "vqLayerCAMMax2DFloat",
        "vqGrad2DForwardFloat",
    ],
    "VQ2DHalf": [
        "vq2DForwardHalf",
        "vq2DBackwardHalf",
        "vq2DBatchDerWeightsHalf",
        "vq2DDerWeightsHalf",
        "vq2DReduceWeightsHalf",
        "vq2DLossHalf",
        "vqLayerCAMMax2DHalf",
        "vqGrad2DForwardHalf",
    ],
    "VQSeqFloat": [
        "vqSeqForwardFloat",
        "vqSeqBackwardFloat",
        "vqSeqBatchDerWeightsFloat",
        "vqSeqDerWeightsFloat",
        "vqSeqLossFloat",
        "vqLayerCAMMaxSeqFloat",
        "vqGradSeqForwardFloat",
    ],
    "VQSeqHalf": [
        "vqSeqForwardHalf",
        "vqSeqBackwardHalf",
        "vqSeqBatchDerWeightsHalf",
        "vqSeqDerWeightsHalf",
        "vqSeqLossHalf",
        "vqLayerCAMMaxSeqHalf",
        "vqGradSeqForwardHalf",
    ],
]
