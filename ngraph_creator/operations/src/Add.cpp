#include <Add.hpp>
#undef LOG_TAG
#define LOG_TAG "Add"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Add::Add(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Add::validate() {
    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Add::createNode() {
    // Creating input nodes
    std::shared_ptr<ngraph::Node> input1, input2;
    std::shared_ptr<ngraph::Node> outputNode;

    input1 = getInputNode(0);
    input2 = getInputNode(1);

    if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) ||
        checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED) ||
        checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_SYMM) ||
        checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL)) {
        const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 0);
        const auto inputOp = sModelInfo->getOperand(inputIndex);
        if (inputOp.lifetime != OperandLifeTime::TEMPORARY_VARIABLE)
            input1 = std::make_shared<ngraph::opset3::Convert>(input1, ngraph::element::f32);
        input1 = addFakeQuantizeNode(input1, 0, 65536);
    }

    if (checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) ||
        checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED) ||
        checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_SYMM) ||
        checkInputOperandType(1, (int32_t)OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL)) {
        const auto& inputIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
        const auto inputOp = sModelInfo->getOperand(inputIndex);
        input2 = std::make_shared<ngraph::opset3::Convert>(input2, ngraph::element::f32);
        if (inputOp.lifetime == OperandLifeTime::TEMPORARY_VARIABLE ||
            inputOp.lifetime == OperandLifeTime::SUBGRAPH_INPUT)
            input2 = addFakeQuantizeNode(input2, 1, 65535);
        else
            input2 = DequantizeNode(input2, inputIndex, ngraph::element::f32);
    }
    const auto& activationIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
    if (sModelInfo->isOperandLifeTimeConst(activationIndex)) {
        auto activationFn = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex, 2);
        auto addNode = std::make_shared<ngraph::opset3::Add>(input1, input2,
                                                             ngraph::op::AutoBroadcastType::NUMPY);
        outputNode = applyActivation(addNode, activationFn);
    } else {
        auto addNode = std::make_shared<ngraph::opset3::Add>(input1, input2,
                                                             ngraph::op::AutoBroadcastType::NUMPY);
        outputNode = addNode;
    }

    auto outputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
    const auto op = sModelInfo->getOperand(outputIndex);
    if (op.lifetime == OperandLifeTime::SUBGRAPH_OUTPUT) {
        if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM) ||
            checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_SYMM)) {
            outputNode = QuantizeNode(outputNode, outputIndex, ngraph::element::u8);
        } else if (checkInputOperandType(0, (int32_t)OperandType::TENSOR_QUANT8_ASYMM_SIGNED)) {
            outputNode = QuantizeNode(outputNode, outputIndex, ngraph::element::i8);
        }
    }

    return outputNode;
}

std::shared_ptr<ngraph::Node> Add::createNodeForPlugin() {
    if (sPluginType == IntelDeviceType::VPU) {
        auto input = mNgraphNodes->getOperationOutput(
            sModelInfo->getOperationInput(mNnapiOperationIndex, 0));
        std::shared_ptr<ngraph::Node> constantOp =
            std::make_shared<ngraph::opset3::Constant>(ngraph::element::f32, input.get_shape());
        auto transposedOp = transpose(NHWC_NCHW, constantOp);
        return std::make_shared<ngraph::opset3::Add>(input, transposedOp,
                                                     ngraph::op::AutoBroadcastType::NUMPY);
    } else {
        return createNode();
    }
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
