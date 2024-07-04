import os
import torch
import torch.optim

from src.anti_spoof_predict import AntiSpoofPredict
from LLIE.Zero_DCE_plus_plus import model as Zero_DCE_model
from LLIE.SCI import model as SCI_model
from torch.autograd import Variable
def Convert_ONNX(torch_model, dummy_input, save_path):
     # Export the model
    torch.onnx.export(
        torch_model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        save_path,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
        dynamic_axes={
            "modelInput": {
                0: "batch_size",
                2: "height",
                3: "width",
            },  # variable length axes
            "modelOutput": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    print(" ")
    print("Model has been converted to ONNX")

def convert_miniFAS():
    model_1 = 'miniFAS/resources/train_SCI_miniFAS/2.7_80x80_MiniFASNetV2.pth'
    model_2 = 'miniFAS/resources/train_SCI_miniFAS/4_0_0_80x80_MiniFASNetV1SE.pth'
    model_save = ''

    for model_path in [model_1, model_2]:
        torch_model = AntiSpoofPredict(0)
        torch_model_loaded = torch_model._load_model(model_path)
        torch_model_loaded.eval()
        model_name = model_path.split("/")[-1].split(".pth")[0]
        model_name = model_name +'.onnx'
        dummy_input = torch.randn(1, 3, 80, 80, requires_grad=True)
        save_path = os.path.join(model_save, model_name)
        Convert_ONNX(torch_model_loaded, dummy_input, save_path)


def convert_Zero_DCE():
    model_path = ''
    model_save = ''
    scale_factor = 12
    torch_model = Zero_DCE_model.enhance_net_nopool(scale_factor)
    torch_model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    torch_model.eval()
    input_shape = (3, 400, 600)
    dummy_input = Variable(torch.randn(1, *input_shape))
    # dummy_input = torch.randn(1, 3, 5472, 3648, requires_grad=True)
    Convert_ONNX(torch_model, dummy_input, model_save)

def convert_SCI():
    model_path = 'miniFAS/LLIE/SCI/difficult.pt'
    model_save = '.onnx'
    torch_model = SCI_model.Finetunemodel(model_path)
    torch_model.eval()
    dummy_input = torch.randn(1, 3, 640, 480, requires_grad=True)
    with torch.no_grad():
        dummy_input = Variable(dummy_input, volatile=True)
    Convert_ONNX(torch_model, dummy_input, model_save)


if __name__ == '__main__':
    # convert_Zero_DCE()
    # convert_miniFAS()
    convert_SCI()