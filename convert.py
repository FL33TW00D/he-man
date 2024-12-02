import torch
from torchvision.models import resnet18

def convert_to_onnx(model_path="resnet18.onnx"):
    # Load the model
    model = resnet18(weights='IMAGENET1K_V1')
    model.eval()
    
    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export the model to ONNX
    torch.onnx.export(
        model,                     # model being converted
        dummy_input,              # model input (or a tuple for multiple inputs)
        model_path,               # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=18,         # the ONNX version to export the model to
        do_constant_folding=True, # whether to execute constant folding for optimization
        input_names=['input'],    # the model's input names
        output_names=['output'],  # the model's output names
    )
    
    print(f"Model has been converted to ONNX and saved as {model_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

if __name__ == "__main__":
    convert_to_onnx()
