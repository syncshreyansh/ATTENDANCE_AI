import torch
import torch.onnx
import torchvision.models as models
import argparse

def convert_model(model_path, output_path):
    # Load your trained model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes: live, photo, screen
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model converted and saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Input PyTorch model')
    parser.add_argument('--output', required=True, help='Output ONNX path')
    args = parser.parse_args()
    
    convert_model(args.model, args.output)