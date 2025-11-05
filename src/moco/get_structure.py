import torch
from collections import OrderedDict

def inspect_pt_file(file_path):
    try:
        # 安全加载模型文件（避免设备不匹配）
        data = torch.load(file_path, map_location=torch.device('cpu'))
        
        
        # 方法2：解析状态字典结构
        if isinstance(data, (dict, OrderedDict)):
            print("\n[State Dictionary Structure]:")
            for key, value in data.items():
                param_type = "Tensor" if torch.is_tensor(value) else type(value).__name__
                shape = tuple(value.shape) if torch.is_tensor(value) else "N/A"
                print(f"│-- {key}: {param_type} {shape}")
                
        elif hasattr(data, 'state_dict'):
            print("\n[Model Architecture]:")
            for name, param in data.state_dict().items():
                print(f"│-- {name}: {tuple(param.shape)}")

            
    except Exception as e:
        print(f"Error loading file: {str(e)}")

# 使用示例
inspect_pt_file("./outputs/WN18RR/2025-03-29_13-19-24/best.pt")
