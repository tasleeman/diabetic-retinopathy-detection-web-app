import torch
import torch.nn.functional as F
from data_loader import preprocess_image  # Changed import
from typing import Dict

def predict_dr(model, image) -> Dict[str, float]:
    """Simplified prediction function"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        input_tensor = preprocess_image(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1).squeeze().tolist()
        
        return {
            'No DR': probabilities[0],
            'Mild DR': probabilities[1],
            'Moderate DR': probabilities[2],
            'Severe DR': probabilities[3],
            'Proliferative DR': probabilities[4]
        }
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {
            'No DR': 0.0,
            'Mild DR': 0.0,
            'Moderate DR': 0.0,
            'Severe DR': 0.0,
            'Proliferative DR': 0.0
        }