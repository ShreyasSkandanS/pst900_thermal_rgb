# =================================================================
# =========== Generates all PST900 dataset results ================
#       Written by Neil Rodrigues & Shreyas S. Shivakumar
# =================================================================

# Specifically for evaluation of FastSCNN
# Please follow the LICENSE and usage guideliness as mentioned
# in the FastSCNN repository:
# https://github.com/Tramac/Fast-SCNN-pytorch
from FastSCNN.eval import Evaluator

# For the rest of the models
from pst_inference import model_inference_loader

if __name__ == "__main__":
    # Generate contents for TABLE II

    # 1. ERFNet RGB
    MODEL_NAME = "erfnet_rgb"
    NUM_CHANNELS = 3
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 2. ERFNet RGB-T
    MODEL_NAME = "erfnet"
    NUM_CHANNELS = 4
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 3. MAVNet RGB
    MODEL_NAME = "mavnet_rgb"
    NUM_CHANNELS = 3
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)
 
    # 4. MAVNet RGB-T
    MODEL_NAME = "mavnet"
    NUM_CHANNELS = 4
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 5. UNet RGB
    MODEL_NAME = "original_unet_rgb"
    NUM_CHANNELS = 3
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 6. UNet RGB-T
    MODEL_NAME = "original_unet"
    NUM_CHANNELS = 4
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 7. FastSCNN RGB
    NUM_CHANNELS = 3
    model_inference_evaluator = Evaluator(NUM_CHANNELS)
    model_inference_evaluator.eval()

    # 8. FastSCNN RGB-T
    NUM_CHANNELS = 4
    model_inference_evaluator = Evaluator(NUM_CHANNELS)
    model_inference_evaluator.eval()
    
    # 9. MFNet 
    MODEL_NAME = "mfnet"
    NUM_CHANNELS = 4
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 10. RTFNet-50
    MODEL_NAME = "rtfnet_50"
    NUM_CHANNELS = 4
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 11. RTFNet-152
    MODEL_NAME = "rtfnet"
    NUM_CHANNELS = 4
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 12. PSTNet RGB
    MODEL_NAME = "pstnet"
    NUM_CHANNELS = 3
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)

    # 13. PSTNet RGB-T 
    MODEL_NAME = "pstnet_thermal"
    NUM_CHANNELS = 4
    model_inference_loader(MODEL_NAME, NUM_CHANNELS)


    
    



