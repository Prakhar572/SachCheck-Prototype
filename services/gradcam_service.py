"""Grad-CAM visual explanation for a MesoNet prediction."""
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.video_detector import image_to_tensor

def save_gradcam(face_path, output_path, model, device):
    """Overlay a heatmap where the CNN relied most heavily for its predicted class."""
    bgr = cv2.imread(str(face_path)); rgb = cv2.cvtColor(cv2.resize(bgr, (256,256)), cv2.COLOR_BGR2RGB).astype(np.float32)/255
    cam = GradCAM(model=model, target_layers=[model.features[-2]])
    mask = cam(input_tensor=image_to_tensor(bgr, device))[0]
    cv2.imwrite(str(output_path), cv2.cvtColor(show_cam_on_image(rgb, mask, use_rgb=True), cv2.COLOR_RGB2BGR))
    return str(output_path)
