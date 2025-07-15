import os
import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import rasterio
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

# Model Components
class FeatureDifferenceModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureDifferenceModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU()

    def forward(self, feat1, feat2):
        x = torch.abs(feat1 - feat2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DeconvDecoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeconvDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels // 2, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class ChangeFormer(nn.Module):
    def __init__(self, img_size=256, num_classes=1):
        super(ChangeFormer, self).__init__()
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
            in_chans=4,
        )
        self.feature_diff = FeatureDifferenceModule(in_channels=384)
        self.decoder = DeconvDecoder(in_channels=384, num_classes=num_classes)
        self.img_size = img_size
        self.patch_size = 16

    def forward(self, img1, img2):
        feat1 = self.encoder.forward_features(img1)
        feat2 = self.encoder.forward_features(img2)
        feat1 = feat1[:, 1:, :]
        feat2 = feat2[:, 1:, :]
        B, N, C = feat1.shape
        h = w = self.img_size // self.patch_size
        feat1 = feat1.transpose(1, 2).view(B, C, h, w)
        feat2 = feat2.transpose(1, 2).view(B, C, h, w)
        diff = self.feature_diff(feat1, feat2)
        out = self.decoder(diff)
        out = F.interpolate(out, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return out

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChangeFormer(num_classes=1).to(device)
print("ChangeFormer Model Initialized!")

# Load model weights
model_path = "/content/drive/MyDrive/DeforestationApp/models/best_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}.")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

PATCH_SIZE = 256
transform = transforms.ToTensor()

def read_patch_4band(path, x, y, size=PATCH_SIZE):
    with rasterio.open(path) as src:
        band_indices = [i for i in range(1, min(src.count, 4) + 1)]  # Bands 1–4
        patch = src.read(band_indices, window=rasterio.windows.Window(x, y, size, size))

        # Optional: cloud masking if band 8 (SCL) is present
        if src.count >= 8:
            scl = src.read(8, window=rasterio.windows.Window(x, y, size, size))
            cloud_mask = (scl == 3) | (scl == 8) | (scl == 9)
            patch[:, cloud_mask] = 0

        patch = np.transpose(patch, (1, 2, 0))
    return patch

def get_patch_coords(path, patch_size=PATCH_SIZE):
    with rasterio.open(path) as src:
        w, h = src.width, src.height
    coords = [(x, y) for y in range(0, h, patch_size)
              for x in range(0, w, patch_size)
              if x + patch_size <= w and y + patch_size <= h]
    return coords, (w, h)

def predict_on_large_4band_tifs(path1, path2):
    coords, full_size = get_patch_coords(path1)
    preds = []
    for i in range(0, len(coords), 4):  # Batch size of 4
        batch_coords = coords[i:i+4]
        batch_t1, batch_t2 = [], []
        for x, y in batch_coords:
            patch1 = read_patch_4band(path1, x, y)
            patch2 = read_patch_4band(path2, x, y)
            batch_t1.append(transform(patch1))
            batch_t2.append(transform(patch2))
        t1 = torch.stack(batch_t1).to(device)
        t2 = torch.stack(batch_t2).to(device)
        with torch.no_grad():
            pred = model(t1, t2)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            for p, (x, y) in zip(pred, batch_coords):
                pred_binary = (p > 0.5).astype(np.uint8)
                preds.append((pred_binary, (x, y)))
    return preds, full_size

def stitch_patches(preds, full_size, patch_size=PATCH_SIZE):
    stitched = np.zeros((full_size[1], full_size[0]), dtype=np.uint8)
    for patch, (x, y) in preds:
        stitched[y:y+patch_size, x:x+patch_size] = patch
    return stitched

def normalize_rgb(path):
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
        rgb = np.transpose(rgb, (1, 2, 0))
        mask = np.any(np.isnan(rgb), axis=-1) | np.all(rgb == 0, axis=-1)
        rgb[mask] = np.nan
        p2 = np.nanpercentile(rgb, 2)
        p98 = np.nanpercentile(rgb, 98)
        if p98 - p2 < 1e-5:
            rgb = np.clip(rgb / 255.0, 0, 1)
        else:
            rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
        rgb = np.nan_to_num(rgb)
    return rgb

def overlay_mask(rgb_img, mask, alpha=0.4):
    mask = mask.astype(np.float32)
    color_mask = np.zeros_like(rgb_img)
    color_mask[..., 0] = mask
    blended = (1 - alpha) * rgb_img + alpha * color_mask
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)

def generate_comment(mask):
    changed_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    percent = (changed_pixels / total_pixels) * 100
    if percent > 5:
        return f"Significant change detected: {percent:.2f}%"
    elif percent > 1:
        return f"Minor change detected: {percent:.2f}%"
    elif percent > 0:
        return f"Minimal change: {percent:.2f}%"
    else:
        return "No change detected."

def clear_outputs():
    return None, None, None, "Please upload new images to generate results."

def predict_change(file1, file2):
    try:
        path1, path2 = file1.name, file2.name
        with rasterio.open(path1) as src:
            if src.count < 4:
                raise ValueError("Input image must have at least 4 bands (RGB+NIR).")

        preds, full_size = predict_on_large_4band_tifs(path1, path2)
        mask = stitch_patches(preds, full_size)
        rgb = normalize_rgb(path2)
        overlay = overlay_mask(rgb, mask)
        return (
            Image.fromarray((rgb * 255).astype(np.uint8)),
            Image.fromarray(overlay),
            Image.fromarray((mask * 255).astype(np.uint8)),
            generate_comment(mask)
        )
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

# ==========================
# Gradio UI
# ==========================
with gr.Blocks() as demo:
    gr.Markdown("### UPLOAD INSTRUCTIONS:\n- **First Image** → OLDER image (earlier date)\n- **Second Image** → NEWER image (later date)\n\n> Both images must have **at least 4 bands (RGB + NIR)**.")
    
    with gr.Row():
        file1 = gr.File(label=" First Image (OLDER)", file_types=[".tif"])
        file2 = gr.File(label=" Second Image (NEWER)", file_types=[".tif"])
    with gr.Row():
        output1 = gr.Image(label="Raw Second Image RGB")
        output2 = gr.Image(label="Overlay with Prediction")
        output3 = gr.Image(label="Binary Change Mask")
    output4 = gr.Textbox(label="Auto-generated Comment")

    file1.upload(clear_outputs, None, [output1, output2, output3, output4])
    file2.upload(clear_outputs, None, [output1, output2, output3, output4])
    
    btn = gr.Button("Submit")
    btn.click(predict_change, inputs=[file1, file2], outputs=[output1, output2, output3, output4])

demo.launch()