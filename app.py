import os
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
import numpy as np
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
import folium
from folium.plugins import HeatMap
import logging
import time
import uuid
import urllib.request
import urllib.error

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Model Definition
# ---------------------------
class FeatureDifferenceModule(nn.Module):
    """Computes feature differences using absolute difference and Conv2D."""
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
    """Deconvolution-based decoder for better spatial reconstruction."""
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
    """Transformer-based Change Detection Model with shared encoder."""
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
logger.info(f"Using device: {device}")
try:
    model_path = "/content/drive/MyDrive/App/models/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

PATCH_SIZE = 256
DEFAULT_THRESHOLD = 0.5
DEFAULT_ALPHA = 0.4
transform = transforms.ToTensor()

# ---------------------------
# Utility Functions
# ---------------------------
def clear_cache():
    """Clear Gradio cache and temporary files to prevent stale data."""
    try:
        if hasattr(gr, 'clear_cache'):
            gr.clear_cache()
            logger.info("Gradio cache cleared.")
        else:
            logger.warning("Gradio cache clearing not supported in this version.")
    except Exception as e:
        logger.error(f"Error clearing Gradio cache: {str(e)}")
    for temp_file in ["overlay.png", "mask.png", "mask_geotiff.tif", "change_polygons.geojson"]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_file}: {str(e)}")

def generate_unique_filename(base_name, extension):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# ---------------------------
# Visualization Functions
# ---------------------------
def add_overlay_legend(overlay_img, original_shape):
    try:
        overlay_pil = Image.fromarray(overlay_img)
        if overlay_pil.size != (original_shape[1], original_shape[0]):
            logger.warning(f"Overlay size {overlay_pil.size} does not match original {original_shape}")
        dpi = 100
        legend_width = 250
        fig_width = (overlay_pil.size[0] + legend_width) / dpi
        fig_height = overlay_pil.size[1] / dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        ax.imshow(overlay_img)
        ax.axis("off")
        red_patch = mpatches.Patch(color='red', alpha=0.5, label='Change Pixels')
        gray_patch = mpatches.Patch(color='lightgray', alpha=0.5, label='Unchanged')
        ax.legend(handles=[red_patch, gray_patch], loc="center left",
                 bbox_to_anchor=(1.0, 0.5), fontsize=12, frameon=True,
                 framealpha=1.0, facecolor='white', edgecolor='black')
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        result_img = Image.open(buf)
        logger.info(f"Overlay with legend size: {result_img.size} (original content: {overlay_pil.size})")
        return result_img
    except Exception as e:
        logger.error(f"Failed to add overlay legend: {str(e)}")
        raise

def add_mask_legend(mask, original_shape):
    try:
        mask_pil = Image.fromarray(mask, mode='L')
        if mask_pil.size != (original_shape[1], original_shape[0]):
            logger.warning(f"Mask size {mask_pil.size} does not match original {original_shape}")
        dpi = 100
        legend_width = 250
        fig_width = (mask_pil.size[0] + legend_width) / dpi
        fig_height = mask_pil.size[1] / dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        cmap = plt.cm.get_cmap("gray", 2)
        ax.imshow(mask, cmap=cmap)
        ax.axis("off")
        black_patch = mpatches.Patch(color='black', label='No Change')
        white_patch = mpatches.Patch(color='white', label='Change Pixels')
        ax.legend(handles=[black_patch, white_patch], loc="center left",
                 bbox_to_anchor=(1.0, 0.5), fontsize=12, frameon=True,
                 framealpha=1.0, facecolor='white', edgecolor='black')
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        result_img = Image.open(buf)
        logger.info(f"Mask with legend size: {result_img.size} (original content: {mask_pil.size})")
        return result_img
    except Exception as e:
        logger.error(f"Failed to add mask legend: {str(e)}")
        raise

def add_heatmap_legend(mask):
    try:
        dpi = 100
        fig, ax = plt.subplots(figsize=(mask.shape[1] / dpi, mask.shape[0] / dpi), dpi=dpi)
        im = ax.imshow(mask, cmap="Reds", interpolation="nearest")
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
        cbar.set_label("Change Intensity", fontsize=12)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    except Exception as e:
        logger.error(f"Heatmap creation failed: {str(e)}")
        raise

def read_patch_4band(path, x, y, size=PATCH_SIZE):
    try:
        with rasterio.open(path) as src:
            patch = src.read([1, 2, 3, 4], window=rasterio.windows.Window(x, y, size, size))
            patch = np.transpose(patch, (1, 2, 0))
        return patch
    except Exception as e:
        logger.error(f"Failed to read patch from {path}: {str(e)}")
        raise

def get_patch_coords(path, patch_size=PATCH_SIZE):
    try:
        with rasterio.open(path) as src:
            w, h = src.width, src.height
        coords = [(x, y) for y in range(0, h, patch_size)
                  for x in range(0, w, patch_size)
                  if x + patch_size <= w and y + patch_size <= h]
        logger.info(f"Found {len(coords)} patches for {path}")
        return coords, (w, h)
    except Exception as e:
        logger.error(f"Failed to get patch coordinates for {path}: {str(e)}")
        raise

def predict_on_large_4band_tifs(path1, path2, threshold=DEFAULT_THRESHOLD):
    logger.info(f"Predicting on {path1} and {path2}")
    try:
        coords, full_size = get_patch_coords(path1)
        preds = []
        for i, (x, y) in enumerate(coords):
            patch1 = read_patch_4band(path1, x, y)
            patch2 = read_patch_4band(path2, x, y)
            t1 = transform(patch1).unsqueeze(0).to(device)
            t2 = transform(patch2).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(t1, t2)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
                pred_binary = (pred > threshold).astype(np.uint8)
            preds.append((pred_binary, (x, y)))
        return preds, full_size
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def stitch_patches(preds, full_size, patch_size=PATCH_SIZE):
    try:
        stitched = np.zeros((full_size[1], full_size[0]), dtype=np.uint8)
        for patch, (x, y) in preds:
            stitched[y:y+patch_size, x:x+patch_size] = patch
        logger.info(f"Stitched mask shape: {stitched.shape}")
        return stitched
    except Exception as e:
        logger.error(f"Stitching failed: {str(e)}")
        raise

def normalize_rgb(path):
    try:
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
    except Exception as e:
        logger.error(f"RGB normalization failed for {path}: {str(e)}")
        raise

def overlay_mask(rgb_img, mask, alpha=DEFAULT_ALPHA):
    try:
        if rgb_img.shape[:2] != mask.shape:
            raise ValueError(f"RGB image shape {rgb_img.shape[:2]} does not match mask shape {mask.shape}")
        mask = mask.astype(np.float32)
        color_mask = np.zeros_like(rgb_img)
        color_mask[..., 0] = mask
        blended = (1 - alpha) * rgb_img + alpha * color_mask
        blended = np.clip(blended, 0, 1)
        return (blended * 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Overlay creation failed: {str(e)}")
        raise

def generate_comment(mask):
    try:
        changed_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        percent = (changed_pixels / total_pixels) * 100
        logger.info(f"Change percentage: {percent:.2f}%")
        if percent > 10:
            return (
                f"Extensive changes detected: {percent:.2f}% of the area shows alteration. "
                "This suggests major land cover transformation, such as widespread deforestation, "
                "large-scale urban growth, or significant flooding/erosion. "
                "The scale of change may have ecological and socio-economic implications and warrants immediate attention."
            )
        elif percent > 1:
            return (
                f"Moderate changes detected: {percent:.2f}% of the area has shifted. "
                "These could correspond to expanding settlements, agricultural conversion, "
                "or seasonal environmental shifts. "
                "While not extreme, these changes may indicate emerging trends that should be monitored."
            )
        elif percent > 0:
            return (
                f"Localized minor changes detected: {percent:.2f}% of the area affected. "
                "These may result from small construction projects, localized vegetation clearing, "
                "or natural disturbances. "
                "Although small in scale, such changes can accumulate over time."
            )
        else:
            return (
                "No measurable change detected between the two time periods. "
                "The landscape appears stable, with no significant disturbances or developments recorded."
            )
    except Exception as e:
        logger.error(f"Comment generation failed: {str(e)}")
        raise

def compute_stats(mask, path2):
    try:
        changed_pixels = int(np.count_nonzero(mask))
        total_pixels = int(mask.size)
        unchanged = total_pixels - changed_pixels
        percent = (changed_pixels / total_pixels) * 100
        with rasterio.open(path2) as src:
            try:
                resolution = abs(src.res[0]) * abs(src.res[1])
            except Exception:
                resolution = 0
            changed_area = changed_pixels * resolution if resolution > 0 else "N/A (No georeferencing)"
            bounds = src.bounds
            crs = src.crs.to_string() if src.crs else "N/A (No CRS)"
        return {
            "Changed Pixels": changed_pixels,
            "Unchanged Pixels": unchanged,
            "Change %": round(percent, 2),
            "Estimated Changed Area (m²)": changed_area,
            "Bounding Box": {
                "Left": bounds.left,
                "Bottom": bounds.bottom,
                "Right": bounds.right,
                "Top": bounds.top
            },
            "Coordinate Reference System": crs
        }
    except Exception as e:
        logger.error(f"Stats computation failed: {str(e)}")
        raise

def plot_stats(stats):
    try:
        labels = ["Changed", "Unchanged"]
        values = [stats["Changed Pixels"], stats["Unchanged Pixels"]]
        fig, ax = plt.subplots()
        ax.bar(labels, values, color=['#ff4d4d', '#4da8ff'])
        ax.set_ylabel("Pixels")
        ax.set_title("Change Statistics")
        return fig
    except Exception as e:
        logger.error(f"Stats plotting failed: {str(e)}")
        raise

def create_interactive_heatmap(mask, path2):
    try:
        with rasterio.open(path2) as src:
            bounds = src.bounds
            transform_affine = src.transform
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            m = folium.Map(location=[(bounds.top + bounds.bottom) / 2,
                                     (bounds.left + bounds.right) / 2],
                           zoom_start=12, tiles="cartodbpositron")
            return m._repr_html_()
        pts = []
        for r, c in zip(rows, cols):
            x, y = rasterio.transform.xy(transform_affine, r, c)
            pts.append([y, x])
        m = folium.Map(location=[(bounds.top + bounds.bottom) / 2,
                                 (bounds.left + bounds.right) / 2],
                       zoom_start=12, tiles="cartodbpositron")
        HeatMap(pts, radius=6, blur=4, min_opacity=0.3).add_to(m)
        folium.LayerControl().add_to(m)
        return m._repr_html_()
    except Exception as e:
        logger.error(f"Interactive heatmap creation failed: {str(e)}")
        raise

def export_geotiff(mask, path2, output_path=None):
    try:
        output_path = output_path or generate_unique_filename("mask_geotiff", "tif")
        with rasterio.open(path2) as src:
            profile = src.profile
            profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(mask.astype(rasterio.uint8), 1)
        return output_path
    except Exception as e:
        logger.error(f"GeoTIFF export failed: {str(e)}")
        raise

def export_geojson(mask, reference_path, out_path=None):
    try:
        out_path = out_path or generate_unique_filename("change_polygons", "geojson")
        with rasterio.open(reference_path) as src:
            transform_affine = src.transform
            crs = src.crs
        features = []
        for geom, val in shapes(mask.astype(np.uint8), mask=(mask > 0), transform=transform_affine):
            try:
                geom_shape = shape(geom)
                if not geom_shape.is_valid or geom_shape.is_empty:
                    continue
                features.append({
                    "type": "Feature",
                    "properties": {"value": int(val)},
                    "geometry": mapping(geom_shape)
                })
            except Exception:
                continue
        geojson = {"type": "FeatureCollection", "features": features}
        with open(out_path, "w") as f:
            json.dump(geojson, f)
        return out_path
    except Exception as e:
        logger.error(f"GeoJSON export failed: {str(e)}")
        raise

# ---------------------------
# Prediction Pipeline
# ---------------------------
def predict_change(file1, file2, threshold, alpha, progress=gr.Progress()):
    logger.info("Starting predict_change")
    progress(0, desc="Checking inputs...")
    clear_cache()
    default_output = (None, None, None, None, None, None, "", {}, None, None, None, None, None, None)
    if file1 is None or file2 is None:
        logger.warning("Missing input files")
        return (*default_output[:6], "Please upload both images.", *default_output[7:])
    path1 = file1.name
    path2 = file2.name
    try:
        with rasterio.open(path1) as src1:
            if src1.count < 4:
                raise ValueError("Image from Year 1 must have at least 4 bands (RGB+NIR).")
            width1, height1 = src1.width, src1.height
        with rasterio.open(path2) as src2:
            if src2.count < 4:
                raise ValueError("Image from Year 2 must have at least 4 bands (RGB+NIR).")
            width2, height2 = src2.width, src2.height
        if (width1, height1) != (width2, height2):
            raise ValueError("Images must have the same dimensions.")
        original_shape = (height1, width1)
    except Exception as e:
        return (*default_output[:6], f"Error: {str(e)}", *default_output[7:])
    progress(0.3, desc="Predicting changes...")
    try:
        preds, full_size = predict_on_large_4band_tifs(path1, path2, threshold=threshold)
        mask = stitch_patches(preds, full_size)
        if mask.shape != original_shape:
            logger.error(f"Stitched mask shape {mask.shape} does not match original shape {original_shape}")
            raise ValueError("Mask dimensions do not match input image dimensions.")
    except Exception as e:
        return (*default_output[:6], f"Error: Prediction failed - {str(e)}", *default_output[7:])
    progress(0.6, desc="Generating visuals...")
    try:
        rgb1 = normalize_rgb(path1)
        rgb2 = normalize_rgb(path2)
        if rgb1.shape[:2] != original_shape or rgb2.shape[:2] != original_shape:
            logger.error(f"RGB image shapes do not match original shape {original_shape}")
            raise ValueError("RGB image dimensions do not match input image dimensions.")
        overlay = overlay_mask(rgb2, mask, alpha=alpha)
        overlay_path = generate_unique_filename("overlay", "png")
        mask_png_path = generate_unique_filename("mask", "png")
        geotiff_path = export_geotiff(mask, path2)
        geojson_path = export_geojson(mask, path2)
        add_overlay_legend(overlay, original_shape).save(overlay_path)
        add_mask_legend(mask, original_shape).save(mask_png_path)
        map_html = create_interactive_heatmap(mask, path2)
        stats = compute_stats(mask, path2)
        stats_plot = plot_stats(stats)
        comment = generate_comment(mask)
    except Exception as e:
        return (*default_output[:6], f"Error: Visualization failed - {str(e)}", *default_output[7:])
    progress(1.0, desc="Complete!")
    return (
        Image.fromarray((rgb1 * 255).astype(np.uint8)),
        Image.fromarray((rgb2 * 255).astype(np.uint8)),
        Image.open(overlay_path),
        Image.open(mask_png_path),
        add_heatmap_legend(mask),
        comment,
        "Processing complete.",
        stats,
        stats_plot,
        overlay_path,
        mask_png_path,
        geotiff_path,
        geojson_path,
        map_html
    )

# ---------------------------
# Clear Inputs Function
# ---------------------------
def clear_inputs():
    """Clear input fields and reset state."""
    clear_cache()
    return None, None, DEFAULT_THRESHOLD, DEFAULT_ALPHA, "Inputs cleared."

# ---------------------------
# Image URL Validation
# ---------------------------
def validate_image_url(url):
    """Validate if an image URL is accessible."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.getcode() == 200:
                return url
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.warning(f"Failed to load image from {url}: {str(e)}")
    return None

# ---------------------------
# Gradio Dashboard with Tabs
# ---------------------------
custom_theme = gr.themes.Default(
    primary_hue=gr.themes.colors.indigo,  # Switched to indigo for a modern, professional tone
    secondary_hue=gr.themes.colors.cyan,  # Cyan for a fresh, vibrant secondary color
    neutral_hue=gr.themes.colors.gray,    # Gray for neutral, versatile backgrounds
    radius_size=gr.themes.sizes.radius_lg,  # Larger radius for softer, modern edges
    text_size=gr.themes.sizes.text_lg,      # Slightly larger text for readability
)

with gr.Blocks(
    title="Land Change Detection System",
    theme=custom_theme,
    css="""
    :root {
        --primary-600: #4f46e5; /* Vibrant indigo */
        --primary-700: #4338ca; /* Darker indigo for hover */
        --secondary-600: #06b6d4; /* Bright cyan */
        --secondary-700: #0891b2; /* Darker cyan for hover */
        --neutral-900: #111827; /* Deep gray for dark mode */
        --neutral-800: #1f2937; /* Slightly lighter gray */
        --neutral-300: #d1d5db; /* Light gray for accents */
        --neutral-200: #e5e7eb; /* Very light gray for backgrounds */
        --background-fill-primary: #f9fafb; /* Clean, off-white background */
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.08);
        --shadow-md: 0 6px 16px rgba(0,0,0,0.12);
        --font-family-base: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-family-heading: 'Poppins', sans-serif;
    }

    .gradio-container {
        background: linear-gradient(135deg, var(--background-fill-primary), var(--neutral-200));
        font-family: var(--font-family-base);
        color: var(--neutral-800);
        line-height: 1.65;
        padding: 2.5rem;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    h1, h2, h3 {
        font-family: var(--font-family-heading);
        font-weight: 700; /* Bolder headings for emphasis */
        color: var(--primary-700);
        margin: 1.5rem 0 0.75rem;
        line-height: 1.3;
    }

    h1 { font-size: 2.25rem; }
    h2 { font-size: 1.75rem; }
    h3 { font-size: 1.25rem; }

    p, li {
        font-size: 1.05rem;
        color: var(--neutral-800);
        line-height: 1.8;
    }

    .card {
        background: var(--background-fill-primary); /* Ties card to theme */
        border-radius: var(--radius_size);
        padding: 1.75rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .card:hover {
        transform: translateY(-0.3rem);
        box-shadow: 0 10px 24px rgba(0,0,0,0.15);
    }

    .hero-section {
    background: rgba(67, 56, 202, 0.65), /* Solid indigo overlay from --primary-700 */
                url('https://images.unsplash.com/photo-1548287053-99cb39e360cd?q=80&w=1600&auto=format&fit=crop') no-repeat center;
    background-size: cover;
    border-radius: var(--radius_size);
    padding: 5rem 2rem;
    margin-bottom: 2.5rem;
    text-align: center;
    color: var(--neutral-200); /* Light gray for contrast */
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.35); /* Subtle shadow for legibility */
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(67, 56, 202, 0.25); /* Lighter indigo overlay for depth */
    z-index: 1;
}

.hero-section > * {
    position: relative;
    z-index: 2;
}

@media (prefers-color-scheme: dark) {
    .hero-section {
        background: rgba(31, 41, 55, 0.7), /* Solid dark gray from --neutral-800 */
                    url('https://images.unsplash.com/photo-1548287053-99cb39e360cd?q=80&w=1600&auto=format&fit=crop') no-repeat center;
        background-size: cover;
    }
    .hero-section::before {
        background: rgba(31, 41, 55, 0.3); /* Slightly lighter dark gray for dark mode */
    }
}

    .primary-button {
        background: linear-gradient(90deg, var(--primary-600), var(--secondary-600));
        color: #ffffff;
        border: none;
        border-radius: 0.5rem;
        padding: 0.9rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        font-family: var(--font-family-base);
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .primary-button:hover {
        background: linear-gradient(90deg, var(--primary-700), var(--secondary-700));
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .secondary-button {
        background: transparent;
        color: var(--primary-700);
        border: 2px solid var(--primary-600);
        border-radius: 0.5rem;
        padding: 0.85rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        font-family: var(--font-family-base);
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .secondary-button:hover {
        background: var(--primary-600);
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: var(--shadow-sm);
    }

    .rounded-image img {
        border-radius: var(--radius_size);
        border: 2px solid var(--neutral-300);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        max-width: 100%;
        height: auto;
    }

    .rounded-image img:hover {
        transform: scale(1.05);
        box-shadow: var(--shadow-md);
    }

    .accordion {
        border-radius: var(--radius_size) !important;
        box-shadow: var(--shadow-sm) !important;
        background: var(--background-fill-primary) !important;
        margin: 1rem 0;
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }

    .accordion:hover {
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-2px);
    }

    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 768px) {
        .gradio-container {
            padding: 1.5rem;
        }
        .gradio-row > div {
            flex: 1 1 100% !important;
            margin-bottom: 1.25rem;
        }
        .hero-section {
            padding: 3rem 1.5rem;
        }
        h1 { font-size: 1.75rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.15rem; }
        .primary-button, .secondary-button {
            padding: 0.75rem 1.5rem;
            font-size: 0.95rem;
        }
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --background-fill-primary: #1f2937; /* Darker background for dark mode */
            --neutral-200: #d1d5db; /* Light gray for text */
            --neutral-300: #9ca3af; /* Slightly darker for accents */
        }
        .gradio-container {
            background: linear-gradient(135deg, var(--neutral-900), var(--neutral-800));
            color: var(--neutral-200);
        }
        p, li {
            color: var(--neutral-200);
        }
        .card, .accordion {
            background: var(--neutral-800) !important;
            box-shadow: var(--shadow-md);
        }
        .card:hover, .accordion:hover {
            box-shadow: 0 12px 28px rgba(0,0,0,0.4);
        }
        .hero-section {
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                        url('https://images.unsplash.com/photo-1548287053-99cb39e360cd?q=80&w=1600&auto=format&fit=crop') no-repeat center;
            background-size: cover;
        }
    }
    """
) as demo:
    gr.Markdown("""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700&display=swap" rel="stylesheet">
    """)
    
    with gr.Tab("Home"):
        gr.Markdown("""
        <div class="hero-section">
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Change Detection System</h1>
            <p style="font-size: 1.25rem; max-width: 800px; margin: 0 auto;">
                Powerful tools for monitoring land cover changes over time using satellite imagery. Suitable for environmental studies, urban planning, and disaster assessment..
            </p>
        </div>
        """, elem_classes=["fade-in"])

        gr.Markdown("<h2>Overview</h2>", elem_classes=["fade-in"])
        gr.Markdown("""
        <div class="card">
            <p>
                Our Land Change Detection System uses state-of-the-art deep learning to analyze multispectral satellite imagery, identifying changes in land cover over time. Designed for researchers, environmental scientists, and urban planners, this tool provides detailed visualizations, geospatial data exports, and actionable insights for monitoring environmental and urban transformations.
            </p>
        </div>
        """, elem_classes=["fade-in"])

        gr.Markdown("<h2>Key Features</h2>", elem_classes=["fade-in"])
        with gr.Row(variant="panel", equal_height=True):
            features = [
                ("chart-line", "Advanced Visualizations", "Interactive heatmaps, overlay maps, and statistical charts with clear, professional-grade legends."),
                ("download", "Geospatial Outputs", "Export results as GeoTIFF, GeoJSON, or PNG files with full georeferencing for GIS integration."),
                ("compass", "User-Friendly Interface", "Streamlined tab-based workflow for seamless navigation and analysis.")
            ]
            for icon, title, desc in features:
                with gr.Column():
                    gr.Markdown(f"""
                    <div class="card fade-in">
                        <i class="fas fa-{icon} fa-2x" style="color: var(--secondary-600); margin-bottom: 0.5rem;"></i>
                        <h3 style="font-size: 1.25rem;">{title}</h3>
                        <p>{desc}</p>
                    </div>
                    """)

        with gr.Accordion("How It Works", open=False, elem_classes=["accordion"]):
            gr.Markdown("""
            <div class="card">
                <p>
                    This system processes two GeoTIFF images to detect changes:
                    <ul style="list-style-type: disc; margin-left: 1.5rem; margin-top: 0.5rem;">
                        <li><strong>Input Validation:</strong> Ensures images have 4+ bands (RGB+NIR) and matching dimensions.</li>
                        <li><strong>Patch Processing:</strong> Analyzes images in 256x256 patches for efficiency.</li>
                        <li><strong>AI Inference:</strong> Uses the ChangeFormer model to detect differences.</li>
                        <li><strong>Output Generation:</strong> Produces stitched masks, visualizations, and exportable geospatial data.</li>
                    </ul>
                </p>
            </div>
            """, elem_classes=["fade-in"])

        gr.Markdown("<h2>How to Use</h2>", elem_classes=["fade-in"])
        with gr.Row(variant="panel", equal_height=True):
            steps = [
                ("1", "Upload Images", "Upload two GeoTIFF files in the Upload tab."),
                ("2", "Set Parameters", "Adjust threshold and opacity settings for tailored results."),
                ("3", "Run Analysis", "Initiate change detection to generate outputs."),
                ("4", "Review & Export", "Explore results and download data for further analysis.")
            ]
            for number, title, desc in steps:
                with gr.Column():
                    gr.Markdown(f"""
                    <div class="card fade-in">
                        <h3 style="font-size: 1.25rem;"><span style="color: var(--secondary-600);">{number}.</span> {title}</h3>
                        <p>{desc}</p>
                    </div>
                    """)

        # gr.Markdown("<h2>Example Outputs</h2>", elem_classes=["fade-in"])
        # with gr.Row():
        #     satellite_url = validate_image_url("https://images.unsplash.com/photo-1451187580459-4349027a8d3c?q=80&w=400&auto=format&fit=crop")
        #     detection_url = validate_image_url("https://images.unsplash.com/photo-1504192010706-8ed0819fb50c?q=80&w=400&auto=format&fit=crop")
        #     gr.Image(value=satellite_url, label="Sample Satellite Imagery", interactive=False, show_download_button=False, height=250, elem_classes=["rounded-image"])
        #     gr.Image(value=detection_url, label="Sample Change Detection", interactive=False, show_download_button=False, height=250, elem_classes=["rounded-image"])

        with gr.Accordion("Applications", open=False, elem_classes=["accordion"]):
            gr.Markdown("""
            <div class="card">
                <ul style="list-style-type: disc; margin-left: 1.5rem; margin-top: 0.5rem;">
                    <li><strong>Environmental Monitoring:</strong> Track deforestation, wetland changes, and climate impacts.</li>
                    <li><strong>Urban Planning:</strong> Monitor infrastructure and urban expansion.</li>
                    <li><strong>Agriculture:</strong> Analyze crop health and land use changes.</li>
                    <li><strong>Research:</strong> Generate data for scientific studies and policy decisions.</li>
                </ul>
            </div>
            """, elem_classes=["fade-in"])

        gr.Markdown("""
        <hr style="border-top: 2px solid var(--neutral-300); margin: 2rem 0;">
        <div style="text-align: center;">
            <h2>Begin Your Analysis</h2>
            <p style="font-size: 1.1rem; margin-bottom: 1rem;">
                Navigate to the <strong>Upload</strong> tab to start analyzing your satellite imagery.
            </p>
             
        </div>
        """, elem_classes=["fade-in"])

    with gr.Tab("Upload"):
        gr.Markdown("""
        <div class="card fade-in">
            <h2>Upload Satellite Imagery</h2>
            <p>Upload two GeoTIFF images with at least 4 bands (RGB + NIR) for change detection analysis.</p>
        </div>
        """)
        with gr.Row():
            file1 = gr.File(label="Year 1 Image (.tif)", file_types=[".tif"], elem_classes=["fade-in"])
            file2 = gr.File(label="Year 2 Image (.tif)", file_types=[".tif"], elem_classes=["fade-in"])
        with gr.Row():
            threshold = gr.Slider(0, 1, value=DEFAULT_THRESHOLD, label="Change Detection Threshold", info="Adjust to control sensitivity of change detection.", elem_classes=["fade-in"])
            alpha = gr.Slider(0, 1, value=DEFAULT_ALPHA, label="Overlay Opacity", info="Adjust transparency of the change overlay.", elem_classes=["fade-in"])
        with gr.Row():
            run_btn = gr.Button("Run Analysis", variant="primary", elem_classes=["primary-button"])
            clear_btn = gr.Button("Clear Inputs", variant="secondary", elem_classes=["secondary-button"])
        status = gr.Textbox(label="Status", interactive=False, elem_classes=["fade-in"])

    with gr.Tab("Results"):
        gr.Markdown("""
        <div class="card fade-in">
            <h2>Analysis Results</h2>
            <p>View the processed images, change detection outputs, and heatmap. Legends are included in overlay and mask images.</p>
        </div>
        """)
        with gr.Row(equal_height=False):
            out_year1 = gr.Image(label="Year 1 RGB Image", interactive=False, show_download_button=True, elem_classes=["rounded-image"])
            out_year2 = gr.Image(label="Year 2 RGB Image", interactive=False, show_download_button=True, elem_classes=["rounded-image"])
        with gr.Row(equal_height=False):
            out_overlay = gr.Image(label="Change Overlay", interactive=False, show_download_button=True, elem_classes=["rounded-image"])
            out_mask = gr.Image(label="Binary Change Mask", interactive=False, show_download_button=True, elem_classes=["rounded-image"])
        with gr.Row(equal_height=False):
            out_heatmap = gr.Image(label="Static Change Heatmap", interactive=False, show_download_button=True, elem_classes=["rounded-image"])
        out_comment = gr.Textbox(label="Analysis Summary", interactive=False, elem_classes=["fade-in"])

    with gr.Tab("Statistics"):
        gr.Markdown("""
        <div class="card fade-in">
            <h2>Change Statistics</h2>
            <p>Detailed metrics on detected changes, including pixel counts and estimated area.</p>
        </div>
        """)
        stats_out = gr.JSON(label="Statistics", elem_classes=["fade-in"])
        stats_plot = gr.Plot(label="Change Distribution", elem_classes=["fade-in"])

    with gr.Tab("Downloads"):
        gr.Markdown("""
        <div class="card fade-in">
            <h2>Download Results</h2>
            <p>Export your analysis results in various formats for further use.</p>
        </div>
        """)
        dl_overlay = gr.File(label="Overlay Image (PNG)", interactive=False, elem_classes=["fade-in"])
        dl_mask = gr.File(label="Binary Mask (PNG)", interactive=False, elem_classes=["fade-in"])
        dl_geotiff = gr.File(label="GeoTIFF Mask", interactive=False, elem_classes=["fade-in"])
        dl_geojson = gr.File(label="GeoJSON Polygons", interactive=False, elem_classes=["fade-in"])

    with gr.Tab("Interactive Map"):
        gr.Markdown("""
        <div class="card fade-in">
            <h2>Interactive Change Heatmap</h2>
            <p>Explore a dynamic map highlighting areas of change intensity.</p>
        </div>
        """)
        heatmap_out = gr.HTML(label="Interactive Heatmap", elem_classes=["fade-in"])

    run_btn.click(
        fn=predict_change,
        inputs=[file1, file2, threshold, alpha],
        outputs=[out_year1, out_year2, out_overlay, out_mask, out_heatmap, out_comment, status, stats_out, stats_plot, dl_overlay, dl_mask, dl_geotiff, dl_geojson, heatmap_out]
    )
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[file1, file2, threshold, alpha, status]
    )

demo.launch()