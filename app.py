import os
import io
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import matplotlib.pyplot as plt
import gradio as gr
import folium
from folium.plugins import HeatMap
import logging
import time
import uuid
import matplotlib.patches as mpatches
import urllib.request
import urllib.error

# ---------------------------
# Setup Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# _________________________
# Initialize the model
# _________________________
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChangeFormer(num_classes=1).to(device)
print("ChangeFormer Model Initialized!")

# ---------------------------
# Load Model
# ---------------------------
logger.info(f"Using device: {device}")
try:
    model = ChangeFormer(num_classes=1).to(device)
    model_path = "/content/drive/MyDrive/newmodel/best_model.pth"
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
# Legend-enhanced Visualization Functions
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
            # Extract bounding box coordinates and CRS
            bounds = src.bounds  # Returns (left, bottom, right, top)
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
    default_output = (None, None, None, None, None, "", {}, None, None, None, None, None, None)
    if file1 is None or file2 is None:
        logger.warning("Missing input files")
        return (*default_output[:5], "Please upload both images.", *default_output[6:])
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
        return (*default_output[:5], f"Error: {str(e)}", *default_output[6:])
    progress(0.3, desc="Predicting changes...")
    try:
        preds, full_size = predict_on_large_4band_tifs(path1, path2, threshold=threshold)
        mask = stitch_patches(preds, full_size)
        if mask.shape != original_shape:
            logger.error(f"Stitched mask shape {mask.shape} does not match original shape {original_shape}")
            raise ValueError("Mask dimensions do not match input image dimensions.")
    except Exception as e:
        return (*default_output[:5], f"Error: Prediction failed - {str(e)}", *default_output[6:])
    progress(0.6, desc="Generating visuals...")
    try:
        rgb = normalize_rgb(path2)
        if rgb.shape[:2] != original_shape:
            logger.error(f"RGB image shape {rgb.shape[:2]} does not match original shape {original_shape}")
            raise ValueError("RGB image dimensions do not match input image dimensions.")
        overlay = overlay_mask(rgb, mask, alpha=alpha)
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
        return (*default_output[:5], f"Error: Visualization failed - {str(e)}", *default_output[6:])
    progress(1.0, desc="Complete!")
    return (
        Image.fromarray((rgb * 255).astype(np.uint8)),
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
with gr.Blocks(title="Change Detection Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Change Detection Dashboard
    A powerful tool for analyzing land cover changes using satellite imagery.
    """)

    with gr.Tab("Home"):
        gr.Markdown("""
        ## Welcome to the Change Detection Dashboard
        Discover insights into land cover transformations with our advanced change detection tool powered by deep learning. Upload satellite images to analyze changes over time, visualize results, and download detailed outputs.

        ---

        ### Key Features
        - **Advanced Analysis**: Detect changes in satellite imagery with at least 4 bands (RGB+NIR) using a transformer-based model.
        - **Comprehensive Visualizations**: View raw images, change overlays, binary masks, heatmaps, and interactive maps.
        - **Downloadable Outputs**: Export results as GeoTIFF, GeoJSON, or PNG files with legends for clarity.
        - **User-Friendly Interface**: Navigate through intuitive tabs for uploading, analyzing, and exploring results.

        ---

        ### How to Use
        1. Go to the **Upload** tab to upload two GeoTIFF images (Year 1 and Year 2) with at least 4 bands.
        2. Adjust the change threshold and overlay opacity for customized analysis.
        3. View results in the **Analysis**, **Stats**, **Downloads**, and **Interactive Heatmap** tabs.
        4. Download high-resolution outputs or explore the interactive heatmap for georeferenced insights.

        ---

        ### Get Started
        Navigate to the **Upload** tab to begin analyzing your satellite imagery.
        """)
        with gr.Row():
            # satellite_url = validate_image_url("https://placehold.co/300x200.png?text=Satellite+Imagery")
            # detection_url = validate_image_url("https://placehold.co/300x200.png?text=Change+Detection")
            gr.Image(value=satellite_url if satellite_url else None,
                     label="Example Satellite Imagery",
                     scale=1,
                     placeholder="Image unavailable")
            gr.Image(value=detection_url if detection_url else None,
                     label="Change Detection Output",
                     scale=1,
                     placeholder="Image unavailable")

    with gr.Tab("Upload"):
        gr.Markdown("Upload two GeoTIFF images with at least 4 bands (RGB+NIR) for analysis. Only the first 4 bands (Red, Green, Blue, Near-Infrared) will be used.")
        with gr.Row():
            file1 = gr.File(label="Image from Year 1 (.tif)", file_types=[".tif"])
            file2 = gr.File(label="Image from Year 2 (.tif)", file_types=[".tif"])
        with gr.Row():
            threshold = gr.Slider(0, 1, value=DEFAULT_THRESHOLD, label="Change Threshold", info="Higher values detect stricter changes.")
            alpha = gr.Slider(0, 1, value=DEFAULT_ALPHA, label="Overlay Opacity", info="Controls transparency of change overlay.")
        with gr.Row():
            run_btn = gr.Button("Run Change Detection", variant="primary")
            clear_btn = gr.Button("Clear Inputs", variant="secondary")
        status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Analysis"):
        gr.Markdown("### Visualization of Change Detection Results")
        gr.Markdown("Note: Overlay and mask images include legends on the right, which extend the image width. Zoom in or download for full clarity.")
        with gr.Row(equal_height=False):
            out1 = gr.Image(label="Raw RGB (Year 2)", interactive=False, show_download_button=True, scale=1)
            out2 = gr.Image(label="Overlay with Prediction", interactive=False, show_download_button=True, scale=1)
        with gr.Row(equal_height=False):
            out3 = gr.Image(label="Binary Change Mask", interactive=False, show_download_button=True, scale=1)
            out4 = gr.Image(label="Change Heatmap (static)", interactive=False, show_download_button=True, scale=1)
        out5 = gr.Textbox(label="System Comment", interactive=False)

    with gr.Tab("Stats"):
        stats_out = gr.JSON(label="Change Statistics")
        stats_plot = gr.Plot(label="Statistics Plot")

    with gr.Tab("Downloads"):
        dl_overlay = gr.File(label="Download Overlay with Legend", interactive=False)
        dl_mask = gr.File(label="Download Mask with Legend", interactive=False)
        dl_geotiff = gr.File(label="Download GeoTIFF Mask", interactive=False)
        dl_geojson = gr.File(label="Download Change Polygons (GeoJSON)", interactive=False)

    with gr.Tab("Interactive Heatmap"):
        heatmap_out = gr.HTML(label="Dynamic Change Intensity Heatmap")

    with gr.Tab("About"):
        gr.Markdown("""
        ### Change Detection Dashboard
        - Upload **two satellite .tif images** with at least 4 bands (RGB+NIR) in the Upload tab. Only the first 4 bands (Red, Green, Blue, Near-Infrared) will be used.
        - Adjust parameters for customized analysis.
        - View **analysis, statistics, heatmaps**, download results (including GeoTIFF & GeoJSON), and explore an **interactive map preview** in their respective tabs.
        - Overlay and mask images include larger legends on the right, extending the image width to preserve original content. Zoom in or download images to view legends clearly.
        - Use the **Clear Inputs** button to reset the input fields and clear previous results.
        - Download images to view them at full resolution, as the interface may scale large images for display.
        """)

    run_btn.click(
        fn=predict_change,
        inputs=[file1, file2, threshold, alpha],
        outputs=[out1, out2, out3, out4, out5, status, stats_out, stats_plot, dl_overlay, dl_mask, dl_geotiff, dl_geojson, heatmap_out]
    )
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[file1, file2, threshold, alpha, status]
    )

demo.launch(debug=True)
