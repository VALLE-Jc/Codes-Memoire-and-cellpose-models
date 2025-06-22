from cellpose import models 
import tifffile
import napari
from skimage import measure
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Parameters ----------
XY_resolution = 0.2840907
min_size = 1500
model_path = r"PATH/TO/MODEL.pth"
export_path = r"PATH/TO/EXPORT/DIRECTORY"
FolderOfInterest = r"PATH/TO/INPUT/DIRECTORY"
max_jump_distance = 200  # Max allowed jump distance between centroids in pixels (adjust as needed)

# ---------- Nearest Neighbor Ordering with Max Jump Limit ----------
def order_centroids_nn_limited(centroids, max_jump):
    if len(centroids) <= 1:
        return centroids
    ordered = [centroids[0]]
    remaining = centroids[1:].tolist()
    while remaining:
        last = ordered[-1]
        dists = cdist([last], remaining)[0]
        valid_indices = np.where(dists <= max_jump)[0]
        if len(valid_indices) == 0:
            print("Warning: no next centroid within max_jump, stopping chain early")
            break
        valid_dists = dists[valid_indices]
        idx_min_valid = valid_indices[np.argmin(valid_dists)]
        ordered.append(remaining.pop(idx_min_valid))
    return np.array(ordered)

# ---------- Load Cellpose Model ----------
model = models.CellposeModel(gpu=True, pretrained_model=model_path)

# ---------- Loop Over All _ch01.tif Images ----------
root = Path(FolderOfInterest)
for folder in root.iterdir():
    if folder.is_dir():
        for ch01_file in folder.glob('*IHC*_ch01.tif'):
            print(f"Found file: {ch01_file}")
            ImageName = ch01_file.name.replace('_ch01.tif', '')
            filename_fig = os.path.join(export_path, (ImageName + "_composite.png"))

            if Path(filename_fig).exists():
                print(f"Image was already analyzed")
                continue

            # ---------- Load Image ----------
            img = tifffile.imread(str(ch01_file))
            if img.ndim > 2:
                img = img.max(axis=0)

            # ---------- Run Cellpose ----------
            masks, flows, styles = model.eval(img, diameter=55, channels=[0, 0])

            # ---------- Filter by Area ----------
            filtered_masks = np.zeros_like(masks)
            for region in measure.regionprops(masks):
                if region.area >= min_size:
                    filtered_masks[masks == region.label] = region.label

            # ---------- Centroids ----------
            centroids = [region.centroid for region in measure.regionprops(filtered_masks)]
            centroids = np.array(centroids)
            if centroids.size == 0:
                centroids = centroids.reshape((0, 2))
            else:
                centroids = centroids[(centroids[:, 0] < img.shape[0]) & (centroids[:, 1] < img.shape[1])]

            # ---------- Spline fitting with try-except ----------
            if len(centroids) >= 2:
                try:
                    ordered_centroids = order_centroids_nn_limited(centroids, max_jump=max_jump_distance)
                    points = ordered_centroids[:, [1, 0]]
                    num_points = points.shape[0]
                    k = min(3, num_points - 1)  # spline degree

                    if k < 1:
                        arc_length_um = None
                        cell_density = None
                        spline_points_rc = None
                    else:
                        tck, u = splprep(points.T, s=0, k=k)
                        unew = np.linspace(0, 1, 1000)
                        out = splev(unew, tck)

                        diffs = np.diff(out, axis=1)
                        segment_lengths = np.sqrt(diffs[0]**2 + diffs[1]**2)
                        arc_length_pixels = np.sum(segment_lengths)
                        arc_length_um = arc_length_pixels * XY_resolution
                        cell_density = len(centroids) / arc_length_um

                        spline_points = np.vstack(out).T
                        spline_points_rc = spline_points[:, [1, 0]]

                except Exception as e:
                    print(f"Warning: spline fitting failed for image {ImageName} with error: {e}")
                    arc_length_um = None
                    cell_density = None
                    spline_points_rc = None

            else:
                arc_length_um = None
                cell_density = None
                spline_points_rc = None

            # ---------- Save Composite Image ----------
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Left: Raw image
            axs[0].imshow(img, cmap='gray')
            axs[0].set_title("Raw Image")
            axs[0].axis('off')

            # Right: Image with overlays
            axs[1].imshow(img, cmap='gray')
            axs[1].imshow(filtered_masks, cmap='viridis', alpha=0.4)

            # Centroids
            if len(centroids) > 0:
                axs[1].scatter(centroids[:, 1], centroids[:, 0], s=10, c='red', label='Centroids')

            # Spline
            if spline_points_rc is not None:
                axs[1].plot(spline_points_rc[:, 1], spline_points_rc[:, 0], c='cyan', linewidth=2, label='Spline Path')

            axs[1].set_title("Segmentation + Path")
            axs[1].axis('off')

            plt.tight_layout()
            os.makedirs(export_path, exist_ok=True)
            plt.savefig(filename_fig, dpi=300)
            plt.close()
            print(f"🖼️ Composite image saved at: {filename_fig}")

            # ---------- Export CSV ----------
            quant = {
                'Image': [ImageName],
                'NumCells': [len(centroids)],
                'ArcLength_um': [arc_length_um],
                'Density_cells_per_um': [cell_density]
            }
            df_quant = pd.DataFrame(quant)
            os.makedirs(export_path, exist_ok=True)
            csv_save_path = os.path.join(export_path, ImageName + "_density.csv")
            df_quant.to_csv(csv_save_path, index=False)
            print(f"💾 CSV saved at: {csv_save_path}")


