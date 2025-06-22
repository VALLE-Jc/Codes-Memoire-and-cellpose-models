import numpy as np
import tifffile
import matplotlib.pyplot as plt
import napari
from cellpose import models
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, binary_closing, remove_small_holes
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import median_filter, gaussian_filter, distance_transform_edt
from scipy.spatial import ConvexHull
from tqdm import tqdm  # Progress bar
import matplotlib

matplotlib.use('TkAgg')  # Required for GUI environments

def load_tiff(filename):
    img = tifffile.imread(filename)
    return img[..., np.newaxis] if img.ndim == 3 else img

def preprocess_channel(img_3d):
    return median_filter(img_3d, size=(1, 3, 3))

def segment_image(
    img, 
    model_path,
    diameter=25,
    anisotropy=1,
    stitch_threshold=0.01,
    min_size=1000,
    flow_threshold=0.1,
    cellprob_threshold=0.05,
    flow3D_smooth=False,
    preprocess=False
):
    model = models.CellposeModel(pretrained_model=model_path, gpu=True)
    masks = []
    all_flows = []

    img_3d = preprocess_channel(img[..., 0]) if preprocess else img[..., 0]

    mask, flows, _ = model.eval(
        img_3d,
        diameter=diameter,
        channels=[0, 0],
        do_3D=False,
        z_axis=0,
        channel_axis=None,
        anisotropy=anisotropy,
        stitch_threshold=stitch_threshold,
        min_size=min_size,
        resample=True,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        flow3D_smooth=flow3D_smooth,
        normalize=True
    )

    # mask = binary_closing(mask, np.ones((3, 3, 3)))
    mask = remove_small_holes(mask, area_threshold=500)
    mask = remove_small_objects(mask, min_size=min_size)

    labeled_mask = mask  # Use raw Cellpose mask

    masks.append(labeled_mask.astype(np.int32))
    all_flows.append(flows)

    return masks, all_flows

def extract_centroids(mask, min_volume=50):
    labeled_mask = label(mask, connectivity=1)
    props = regionprops(labeled_mask)

    centroids = []
    print("🔍 Extracting centroids from labeled regions...")
    for p in tqdm(props):
        if p.area >= min_volume:
            centroids.append(p.centroid)

    return np.array(centroids), labeled_mask

def compute_convex_hull_volume(centroids):
    if len(centroids) < 4:
        print("⚠️ At least 4 points are required to compute a 3D convex hull.")
        return None, None
    hull = ConvexHull(centroids)
    return hull, hull.volume

def visualize_napari(img, mask, centroids, hull, labeled_mask):
    viewer = napari.Viewer()

    viewer.add_image(
        img[..., 0],
        name="Channel 0",
        contrast_limits=np.percentile(img[..., 0], [1, 99])
    )

    viewer.add_labels(mask, name="Segmentation", opacity=0.6)
    viewer.add_labels(labeled_mask, name="Labeled Mask", opacity=0.3)
    viewer.add_points(centroids, name="Centroids", size=8, face_color="red")

    if hull is not None:
        vertices = centroids[hull.vertices]
        faces = hull.simplices
        viewer.add_surface((centroids, faces), name="Convex Hull", opacity=0.3, shading="smooth", colormap="cyan")

    napari.run()

if __name__ == "__main__":
    filepath = r"YOUR/LOCAL/PATH/TO/FILE.tif"
    img = load_tiff(filepath)

    if img is None or img.size == 0:
        print("❌ Error: The image could not be loaded.")
        exit()

    model_path = r"YOUR/LOCAL/PATH/TO/MODEL"

    masks, flows = segment_image(
        img,
        model_path=model_path,
        diameter=35,
        anisotropy=1,
        stitch_threshold=0.1,
        min_size=4000,
        flow_threshold=0.4,
        cellprob_threshold=0.4,
        preprocess=False
    )

    if not masks or all(np.max(mask) == 0 for mask in masks):
        print("⚠️ Warning: No objects were detected in the masks.")
    else:
        mask = masks[0]

        if np.max(mask) == 0:
            print("⚠️ No labeled objects found in mask.")
        else:
            centroids, labeled_mask = extract_centroids(mask)

            if len(centroids) == 0:
                print("⚠️ No centroids extracted. Check segmentation thresholds or mask quality.")
                hull, hull_volume = None, None
            else:
                hull, hull_volume = compute_convex_hull_volume(centroids)

            print("\n📊 Summary:")
            print(f"🟢 Total segmented objects (labeled): {labeled_mask.max()}")
            print(f"🔴 Total centroids detected: {len(centroids)}")

            # 🔽🔽🔽 NEW BLOCK FOR VOLUME AND DENSITY 🔽🔽🔽
            XYresolution = 0.28  # µm
            Zresolution = 1.0    # µm

            if hull is not None:
                physical_volume_um3 = hull_volume * (XYresolution * XYresolution * Zresolution)
                SGN_density = (len(centroids) / physical_volume_um3) * 1e5

                print(f"📐 Physical volume: {physical_volume_um3:.2f} µm³")
                print(f"🧠 SGN density: {SGN_density:.2f} SGN / 10⁵ µm³")
                print(f"🔷 Convex hull volume: {hull_volume:.2f} voxels")
                print(f"🔺 Number of vertices in hull: {len(hull.points)}")
                print(f"🔻 Number of triangles in hull: {len(hull.simplices)}")
            else:
                print("⚠️ Convex hull was not computed (less than 4 points).")
            # 🔼🔼🔼 END OF NEW BLOCK 🔼🔼🔼

            visualize_napari(img, mask, centroids, hull, labeled_mask)




