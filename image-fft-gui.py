import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import threading
from matplotlib.colors import LinearSegmentedColormap
import cv2

COLOR_MAPS = [
    "gray", "viridis", "plasma", "inferno", "magma", "cividis",
    "jet", "hot", "cool", "spring", "summer", "autumn", "winter"
]

ZOOM_LEVELS = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100%

def detect_symmetry(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    spectrum = magnitude_spectrum.astype(np.uint8)
    _, thresh = cv2.threshold(spectrum, 180, 255, cv2.THRESH_BINARY)

    h, w = thresh.shape
    center_mask = np.zeros_like(thresh)
    cv2.circle(center_mask, (w//2, h//2), 10, 255, -1)
    isolated = cv2.bitwise_and(thresh, cv2.bitwise_not(center_mask))
    num_peaks = cv2.countNonZero(isolated)

    h_img, w_img = gray.shape
    half_w = w_img // 2
    half_h = h_img // 2
    left = gray[:, :half_w]
    right = np.fliplr(gray[:, w_img - half_w:])
    vert_sym = np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) < 50

    top = gray[:half_h, :]
    bottom = np.flipud(gray[h_img - half_h:, :])
    horiz_sym = np.mean(np.abs(top.astype(np.float32) - bottom.astype(np.float32))) < 50

    if not vert_sym and not horiz_sym:
        if num_peaks > 100:
            return "p4 or p6 (chiral, strong grid)"
        elif num_peaks > 50:
            return "p2 or p3 (chiral, moderate)"
        elif num_peaks > 20:
            return "pg or pgg (some structure)"
        else:
            return "p1 (chiral, weak symmetry)"
    else:
        if num_peaks > 100:
            return "p4m or p6m (strong grid + reflection)"
        elif num_peaks > 50:
            return "pmm or cmm (moderate symmetry + reflection)"
        elif num_peaks > 20:
            return "pmg or pgg (some repeating structure)"
        else:
            return "p1 (weak symmetry)"

def get_dominant_colors_kmeans(img_array, n_colors=3):
    img_flat = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(img_flat)
    return kmeans.cluster_centers_

def generate_custom_colormap(dominant_colors):
    colors = [tuple(color / 255) for color in dominant_colors]
    return LinearSegmentedColormap.from_list('custom_cmap', colors)

def zoom_fft_image(magnitude, zoom_factor):
    h, w = magnitude.shape
    cx, cy = w // 2, h // 2
    zw, zh = int(w * zoom_factor), int(h * zoom_factor)
    return magnitude[cy - zh // 2:cy + zh // 2, cx - zw // 2:cx + zw // 2]

def enhance_contrast_methods(magnitude):
    norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    gamma_corrected = np.power(norm, 0.4)
    stretched = np.clip(norm * 1.5, 0, 1)
    return {
        "log": norm,
        "gamma": gamma_corrected,
        "stretch": stretched
    }

def save_fft_versions(img_rgb, contrast_maps, cmap, output_folder, base_name, zoom_label):
    # Convert RGB (PIL/Numpy) → BGR for OpenCV
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    wallpaper_group = detect_symmetry(img_bgr)

    for method, data in contrast_maps.items():
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image", fontsize=12)
        plt.imshow(img_rgb)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"FFT ({zoom_label} | {method})", fontsize=12)
        plt.imshow(data, cmap=cmap)
        plt.axis("off")

        # Add label under image
        plt.figtext(0.5, 0.02,
                    f"Wildly tentative wallpaper group assignment: {wallpaper_group}",
                    ha="center", fontsize=10, color="gray")

        plt.subplots_adjust(top=0.85, bottom=0.12)
        out_path = os.path.join(output_folder, f"{base_name}_fft_{zoom_label}_{method}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def compute_fft_and_save(image_path, output_folder, _cmap_choice):
    try:
        img_rgb = Image.open(image_path).convert('RGB')
        img_array_rgb = np.array(img_rgb)

        img_gray = img_rgb.convert('L')
        img_array_gray = np.array(img_gray)

        f = np.fft.fft2(img_array_gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)

        dominant_colors = get_dominant_colors_kmeans(img_array_rgb, n_colors=5)
        custom_cmap = generate_custom_colormap(dominant_colors)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for zoom in ZOOM_LEVELS:
            zoomed = zoom_fft_image(magnitude, zoom)
            contrast_maps = enhance_contrast_methods(zoomed)
            zoom_label = f"{int(zoom * 100)}pct"
            save_fft_versions(img_array_rgb, contrast_maps, custom_cmap, output_folder, base_name, zoom_label)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def batch_process(folder_path, cmap_choice):
    output_folder = os.path.join(folder_path, "output_fft")
    os.makedirs(output_folder, exist_ok=True)

    extensions = ('.jpg', '.jpeg', '.png')
    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(extensions)
        and not f.startswith("output_fft")
        and "_fft_" not in f
    ]

    if not images:
        messagebox.showinfo("No Images", "No unprocessed JPG or PNG images found in the selected folder.")
        return

    print(f"Processing {len(images)} images...")

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        print(f"Processing: {img_file}")
        compute_fft_and_save(img_path, output_folder, cmap_choice)

    messagebox.showinfo("Done", f"Processed {len(images)} images.\nSaved in:\n{output_folder}")
    print("All images processed.")

def start_batch_process(folder_path, cmap_choice):
    threading.Thread(target=batch_process, args=(folder_path, cmap_choice), daemon=True).start()

def browse_folder():
    folder = filedialog.askdirectory()
    if folder:
        selected_cmap = cmap_var.get()
        start_batch_process(folder, selected_cmap)

# GUI
root = tk.Tk()
root.title("Batch 2D FFT + Multi-Zoom + Contrast Enhancement")

tk.Label(root, text="Colormap (Auto = based on image)").pack(pady=5)
cmap_var = tk.StringVar(value="Auto")
cmap_dropdown = ttk.Combobox(root, textvariable=cmap_var, values=["Auto"] + COLOR_MAPS, state="readonly", width=30)
cmap_dropdown.pack(pady=5)

browse_button = ttk.Button(root, text="Choose Folder and Start", command=browse_folder)
browse_button.pack(pady=20)

root.mainloop()
