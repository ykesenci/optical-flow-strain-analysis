import os
from utils import load_files, plot_all_boxplot_pairs, _normalize_video
from deformations import compute_deformation
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import laplace

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Convert string "None" to None for known keys
    if isinstance(config.get("plot_threshold"), str) and config["plot_threshold"].lower() == "none":
        config["plot_threshold"] = None

    return config


def main():
    deformation_data = {}
    population_directories = sorted([os.path.join(os.path.abspath("Data"), path) for path in os.listdir("Data")])
    configurations = load_config("configuration.yaml")
    denoising_indices = configurations.get("denoising_indices", None)
    for population_directory in population_directories:
        print(f"\n----------- PROCESSING DIRECTORY : {os.path.basename(population_directory)} -----------\n")
        # Load images
        imgs_dir = os.path.join(population_directory, "Images")
        masks_dir = os.path.join(population_directory, "Masks")
        imgs, imgs_filenames = load_files(imgs_dir)
        masks, _ = load_files(masks_dir)

        # Compute deformations
        enorms, deformations = compute_deformation(
            imgs,
            masks,
            alpha=configurations["optical_flow_regularisation"],
            num_iter=configurations["num_iter"],
            num_warp=configurations["num_warp"],
            # sigma=configurations["denoising_std"],
            indices=denoising_indices[os.path.basename(population_directory)] if denoising_indices is not None else None,
        )

        pop_name = os.path.basename(population_directory)
        def_arr = np.array(deformations)
        valid_indices = ~np.isnan(def_arr)

        # Remove the nans from the deformations
        enorms = [e for i, e in enumerate(enorms) if valid_indices[i]]
        deformations = def_arr[valid_indices].tolist()
        deformation_data[pop_name] = deformations

        # Save representative z-stacks
        if configurations["visuals"]:
            for index, enorm in enumerate(enorms):
                # Skip if enorm is NaN or not a valid array
                if enorm is None or (isinstance(enorm, (int, float, np.number)) and np.isnan(enorm)):
                    print(f"⚠️ Skipping invalid enorm for video {index} (enorm={enorm})")
                    continue

                # Check if enorm has the expected shape (e.g., 3D)
                if enorm.ndim < 3:
                    print(f"⚠️ Skipping enorm with unexpected shape {enorm.shape} for video {index}")
                    continue
                # Find the most representative z-stack
                z_norms = np.linalg.norm(enorm, axis = (1,2))
                z_index = np.argmax(z_norms)
                representative_stack = enorm[z_index]

                # Save it
                imshow_kwargs = {}
                vmax = configurations["plot_threshold"]
                um_per_pixel = configurations["XY_resolution"]
                if vmax is not None:
                    imshow_kwargs["vmax"] = vmax
                fig, ax = plt.subplots()
                im = ax.imshow(representative_stack, **imshow_kwargs)
                ax.axis("off")

                # Add colorbar
                plt.colorbar(im, ax=ax)

                # Add scalebar (units are micrometers)
                scalebar = ScaleBar(um_per_pixel, "µm", location="lower right", length_fraction=0.2)
                ax.add_artist(scalebar)

                # Create directory
                plot_dir = os.path.join("Figures/Mean_norms", os.path.basename(population_directory))
                os.makedirs(plot_dir, exist_ok=True)

                filename = os.path.splitext(os.path.basename(imgs_filenames[index]))[0]

                plt.savefig(
                    os.path.join(plot_dir, f"{filename}.pdf"),
                    bbox_inches="tight",
                    pad_inches=0,
                    format="pdf"
                )

                plt.close(fig)
        
    # Find the maximum number of deformations across all populations
    max_deformations = max(len(deforms) for deforms in deformation_data.values())

    # Pad shorter lists with NaN to ensure equal row counts
    padded_data = {}
    for pop_name, deforms in deformation_data.items():
        padded_deforms = deforms + [np.nan] * (max_deformations - len(deforms))
        padded_data[pop_name] = padded_deforms

    # Create DataFrame
    df = pd.DataFrame(padded_data)

    # Save to Excel
    output_file = "Figures/deformations.xlsx"
    df.to_excel(output_file, index=False)

    # Save the boxplots
    plot_all_boxplot_pairs(deformation_data, output_dir="Figures", test_type="t-test")

if __name__ == "__main__":
    main()