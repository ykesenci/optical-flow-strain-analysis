# **Strain analysis of freely moving cells in 3D microscopy**

Code of the paper: "Cytoplasmic intermediate filaments promote glioblastoma cell invasion by controlling cell deformability and mechanosensitive gene expression", published in Cell Reports 2025.

This repository contains the optical flow function that processes 3D images of freely moving nuclei to compute their mean deformation over time. It also contains a pipeline that leverages this function to compute the mean deformation of several populations of freely moving nuclei and to perform statistical analysis between each population.

## **Install**
You will first require the following:

- Python **3.13.7**
- [Conda](https://docs.conda.io/en/latest/) (recommended) *or* `pip`


### Option 1: Conda (Recommended)
```bash
git clone git@github.com:ykesenci/deformation_analysis.git
cd deformation_analysis
conda env create -f environment.yaml
conda activate deformation_analysis
```
### Option 2: Using pip
```bash
git clone git@github.com:ykesenci/deformation_analysis.git
cd deformation_analysis
pip install -r requirements.txt
```
## **Run**

### **Prepare the dataset**

This repository includes a minimal working example of how the videos should be organised. The videos are to be put in the Data directory. The Data directory will contain as many Population directories as required. Each Population directory will contain an Images and a Masks subdirectories. Videos shall be put in Images, and their corresponding segmetation binary masks in Masks. Both .npy and .tiff files are accepted. The Population directories, the videos and the masks files can be named arbitrarily. The videos and masks must contain a single number in their title that coincide. 

```bash
deformation_analysis/
│
├── Data/
│   │
│   ├── Population_1/ # This folder can be named arbitrarily
│   │   ├── Images/
│   │   │   ├── video_001.npy # .npy of .tif extension
│   │   │   ├── video_002.npy
│   │   │   └── ...
│   │   │
│   │   └── Masks/
│   │       ├── video_001.npy # The numbers must coincide with the corresponding video
│   │       ├── video_002.npy
│   │       └── ...
│   │
│   ├── Population_2/
│   │   ├── Images/
│   │   │   ├── cell_42.tif # The name of the file does not matter. Only the number does
│   │   │   └── ...
│   │   └── Masks/
│   │       ├── cell_42.tif
│   │       └── ...
│   │
│   └── ...
│
├── README.md
├── environment.yaml
└── requirements.txt
```

### **Set the configuration parameters**

The `configuration.yaml` file controls the deformation analysis pipeline. Below are the available parameters and their functions:

<custom-element data-json="%7B%22type%22%3A%22table-metadata%22%2C%22attributes%22%3A%7B%22title%22%3A%22Configuration%20Parameters%22%7D%7D" />

| Parameter                     | Description                                                                                                                                                                                                 | Typical Values/Notes                                                                                     |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **`optical_flow_regularisation`** | Controls smoothness of deformation fields. Lower values preserve fine details but may introduce noise. Higher values create smoother flow fields.                                                          | `0.01` (default)                                                                 |
| **`num_iter`**                | Number of iterations for optical flow computation. More iterations improve accuracy but increase computation time.                                                                                     | `100` (default)                                                                      |
| **`num_warp`**                | Number of warping steps for image alignment. `1` for small deformations, higher values for complex movements.                                                                                           | `1` (default)                                                                           |
| **`denoising_std`**           | Standard deviation of Gaussian filter for preprocessing noisy images. Higher values smooth more aggressively.                                                                                         | `0.5` (default)                                                                      |
| **`denoising_indices`**       | Specifies which video shall be denoised with a Gaussian filter within each population. Empty list `[]` means no denoising.                                                                                       | Example: `Population_1: [0, 1]` (denoise first two videos)                                             |
| **`visuals`**                | Boolean to enable/disable for each video the highest deforming stack. `True` saves deformation maps and plots.                                                                                                             | `True` (default)                                                   |
| **`plot_threshold`**         | Maximum deformation magnitude to threshold the strain visualizations.                                                                                                 | `1.5` (default)                                                  |
| **`XY_resolution`**          | Physical resolution in micrometers per pixel (µm/px). The z-resolution si assumed to be 1.                                                                            | `0.1625` (default)|

---

### **Launch the analysis**

Run the following bash command within the root of your directory

```bash
python3 main.py
```

### **Check the results**

The results will be stored in the Figures directory.
