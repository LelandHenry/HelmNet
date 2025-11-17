# HelmNet
HelmNet — Safety Helmet Classification
======================================

HelmNet is a binary image classification project that detects whether a person
is "With Helmet" or "Without Helmet" using convolutional neural networks and
a VGG16-based transfer learning pipeline.

The main notebook is:
- HelmNet_Submission.ipynb

Project Structure (recommended)
-------------------------------
You can organize the repository like this:

HelmNet/
├─ data/
│  ├─ images_proj.npy        # Image dataset (numpy array)
│  └─ Labels_proj.csv        # Labels for each image (0/1 or equivalent)
├─ models/
│  └─ helmnet_best.h5        # (Optional) Saved Keras model weights
├─ HelmNet_Submission.ipynb
├─ requirements.txt
└─ README.txt

Dataset
-------
The notebook expects:

- A numpy array of images:
  - Path (default in the notebook): /content/images_proj.npy
  - Fallback when running locally: /mnt/data/images_proj.npy (used inside the notebook)

- A CSV file of labels:
  - Path (default in the notebook): /content/Labels_proj.csv
  - Fallback when running locally: /mnt/data/Labels_proj.csv

To use this repo outside of Google Colab, the simplest approach is:

1. Create a `data/` folder in the repo root.
2. Put your dataset files there:
   - data/images_proj.npy
   - data/Labels_proj.csv
3. Update the paths near the top of the notebook:

   IMAGES_PATH = "data/images_proj.npy"
   LABELS_PATH = "data/Labels_proj.csv"

Environment & Dependencies
--------------------------
Tested with:
- Python 3.9+ (3.10 also works fine in most cases)
- GPU acceleration is recommended but not required.

Install dependencies with:

   pip install -r requirements.txt

Core libraries used in the notebook:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras
- opencv-python

Jupyter / Colab specific:
- jupyter, notebook (if running locally)
- google-colab (only when running on Colab; not required locally)

Running the Notebook
--------------------
1. Clone the repository (example):

   git clone https://github.com/<your-username>/HelmNet.git
   cd HelmNet

2. (Optional but recommended) Create and activate a virtual environment:

   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate

3. Install dependencies:

   pip install -r requirements.txt

4. Ensure the dataset files are in the correct location. For example:

   HelmNet/
   ├─ data/
   │  ├─ images_proj.npy
   │  └─ Labels_proj.csv

5. Launch Jupyter and open the notebook:

   jupyter notebook

   Then open `HelmNet_Submission.ipynb` and run all cells (Kernel -> Restart & Run All).

Model Variants
--------------
The notebook builds and evaluates multiple models:

- Basic CNN
  - A small convolutional network with several Conv2D, MaxPooling2D,
    BatchNormalization, and Dense layers.

- VGG16
  - Transfer learning from ImageNet, using VGG16 as a feature extractor
    with a shallow classification head.

- VGG16 + FFNN
  - VGG16 backbone with a deeper fully connected classification head.

- VGG16 + Augmentation ("VGG16+Aug")
  - Same backbone with an ImageDataGenerator-based augmentation pipeline
    (rotation, shifts, zoom, horizontal flips, etc.).

The notebook compares validation accuracies and then evaluates the best
model on a held-out test set, printing:

- Test Accuracy
- Precision
- Recall
- F1-score
- A confusion matrix

Saving the Best Model
---------------------
If you want to export the final model to reuse it elsewhere, you can add
the following at the end of the notebook after the best model is chosen:

   model.save("models/helmnet_best.h5")

This will create a `models/` directory (if it doesn’t exist, create it
first) and save the trained Keras model there.

To load this model in another script:

   from tensorflow.keras.models import load_model
   model = load_model("models/helmnet_best.h5")

Notes for Colab vs. Local Usage
-------------------------------
- On Google Colab:
  - You can keep the original `/content/...` paths and upload the dataset
    via the Colab UI or mount Google Drive.
  - `from google.colab.patches import cv2_imshow` is available by default.

- Locally:
  - You do not need `google.colab` at all.
  - Use `matplotlib.pyplot.imshow` or `cv2.imshow` instead of `cv2_imshow`
    if you remove Colab-specific code.
  - Make sure your dataset paths are local (e.g., `data/images_proj.npy`).

License
-------
Add a LICENSE file here if you want others to know how they can use,
modify, and share this project (e.g., MIT, Apache-2.0, etc.).
