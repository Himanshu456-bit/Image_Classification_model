# Dynamic Image Classifier (User Categories + Tkinter GUI)

A dynamic image classification desktop application built with Python, Tkinter, PyTorch, and scikit-learn.

Instead of being restricted to fixed hardcoded categories, **you can add any images, create your own custom categories, and train the model specifically for your needs** right inside the GUI!

---

## Key Features

1. **Dynamic Category Creation**:
   - **From Folder**: Select any folder of images; the folder name automatically becomes the category name!
   - **Manual**: Enter a category name and add image files (`.jpg`, `.png`, `.jpeg`, `.webp`, `.bmp`).
2. **Instant Deep Transfer Learning**:
   - Uses deep feature representations (MobileNetV3 backbone) to extract rich visual embeddings.
   - Learns your custom categories in just **1 to 2 seconds**, even with only a few sample images per category.
3. **Classify Any New Image**:
   - Upload any image and immediately classify it against your custom categories with visual confidence progress bars.
   - You can also toggle over to the **General Vision Model (1,000 everyday categories)** at any time.

---

## Project Structure

```text
simple_image_classifier/
├── app.py                         # Main Tkinter desktop application
├── sample_dataset/                # Pre-generated sample category folders
│   ├── Red_Apples/
│   ├── Green_Pears/
│   └── Blue_Berries/
├── generate_sample_categories.py  # Script that created sample folders
├── custom_trained_model.pkl       # Saved trained model for user categories
└── README.md                      # Documentation
```

---

## How to Run

Open your terminal or PowerShell and run:

```powershell
python C:\Users\HIM-SHU\.gemini\antigravity\scratch\simple_image_classifier\app.py
```

### Quick Walkthrough:
1. Go to tab **"2. Create Categories & Train Model"**.
2. Click **"+ Add Category From Folder"** (you can select any folder of images, such as one from `sample_dataset/`).
3. Add at least 2 categories with images.
4. Click **"Train Model Now"** — it finishes in 1–2 seconds.
5. The app automatically switches to **"1. Classify Image"**, where you can pick any test image and see how your model classifies it!
