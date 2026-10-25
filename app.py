import os
import sys
import threading
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import torch
import torchvision.models as models
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "custom_trained_model.pkl")


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Image Classifier - Custom Categories & Training")
        self.root.geometry("740x840")
        self.root.minsize(620, 700)
        self.root.configure(bg="#f8f9fa")

        # Feature extractor and models
        self.feature_extractor = None
        self.preprocess = None
        self.imagenet_categories = []
        self.imagenet_model = None

        # Custom dataset in memory: { category_name: [list of image paths] }
        self.custom_dataset = {}
        self.custom_model = None
        self.custom_classes = []

        # Classification state
        self.classify_image_path = None
        self.tk_classify_image = None

        self._build_ui()
        self._load_base_models_async()
        self._load_existing_custom_model()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f8f9fa")
        style.configure("TNotebook", background="#f8f9fa")
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[16, 6])
        style.configure("Header.TLabel", background="#f8f9fa", font=("Segoe UI", 15, "bold"), foreground="#212529")
        style.configure("Subheader.TLabel", background="#f8f9fa", font=("Segoe UI", 9), foreground="#6c757d")
        style.configure("Action.TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("Status.TLabel", background="#f8f9fa", font=("Segoe UI", 9, "italic"), foreground="#495057")

        # Header Frame
        header = ttk.Frame(self.root, padding="16 12 16 6")
        header.pack(fill=tk.X)

        ttk.Label(header, text="Dynamic Image Classifier", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Add any images to create your own custom categories, train a model on them, and classify any image.",
            style="Subheader.TLabel"
        ).pack(anchor=tk.W, pady=(2, 0))

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        # Tab 1: Classify Images
        self.tab_classify = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_classify, text="  1. Classify Image  ")

        # Tab 2: Create Categories & Train Model
        self.tab_train = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text="  2. Create Categories & Train Model  ")

        self._build_classify_tab()
        self._build_train_tab()

        # Bottom Status Bar
        self.status_var = tk.StringVar(value="Initializing base vision model...")
        status_lbl = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel", padding="16 4 16 8")
        status_lbl.pack(fill=tk.X, side=tk.BOTTOM)

    # =========================================================================
    # TAB 1: CLASSIFY IMAGES
    # =========================================================================
    def _build_classify_tab(self):
        # Settings Bar
        settings_box = tk.LabelFrame(
            self.tab_classify,
            text=" Classification Mode ",
            font=("Segoe UI", 9, "bold"),
            bg="#ffffff",
            fg="#212529",
            padx=10,
            pady=8
        )
        settings_box.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(settings_box, text="Active Model:", font=("Segoe UI", 9), bg="#ffffff").pack(side=tk.LEFT, padx=(0, 6))

        self.active_model_var = tk.StringVar(value="Custom User-Trained Model")
        self.combo_active_model = ttk.Combobox(
            settings_box,
            textvariable=self.active_model_var,
            values=["Custom User-Trained Model", "General Vision Model (1,000 Categories)"],
            state="readonly",
            width=32
        )
        self.combo_active_model.pack(side=tk.LEFT, padx=(0, 16))
        self.combo_active_model.bind("<<ComboboxSelected>>", self._on_model_mode_changed)

        self.lbl_model_info = tk.Label(settings_box, text="", font=("Segoe UI", 9, "italic"), bg="#ffffff", fg="#0d6efd")
        self.lbl_model_info.pack(side=tk.LEFT)

        # Buttons
        btn_bar = ttk.Frame(self.tab_classify, padding="12 4")
        btn_bar.pack(fill=tk.X)

        self.btn_select_test_img = ttk.Button(
            btn_bar,
            text="Choose Image to Classify",
            command=self.choose_classify_image,
            style="Action.TButton"
        )
        self.btn_select_test_img.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_run_classify = ttk.Button(
            btn_bar,
            text="Classify Image",
            command=self.run_classification,
            style="Action.TButton",
            state=tk.DISABLED
        )
        self.btn_run_classify.pack(side=tk.LEFT)

        # Image Canvas
        preview_container = ttk.Frame(self.tab_classify, padding="12 0 12 8")
        preview_container.pack(fill=tk.BOTH, expand=True)

        self.classify_canvas = tk.Canvas(
            preview_container,
            bg="#e9ecef",
            highlightthickness=1,
            highlightbackground="#ced4da"
        )
        self.classify_canvas.pack(fill=tk.BOTH, expand=True)
        self.classify_canvas_text = self.classify_canvas.create_text(
            270, 110,
            text="No image selected.\nClick 'Choose Image to Classify' to test any image.",
            font=("Segoe UI", 11),
            fill="#6c757d",
            justify=tk.CENTER
        )

        # Predictions Output Card
        self.classify_results_card = tk.LabelFrame(
            self.tab_classify,
            text=" Prediction Results ",
            font=("Segoe UI", 10, "bold"),
            bg="#ffffff",
            fg="#212529",
            padx=12,
            pady=8
        )
        self.classify_results_card.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.prediction_rows_container = tk.Frame(self.classify_results_card, bg="#ffffff")
        self.prediction_rows_container.pack(fill=tk.X)
        self.prediction_rows = []
        self._rebuild_prediction_rows(5)

    def _rebuild_prediction_rows(self, count):
        for w in self.prediction_rows_container.winfo_children():
            w.destroy()

        self.prediction_rows = []
        for i in range(count):
            row = tk.Frame(self.prediction_rows_container, bg="#ffffff")
            row.pack(fill=tk.X, pady=2)

            name_lbl = tk.Label(row, text=f"{i+1}. ---", width=25, anchor="w", font=("Segoe UI", 9), bg="#ffffff")
            name_lbl.pack(side=tk.LEFT)

            pbar = ttk.Progressbar(row, orient="horizontal", length=220, mode="determinate")
            pbar.pack(side=tk.LEFT, padx=6)

            pct_lbl = tk.Label(row, text="0.0%", width=8, anchor="e", font=("Segoe UI", 9), bg="#ffffff")
            pct_lbl.pack(side=tk.LEFT)

            self.prediction_rows.append((name_lbl, pbar, pct_lbl))

    # =========================================================================
    # TAB 2: CREATE CATEGORIES & TRAIN MODEL
    # =========================================================================
    def _build_train_tab(self):
        # Step 1: Add categories & images
        step1_box = tk.LabelFrame(
            self.tab_train,
            text=" Step 1: Add Your Categories & Images ",
            font=("Segoe UI", 10, "bold"),
            bg="#ffffff",
            fg="#212529",
            padx=12,
            pady=10
        )
        step1_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Quick Add from Folder or Manual
        actions_frame = tk.Frame(step1_box, bg="#ffffff")
        actions_frame.pack(fill=tk.X, pady=(0, 8))

        btn_add_folder = ttk.Button(
            actions_frame,
            text="+ Add Category From Folder",
            command=self.add_category_from_folder,
            style="Action.TButton"
        )
        btn_add_folder.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(actions_frame, text="Or Category Name:", bg="#ffffff", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.entry_cat_name = ttk.Entry(actions_frame, width=16)
        self.entry_cat_name.pack(side=tk.LEFT, padx=(0, 6))

        btn_create_cat = ttk.Button(
            actions_frame,
            text="Create Category",
            command=self.create_category_manual,
            style="Action.TButton"
        )
        btn_create_cat.pack(side=tk.LEFT)

        # Category Treeview / Table
        tree_frame = tk.Frame(step1_box, bg="#ffffff")
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        columns = ("category", "image_count", "sample_files")
        self.category_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=7)
        self.category_tree.heading("category", text="Category Name")
        self.category_tree.heading("image_count", text="Images Added")
        self.category_tree.heading("sample_files", text="Example Image Files")

        self.category_tree.column("category", width=140, anchor="w")
        self.category_tree.column("image_count", width=100, anchor="center")
        self.category_tree.column("sample_files", width=360, anchor="w")

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.category_tree.yview)
        self.category_tree.configure(yscrollcommand=tree_scroll.set)

        self.category_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Category action buttons
        cat_btn_bar = tk.Frame(step1_box, bg="#ffffff")
        cat_btn_bar.pack(fill=tk.X, pady=(6, 0))

        btn_add_imgs = ttk.Button(
            cat_btn_bar,
            text="+ Add Images to Selected Category",
            command=self.add_images_to_selected_category,
            style="Action.TButton"
        )
        btn_add_imgs.pack(side=tk.LEFT, padx=(0, 8))

        btn_remove_cat = ttk.Button(
            cat_btn_bar,
            text="Remove Category",
            command=self.remove_selected_category,
            style="Action.TButton"
        )
        btn_remove_cat.pack(side=tk.LEFT, padx=(0, 8))

        btn_clear_all = ttk.Button(
            cat_btn_bar,
            text="Clear All",
            command=self.clear_all_categories,
            style="Action.TButton"
        )
        btn_clear_all.pack(side=tk.LEFT)

        # Step 2: Train Model
        step2_box = tk.LabelFrame(
            self.tab_train,
            text=" Step 2: Train Model On Your Categories ",
            font=("Segoe UI", 10, "bold"),
            bg="#ffffff",
            fg="#212529",
            padx=12,
            pady=10
        )
        step2_box.pack(fill=tk.X, padx=12, pady=(0, 8))

        train_top_bar = tk.Frame(step2_box, bg="#ffffff")
        train_top_bar.pack(fill=tk.X)

        self.btn_train_model = ttk.Button(
            train_top_bar,
            text="Train Model Now",
            command=self.train_custom_model,
            style="Action.TButton"
        )
        self.btn_train_model.pack(side=tk.LEFT, padx=(0, 12))

        self.train_status_lbl = tk.Label(
            train_top_bar,
            text="Add at least 2 categories (with images) to train your model.",
            font=("Segoe UI", 9),
            bg="#ffffff",
            fg="#6c757d"
        )
        self.train_status_lbl.pack(side=tk.LEFT)

        self.train_pbar = ttk.Progressbar(step2_box, mode="indeterminate")
        self.train_pbar.pack(fill=tk.X, pady=(8, 0))

    # =========================================================================
    # MODEL LOADING (BACKGROUND)
    # =========================================================================
    def _load_base_models_async(self):
        def load():
            try:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
                net_model = models.mobilenet_v3_large(weights=weights)
                net_model.eval()

                # Feature extractor (outputs 960-dim embedding for any image)
                extractor = models.mobilenet_v3_large(weights=weights)
                extractor.classifier = torch.nn.Identity()
                extractor.eval()

                self.imagenet_model = net_model
                self.feature_extractor = extractor
                self.preprocess = weights.transforms()
                self.imagenet_categories = weights.meta["categories"]

                self.root.after(0, lambda: self.status_var.set("Base feature extractor & vision models ready!"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error initializing vision model: {e}"))

        threading.Thread(target=load, daemon=True).start()

    def _load_existing_custom_model(self):
        if os.path.exists(MODEL_SAVE_PATH):
            try:
                with open(MODEL_SAVE_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.custom_model = data["model"]
                    self.custom_classes = data["classes"]
                    info_text = f"Custom model ready ({len(self.custom_classes)} categories: {', '.join(self.custom_classes[:4])}...)"
                    self.lbl_model_info.config(text=info_text)
                    self.active_model_var.set("Custom User-Trained Model")
            except Exception:
                pass

    # =========================================================================
    # DATASET MANAGEMENT (TAB 2)
    # =========================================================================
    def add_category_from_folder(self):
        """Allows user to select a folder of images; folder name becomes category."""
        folder = filedialog.askdirectory(title="Select Folder for Category")
        if not folder:
            return

        cat_name = os.path.basename(os.path.normpath(folder)).strip()
        if not cat_name:
            cat_name = "Category_" + str(len(self.custom_dataset) + 1)

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        if not image_files:
            messagebox.showwarning("No Images Found", f"No supported image files (.jpg, .png, etc.) found in:\n{folder}")
            return

        if cat_name in self.custom_dataset:
            # Append images
            self.custom_dataset[cat_name] = list(set(self.custom_dataset[cat_name] + image_files))
        else:
            self.custom_dataset[cat_name] = image_files

        self._refresh_category_tree()
        self.status_var.set(f"Added category '{cat_name}' with {len(image_files)} images.")

    def create_category_manual(self):
        cat_name = self.entry_cat_name.get().strip()
        if not cat_name:
            messagebox.showinfo("Enter Name", "Please enter a category name first.")
            return

        if cat_name not in self.custom_dataset:
            self.custom_dataset[cat_name] = []
            self.entry_cat_name.delete(0, tk.END)
            self._refresh_category_tree()
            self.status_var.set(f"Created category '{cat_name}'. Now click 'Add Images' to add pictures.")
        else:
            messagebox.showinfo("Exists", f"Category '{cat_name}' already exists.")

    def add_images_to_selected_category(self):
        selected = self.category_tree.selection()
        if not selected:
            messagebox.showinfo("Select Category", "Please click on a category from the list above first.")
            return

        item = self.category_tree.item(selected[0])
        cat_name = item["values"][0]

        files = filedialog.askopenfilenames(
            title=f"Select Images for Category '{cat_name}'",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not files:
            return

        existing = self.custom_dataset.get(cat_name, [])
        self.custom_dataset[cat_name] = list(set(existing + list(files)))
        self._refresh_category_tree()
        self.status_var.set(f"Added {len(files)} image(s) to '{cat_name}'. Total: {len(self.custom_dataset[cat_name])}")

    def remove_selected_category(self):
        selected = self.category_tree.selection()
        if not selected:
            return
        item = self.category_tree.item(selected[0])
        cat_name = str(item["values"][0])
        if cat_name in self.custom_dataset:
            del self.custom_dataset[cat_name]
            self._refresh_category_tree()
            self.status_var.set(f"Removed category '{cat_name}'.")

    def clear_all_categories(self):
        if not self.custom_dataset:
            return
        if messagebox.askyesno("Confirm", "Clear all created categories and images?"):
            self.custom_dataset.clear()
            self._refresh_category_tree()
            self.status_var.set("Cleared all categories.")

    def _refresh_category_tree(self):
        for item in self.category_tree.get_children():
            self.category_tree.delete(item)

        for cat_name, file_list in self.custom_dataset.items():
            sample_names = ", ".join([os.path.basename(p) for p in file_list[:3]])
            if len(file_list) > 3:
                sample_names += f" (+{len(file_list) - 3} more)"
            self.category_tree.insert("", tk.END, values=(cat_name, len(file_list), sample_names or "(No images yet)"))

        cat_count = len(self.custom_dataset)
        total_images = sum(len(fl) for fl in self.custom_dataset.values())
        if cat_count < 2 or total_images < 2:
            self.train_status_lbl.config(
                text=f"Have {cat_count} category(ies), {total_images} total image(s). Need at least 2 categories with images.",
                fg="#dc3545"
            )
        else:
            self.train_status_lbl.config(
                text=f"Ready to train! {cat_count} categories, {total_images} total images.",
                fg="#198754"
            )

    # =========================================================================
    # STEP 2: TRAIN CUSTOM MODEL
    # =========================================================================
    def train_custom_model(self):
        valid_cats = {k: v for k, v in self.custom_dataset.items() if len(v) > 0}
        if len(valid_cats) < 2:
            messagebox.showwarning(
                "Need More Categories",
                "To train an image classification model, please create at least 2 categories and add images to each."
            )
            return

        if self.feature_extractor is None:
            messagebox.showinfo("Please wait", "Vision feature extractor is still initializing. Please wait a moment.")
            return

        self.btn_train_model.config(state=tk.DISABLED)
        self.train_pbar.start(10)
        self.train_status_lbl.config(text="Extracting image features & training model...", fg="#0d6efd")

        def run_training():
            try:
                feature_list = []
                label_list = []
                categories = list(valid_cats.keys())

                for cat_idx, cat_name in enumerate(categories):
                    for img_path in valid_cats[cat_name]:
                        try:
                            img = Image.open(img_path).convert("RGB")
                            batch = self.preprocess(img).unsqueeze(0)
                            with torch.no_grad():
                                feat = self.feature_extractor(batch).squeeze(0).numpy()
                            feature_list.append(feat)
                            label_list.append(cat_idx)
                        except Exception as img_err:
                            print(f"Skipping corrupt image {img_path}: {img_err}")

                X = np.array(feature_list)
                y = np.array(label_list)

                if len(X) < 2 or len(np.unique(y)) < 2:
                    raise ValueError("Not enough valid images were extracted to train.")

                # Train instant classifier
                clf = LogisticRegression(max_iter=200, C=1.0)
                clf.fit(X, y)

                # Save model to disk
                self.custom_model = clf
                self.custom_classes = categories

                with open(MODEL_SAVE_PATH, "wb") as f:
                    pickle.dump({"model": clf, "classes": categories}, f)

                def on_success():
                    self.train_pbar.stop()
                    self.btn_train_model.config(state=tk.NORMAL)
                    success_msg = f"Model trained successfully on {len(categories)} categories ({len(X)} images)!"
                    self.train_status_lbl.config(text=success_msg, fg="#198754")
                    self.lbl_model_info.config(text=f"Custom: {', '.join(categories)}")
                    self.active_model_var.set("Custom User-Trained Model")
                    messagebox.showinfo("Training Complete", f"{success_msg}\n\nSwitching to 'Classify Image' tab so you can test it!")
                    self.notebook.select(self.tab_classify)

                self.root.after(0, on_success)
            except Exception as e:
                def on_fail():
                    self.train_pbar.stop()
                    self.btn_train_model.config(state=tk.NORMAL)
                    self.train_status_lbl.config(text=f"Training failed: {e}", fg="#dc3545")
                    messagebox.showerror("Training Error", str(e))
                self.root.after(0, on_fail)

        threading.Thread(target=run_training, daemon=True).start()

    # =========================================================================
    # TAB 1: INFERENCE & PREDICTIONS
    # =========================================================================
    def _on_model_mode_changed(self, event=None):
        mode = self.active_model_var.get()
        if "Custom" in mode:
            if not self.custom_model:
                self.lbl_model_info.config(text="(No custom model trained yet - go to Tab 2 to train!)", fg="#dc3545")
            else:
                self.lbl_model_info.config(text=f"Custom: {len(self.custom_classes)} categories", fg="#198754")
        else:
            self.lbl_model_info.config(text="ImageNet 1,000 standard categories", fg="#0d6efd")

        if self.classify_image_path:
            self.run_classification()

    def choose_classify_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Any Image to Classify", filetypes=filetypes)
        if not path:
            return

        self.classify_image_path = path
        self._display_classify_image(path)
        self.btn_run_classify.config(state=tk.NORMAL)
        self.status_var.set(f"Selected: {os.path.basename(path)}")
        self._reset_predictions()
        self.run_classification()

    def _display_classify_image(self, path):
        try:
            img = Image.open(path)
            self.root.update_idletasks()
            c_width = max(self.classify_canvas.winfo_width(), 320)
            c_height = max(self.classify_canvas.winfo_height(), 180)

            img.thumbnail((c_width - 20, c_height - 20), Image.Resampling.LANCZOS)
            self.tk_classify_image = ImageTk.PhotoImage(img)

            self.classify_canvas.delete("all")
            self.classify_canvas.create_image(c_width // 2, c_height // 2, image=self.tk_classify_image, anchor=tk.CENTER)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _reset_predictions(self):
        for i, (name_lbl, pbar, pct_lbl) in enumerate(self.prediction_rows):
            name_lbl.config(text=f"{i+1}. ---")
            pbar["value"] = 0
            pct_lbl.config(text="0.0%")

    def run_classification(self):
        if not self.classify_image_path:
            return

        mode = self.active_model_var.get()
        use_custom = "Custom" in mode

        if use_custom and not self.custom_model:
            messagebox.showwarning(
                "No Custom Model",
                "You haven't trained a model on your categories yet!\n\nPlease switch to '2. Create Categories & Train Model' tab, add your categories and images, and click 'Train Model Now'."
            )
            return

        self.btn_run_classify.config(state=tk.DISABLED)
        self.status_var.set("Classifying image...")

        def do_inference():
            try:
                img = Image.open(self.classify_image_path).convert("RGB")
                batch = self.preprocess(img).unsqueeze(0)

                results = []

                if use_custom:
                    with torch.no_grad():
                        feat = self.feature_extractor(batch).squeeze(0).numpy()
                    probs = self.custom_model.predict_proba([feat])[0]
                    # Map classes
                    for idx, cat_name in enumerate(self.custom_classes):
                        results.append((cat_name, probs[idx] * 100))
                    # Sort descending
                    results.sort(key=lambda x: x[1], reverse=True)
                else:
                    with torch.no_grad():
                        output = self.imagenet_model(batch).squeeze(0).softmax(0)
                        top5_prob, top5_catid = torch.topk(output, min(5, len(self.imagenet_categories)))
                    for p, cid in zip(top5_prob, top5_catid):
                        raw_name = self.imagenet_categories[cid.item()]
                        formatted = raw_name.replace("_", " ").title()
                        results.append((formatted, p.item() * 100))

                self.root.after(0, lambda: self._show_prediction_results(results))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Inference Error", str(e)))
                self.root.after(0, lambda: self.status_var.set("Classification failed."))
            finally:
                self.root.after(0, lambda: self.btn_run_classify.config(state=tk.NORMAL))

        threading.Thread(target=do_inference, daemon=True).start()

    def _show_prediction_results(self, results):
        self._rebuild_prediction_rows(len(results))
        for i, (name, pct) in enumerate(results):
            if i < len(self.prediction_rows):
                name_lbl, pbar, pct_lbl = self.prediction_rows[i]
                name_lbl.config(text=f"{i+1}. {name}")
                pbar["value"] = pct
                pct_lbl.config(text=f"{pct:.1f}%")

        if results:
            top_name, top_score = results[0]
            self.status_var.set(f"Predicted: '{top_name}' ({top_score:.1f}% confidence)")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
