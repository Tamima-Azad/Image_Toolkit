import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Menu, Toplevel, StringVar
from tkinter.ttk import Scale
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import sys
import os

# Set a minimum standard size for display panels
PANEL_SIZE = (400, 400)

# --- NEW Custom Dialog Class for Combined Operation Parameters and Source Selection ---
class OperationParameterDialog(Toplevel):
    def __init__(self, parent, title, label1_text, label2_text, default1, default2, is_smoothing=False, show_parameters=True):
        super().__init__(parent)
        self.transient(parent)
        self.title(title)
        self.result = None # (val1, val2, source_choice)
        self.grab_set()

        self.label1_text = label1_text
        self.label2_text = label2_text
        self.default1 = default1
        self.default2 = default2
        self.is_smoothing = is_smoothing
        self.show_parameters = show_parameters # Flag to show/hide parameter inputs

        self.source_choice_var = StringVar(self, value="processed") # Default to 'processed'

        # Center the dialog (best effort)
        try:
            parent_x = parent.winfo_x()
            parent_y = parent.winfo_y()
            parent_w = parent.winfo_width()
            parent_h = parent.winfo_height()
            self.geometry(f"+{parent_x + parent_w//2 - 175}+{parent_y + parent_h//2 - 120}")
        except Exception:
            pass

        self.body(self)
        self.buttonbox()

        self.wait_window(self)

    def body(self, master):
        main_frame = tk.Frame(master, padx=15, pady=15)
        main_frame.pack()

        # --- Source Selection Section ---
        source_frame = tk.LabelFrame(main_frame, text="Apply Operation to:", font=("Segoe UI", 10, "bold"), padx=10, pady=5)
        source_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        tk.Radiobutton(
            source_frame, text="Original Image", variable=self.source_choice_var, value="original",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Radiobutton(
            source_frame, text="Processed Image", variable=self.source_choice_var, value="processed",
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=10, pady=5)

        # --- Parameters Section (if show_parameters is True) ---
        if self.show_parameters:
            param_frame = tk.Frame(main_frame)
            param_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10,0))

            tk.Label(param_frame, text=f"{self.label1_text}:", font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=5)
            self.entry1 = tk.Entry(param_frame, width=10)
            self.entry1.insert(0, str(self.default1))
            self.entry1.grid(row=0, column=1, padx=5, pady=5)

            tk.Label(param_frame, text=f"{self.label2_text}:", font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=5)
            self.entry2 = tk.Entry(param_frame, width=10)
            self.entry2.insert(0, str(self.default2))
            self.entry2.grid(row=1, column=1, padx=5, pady=5)

            if self.is_smoothing:
                tk.Label(param_frame, text="N must be odd (3, 5, 7...)", fg="gray", font=("Segoe UI", 8)).grid(row=0, column=2, sticky="w")
            elif self.label1_text == "Kernel Size (3x3 fixed)": # For Sharpening/Edge, to visually indicate fixed
                 tk.Label(param_frame, text="(Fixed)", fg="gray", font=("Segoe UI", 8)).grid(row=0, column=2, sticky="w")
        else:
            self.entry1 = None # No entry widgets
            self.entry2 = None

    def buttonbox(self):
        box = tk.Frame(self)
        box.pack(pady=10)

        w = tk.Button(box, text="Apply", width=10, command=self.ok, default=tk.ACTIVE, bg="#219ebc", fg="#ffffff")
        w.pack(side=tk.LEFT, padx=5, pady=5)
        c = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        c.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def ok(self, event=None):
        val1 = self.default1 # Default values if parameters are hidden
        val2 = self.default2
        
        if self.show_parameters:
            try:
                val1_str = self.entry1.get()
                val2_str = self.entry2.get()

                if self.is_smoothing:
                    # Kernel size must be integer (odd)
                    n = int(val1_str)
                    intensity = float(val2_str)
                    if n % 2 == 0 or n < 3:
                        messagebox.showerror("Error", "Kernel size N must be an odd integer (3, 5, 7...).")
                        return
                    val1, val2 = n, intensity
                else:
                    # Try int first, fallback to float (covers width/height, gamma, thresholds, etc.)
                    try:
                        val1 = int(val1_str)
                    except Exception:
                        try:
                            val1 = float(val1_str)
                        except Exception:
                            val1 = val1_str  # keep raw if not numeric

                    try:
                        val2 = int(val2_str)
                    except Exception:
                        try:
                            val2 = float(val2_str)
                        except Exception:
                            val2 = val2_str

            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter valid numbers.")
                return

        self.result = (val1, val2, self.source_choice_var.get())
        self.destroy()

    def cancel(self, event=None):
        self.result = None
        self.destroy()

# --- ImageToolkit Class (MODIFIED) ---

class ImageToolkit:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Toolkit")
        self.root.minsize(850, 700)
        self.img = None
        self.processed_img = None
        self.img_name = "No Image"
        self.proc_name = "No Image"

        # History stack for Undo (stores tuples of (PIL.Image, name))
        self.history = []
        self.max_history = 25  # keep last 25 steps
        
        # --- Professional Modern Palette ---
        bg_main   = "#c0eeee"
        bg_panel  = "#c8e7f6"
        title_bg  = "#8ecae6"
        title_fg  = "#023047"
        info_bg   = "#e1ba57"
        info_fg   = "#023047"
        button_bg = "#219ebc"
        button_fg = "#ffffff"

        self.root.configure(bg=bg_main)

        # ----------------------------------------
        # --- 1. Menu Bar (Dropdown System) ⬇️ ---
        # ----------------------------------------
        font_setting = ("Segoe UI", 16,)  # <-- Increased font size and made bold for menu bar and dropdowns

        menubar = Menu(root, bg=button_bg, fg=button_fg, font=font_setting)
        root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        menubar.add_cascade(label="File 📂", menu=file_menu, underline=0)
        file_menu.add_command(label="Open Image 🖼️", command=self.open_image)
        file_menu.add_command(label="Save Image 💾", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Reset Image (to Original) ↩️", command=self.reset_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit 🚪", command=root.quit)

        # --- Edit menu with Undo ---
        edit_menu = Menu(menubar, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        menubar.add_cascade(label="Edit ✏️", menu=edit_menu)
        edit_menu.add_command(label="Undo Last Change\tCtrl+Z", command=self.undo_last_change, accelerator="Ctrl+Z")
        # bind Ctrl+Z globally
        root.bind_all("<Control-z>", lambda e: self.undo_last_change())

        # Intensity Menu
        intensity_menu = Menu(menubar, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        menubar.add_cascade(label="Intensity & Contrast ☀️", menu=intensity_menu, underline=0)

        # Slider-based control
        intensity_menu.add_command(label="Brightness & Contrast Slider 🔆", command=self.show_intensity_slider)
        intensity_menu.add_separator()

        intensity_menu.add_command(label="Negative 🔄", command=self.negative)
        intensity_menu.add_command(label="Threshold (Binarize) ⚫⚪", command=self.threshold)

        # Cascaded Transform Menu
        transform_menu = Menu(intensity_menu, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        intensity_menu.add_cascade(label="Log/Gamma Transform ✨", menu=transform_menu, underline=0)
        transform_menu.add_command(label="Log Transform", command=lambda: self.intensity_transform("1"))
        transform_menu.add_command(label="Gamma Transform", command=lambda: self.intensity_transform("2"))

        intensity_menu.add_separator()
        intensity_menu.add_command(label="Histogram View 📊", command=self.histogram)

        # Filtering Menu (CASCADES)
        filter_menu = Menu(menubar, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        menubar.add_cascade(label="Filtering ⚙️", menu=filter_menu, underline=0)

        # Cascaded Smoothing Menu
        smoothing_menu = Menu(filter_menu, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        filter_menu.add_cascade(label="Smoothing (Blur) 🌫️", menu=smoothing_menu, underline=0)
        smoothing_menu.add_command(label="Mean Filter", command=lambda: self.smoothing("mean"))
        smoothing_menu.add_command(label="Gaussian/Weighted Average", command=lambda: self.smoothing("gaussian"))

        # Cascaded Sharpening Menu
        sharpen_menu = Menu(filter_menu, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        filter_menu.add_cascade(label="Sharpening (Enhance) 🔍", menu=sharpen_menu, underline=0)
        sharpen_menu.add_command(label="Laplacian Filter", command=lambda: self.sharpen("1"))
        sharpen_menu.add_command(label="Composite Laplacian Filter", command=lambda: self.sharpen("2"))

        # Edge Detection Menu (CASCADES)
        edge_menu = Menu(menubar, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        menubar.add_cascade(label="Edge Detection 🚧", menu=edge_menu, underline=0)
        edge_menu.add_command(label="Sobel", command=lambda: self.edge_detection("sobel"))
        edge_menu.add_command(label="Prewitt", command=lambda: self.edge_detection("prewitt"))
        edge_menu.add_command(label="Roberts", command=lambda: self.edge_detection("roberts"))
        edge_menu.add_command(label="Canny", command=lambda: self.edge_detection("canny"))

        # Geometry Menu
        geometry_menu = Menu(menubar, tearoff=0, bg=bg_panel, fg=title_fg, font=font_setting)
        menubar.add_cascade(label="Geometry 📏", menu=geometry_menu, underline=0)
        geometry_menu.add_command(label="Resize (Custom Width & Height)", command=self.resize)

        
        # ----------------------------------------
        # --- 2. Title Bar ---
        # ----------------------------------------
        title_bar = tk.Label(
            root, text="Digital Image Toolkit",
            font=("Segoe UI", 16, "bold"),
            bg=title_bg, fg=title_fg,
            pady=12, padx=10
        )
        title_bar.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 10))

        # ----------------------------------------
        # --- 3. Info Labels (Status Bars) ---
        # ----------------------------------------
        info_row = 1 
        self.orig_info = tk.Label(root, text="Original Image: No Image | Size: -", font=("Segoe UI", 10, "bold"),
                                  bg=info_bg, fg=info_fg, anchor="w", padx=8)
        self.orig_info.grid(row=info_row, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 0))
        self.proc_info = tk.Label(root, text="Processed Image: No Image | Size: -", font=("Segoe UI", 10, "bold"),
                                  bg=info_bg, fg=info_fg, anchor="w", padx=8)
        self.proc_info.grid(row=info_row, column=3, columnspan=3, sticky="ew", padx=10, pady=(10, 0))

        # ----------------------------------------
        # --- 4. Image Panels ---
        # ----------------------------------------
        panel_row = 2 
        self.panel_original = tk.Label(root, text="Original Image", bg=bg_panel, relief="groove", width=50, height=25)
        self.panel_original.grid(row=panel_row, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.panel_processed = tk.Label(root, text="Processed Image", bg=bg_panel, relief="groove", width=50, height=25)
        self.panel_processed.grid(row=panel_row, column=3, columnspan=3, padx=10, pady=10, sticky="nsew")

        root.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)
        root.grid_rowconfigure(panel_row, weight=1)

        blank_img = Image.new("L", PANEL_SIZE, color="lightgray")
        self.display_original(blank_img)
        self.display_processed(blank_img)

        # ----------------------------------------
        # --- 5. Developer Info ---
        # ----------------------------------------
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        self.developer_name = "Tamima Azad"
        self.developer_id = "ID: 0812220105101060"

        try:
            photo_name = "Untitled design.png"
            possible_paths = [
                os.path.join(base_path, "images", photo_name),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", photo_name),
            ]
            dev_img_path = next((p for p in possible_paths if os.path.exists(p)), None)

            if dev_img_path:
                dev_img = Image.open(dev_img_path).resize((100, 120))
            else:
                raise FileNotFoundError("Developer image not found in known paths.")

            self.dev_photo = ImageTk.PhotoImage(dev_img)

        except Exception:
            dev_img = Image.new("RGB", (100, 120), color="gray")
            self.dev_photo = ImageTk.PhotoImage(dev_img)

        dev_frame = tk.Frame(root, bg=bg_main)
        dev_frame.place(relx=1.0, rely=0.0, anchor="ne", x=-18, y=18)
        tk.Label(dev_frame, image=self.dev_photo, bg=bg_main, bd=2, relief="ridge").pack()
        tk.Label(dev_frame, text=self.developer_name, font=("Segoe UI", 10, "bold"), bg=bg_main, fg="#22223b").pack(pady=(6, 0))
        tk.Label(dev_frame, text=self.developer_id, font=("Segoe UI", 10), bg=bg_main, fg="#22223b").pack()

    # ---------------- MODIFIED Utility Functions ----------------

    def _get_image_for_operation(self, source_choice):
        """Helper to get the correct PIL Image based on user's source choice."""
        if source_choice == "original":
            if self.img is None:
                messagebox.showerror("Error", "No original image loaded!")
                return None
            return self.img.copy()
        elif source_choice == "processed":
            if self.processed_img is None:
                messagebox.showwarning("Warning", "Processed image is empty. Using Original image.")
                return self.img.copy() # Fallback to original
            return self.processed_img.copy()
        return None # Should not happen

    def _get_array_for_operation(self, source_choice):
        """Helper to get the NumPy array of the selected source image."""
        img = self._get_image_for_operation(source_choice)
        if img:
            return np.array(img.convert("RGB")) if img.mode == "RGB" else np.array(img.convert("L"))
        return None

    # --- Slider Functions (Modified to track state to prevent infinite dialog loop) ---
    def show_intensity_slider(self):
        """Opens a top-level window for brightness and contrast control, with source selection."""
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Use the combined dialog, but hide parameters as they are controlled by sliders
        dialog = OperationParameterDialog(
            self.root, 
            "Brightness & Contrast Source", 
            "Brightness", "Contrast", 1.0, 1.0, 
            show_parameters=False
        )
        source_choice = dialog.result[2] if dialog.result else None
        
        if source_choice is None: return

        self.slider_source_img = self._get_image_for_operation(source_choice)
        if self.slider_source_img is None: return

        # Set a flag on the root window to prevent SourceSelectDialog from appearing 
        # when the slider commands indirectly call display_processed or other methods.
        self.root.is_slider_active = True 

        slider_win = Toplevel(self.root)
        slider_win.title("Brightness & Contrast Control")
        slider_win.geometry("400x250")
        slider_win.transient(self.root)
        slider_win.grab_set()
        slider_win.focus_set()

        # --- Brightness Control ---
        tk.Label(slider_win, text="Brightness (Factor):", font=("Segoe UI", 10, "bold")).pack(pady=(10, 0))
        self.brightness_label = tk.Label(slider_win, text="1.00", width=5)
        self.brightness_label.pack()

        self.brightness_scale = Scale(
            slider_win, from_=0.0, to=2.0, orient='horizontal', length=350, resolution=0.01
        )
        self.brightness_scale.set(1.0)
        self.brightness_scale.pack(pady=(0, 10))

        # --- Contrast Control ---
        tk.Label(slider_win, text="Contrast (Factor):", font=("Segoe UI", 10, "bold")).pack(pady=(10, 0))
        self.contrast_label = tk.Label(slider_win, text="1.00", width=5)
        self.contrast_label.pack()

        self.contrast_scale = Scale(
            slider_win, from_=0.0, to=2.0, orient='horizontal', length=350, resolution=0.01
        )
        self.contrast_scale.set(1.0)
        self.contrast_scale.pack(pady=(0, 10))

        # Now that both scales exist, configure their commands
        self.brightness_scale.configure(command=lambda val: self.apply_brightness_contrast(float(val), float(self.contrast_scale.get()), slider_win))
        self.contrast_scale.configure(command=lambda val: self.apply_brightness_contrast(float(self.brightness_scale.get()), float(val), slider_win))

        # --- Apply/Close Button ---
        tk.Button(
            slider_win, text="Apply Changes",
            command=lambda: self.finalize_brightness_contrast(slider_win),
            bg="#219ebc", fg="#ffffff", font=("Segoe UI", 10, "bold")
        ).pack(pady=10)

        # Handle window close event
        slider_win.protocol("WM_DELETE_WINDOW", lambda: self.cancel_brightness_contrast(slider_win))

    def finalize_brightness_contrast(self, slider_win):
        """Commits the final changes and closes the slider window."""
        self.root.is_slider_active = False # Reset flag
        if hasattr(slider_win, 'current_result'):
            self.update_image(
                slider_win.current_result,
                name=f"B={self.brightness_scale.get():.2f}, C={self.contrast_scale.get():.2f}"
            )
        slider_win.destroy()

    def cancel_brightness_contrast(self, slider_win):
        """Cancels changes and reverts the processed image panel to its state before the slider was opened."""
        self.root.is_slider_active = False # Reset flag
        # Revert the processed panel back to its state *before* the slider opened
        self.display_processed(self.processed_img if self.processed_img is not None else self.img) 
        slider_win.destroy()

    # ---------------- Standard Utility/Operators (MODIFIED to use OperationParameterDialog) ----------------
    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")]
        )
        if path:
            self.img = Image.open(path).convert("RGB")
            self.processed_img = self.img.copy()
            self.img_name = os.path.basename(path)
            self.proc_name = self.img_name
            # clear undo history on new image load
            self.history.clear()
            self.display_original(self.img)

    def save_image(self):
        if self.processed_img is None:
            messagebox.showerror("Error", "No processed image to save!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            self.processed_img.save(path)
            self.proc_name = os.path.basename(path)
            self.update_proc_info()

    def display_original(self, img):
        disp_img = img.copy()
        disp_img.thumbnail(PANEL_SIZE)
        tk_img = ImageTk.PhotoImage(disp_img)
        self.panel_original.config(image=tk_img)
        self.panel_original.image = tk_img
        size = img.size
        self.orig_info.config(text=f"Original Image: {self.img_name} | Size: {size[0]}x{size[1]}")

    def display_processed(self, img):
        disp_img = img.copy()
        actual_size = img.size
        disp_img.thumbnail(PANEL_SIZE)
        tk_img = ImageTk.PhotoImage(disp_img)
        self.panel_processed.config(image=tk_img)
        self.panel_processed.image = tk_img

        self.proc_info.config(
            text=f"Processed Image: {getattr(self, 'proc_name', 'Edited')} | "
            f"Actual Size: {actual_size[0]}x{actual_size[1]}"
        )

    def update_image(self, new_img, name="Edited", push_history=True):
        """Updates the processed image panel with a PIL Image object.
           If push_history is True, the current processed image is saved to history before update.
        """
        # push previous processed state to history for undo
        if push_history and self.processed_img is not None:
            try:
                self.history.append((self.processed_img.copy(), self.proc_name))
                if len(self.history) > self.max_history:
                    self.history.pop(0)
            except Exception:
                pass
        self.processed_img = new_img
        self.proc_name = name
        self.display_processed(self.processed_img)

    def update_proc_info(self):
        if self.processed_img is not None:
            size = self.processed_img.size
            self.proc_info.config(text=f"Processed Image: {self.proc_name} | Size: {size[0]}x{size[1]}")

    def reset_image(self):
        if self.img is not None:
            # allow undo of reset by pushing current state
            if self.processed_img is not None:
                try:
                    self.history.append((self.processed_img.copy(), self.proc_name))
                    if len(self.history) > self.max_history:
                        self.history.pop(0)
                except Exception:
                    pass
            self.processed_img = self.img.copy()
            self.proc_name = self.img_name
            self.display_processed(self.processed_img)

    def apply_brightness_contrast(self, brightness_factor, contrast_factor, slider_win):
        """Applies brightness and contrast changes in real-time and updates the display."""

        # 1. Update Real-time Value Labels
        try:
            self.brightness_label.config(text=f"{brightness_factor:.2f}")
            self.contrast_label.config(text=f"{contrast_factor:.2f}")
        except Exception:
            pass

        # 2. Apply Transformation (using PIL's ImageEnhance for speed)
        img_copy = self.slider_source_img.copy()

        enhancer_b = ImageEnhance.Brightness(img_copy)
        img_bright = enhancer_b.enhance(brightness_factor)

        enhancer_c = ImageEnhance.Contrast(img_bright)
        final_img = enhancer_c.enhance(contrast_factor)

        # 3. Update Processed Panel
        disp_img = final_img.copy()
        disp_img.thumbnail(PANEL_SIZE)
        tk_img = ImageTk.PhotoImage(disp_img)
        self.panel_processed.config(image=tk_img)
        self.panel_processed.image = tk_img

        slider_win.current_result = final_img

    def negative(self):
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        dialog = OperationParameterDialog(
            self.root, 
            "Negative Filter", 
            "N/A", "N/A", 0, 0, # Placeholder defaults
            show_parameters=False # No parameters needed for negative filter
        )
        if dialog.result is None: return

        _, _, source_choice = dialog.result
        arr = self._get_array_for_operation(source_choice)
        if arr is None: return

        if arr.ndim == 3 and arr.shape[2] == 3:
            neg_arr = 255 - arr
            mode = "RGB"
        else:
            max_pixel = 255
            neg_arr = max_pixel - arr
            mode = "L"

        self.update_image(Image.fromarray(np.clip(neg_arr, 0, 255).astype(np.uint8), mode=mode), name="Negative")

    def smoothing(self, filter_type):
        """
        Applies a Mean or Gaussian smoothing filter using a custom dialog for N, I, and source.
        """
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Custom Dialog for Kernel Size (N), Intensity (I), and Source
        dialog = OperationParameterDialog(
            self.root,
            f"{filter_type.title()} Filter",
            "Kernel Size (N)",
            "Intensity (I)",
            "3",
            "1.0",
            is_smoothing=True
        )

        if dialog.result is None: return

        kernel_size_n, intensity, source_choice = dialog.result
        arr = self._get_array_for_operation(source_choice)
        if arr is None: return

        # Convert to grayscale if necessary
        if arr.ndim == 3 and arr.shape[2] == 3:
            img_gray = np.mean(arr, axis=2).astype(np.uint8)
        else:
            img_gray = arr.copy().astype(np.uint8)

        # 1. Determine Kernel
        if filter_type.lower() == "mean":
            kernel = np.ones((kernel_size_n, kernel_size_n)) / (kernel_size_n * kernel_size_n)
            filter_name = "Mean Filter"
        elif filter_type.lower() == "gaussian":
            # For simplicity, we only generate the standard 3x3 Gaussian kernel
            if kernel_size_n != 3:
                messagebox.showwarning("Warning", "Only 3x3 Gaussian kernel is implemented. Using 3x3.")
                kernel_size_n = 3
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
            filter_name = "Gaussian Filter"
        else:
            return

        # 2. Apply Convolution
        smoothed = convolve(img_gray.astype(np.float32), kernel, mode='constant', cval=0.0)

        # 3. Blend with Original (Sharpening/Unsharp-like approach to control blur strength)
        smoothed = (1 - intensity) * img_gray.astype(np.float32) + intensity * smoothed

        # 4. Update Image
        self.update_image(
            Image.fromarray(np.clip(smoothed, 0, 255).astype(np.uint8)),
            name=f"Smoothed ({filter_name} N={kernel_size_n}, I={intensity:.2f})"
        )

    def sharpen(self, choice):
        """
        Applies a Laplacian sharpening filter using a custom dialog for intensity and source.
        Kernel is fixed at 3x3.
        """
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Custom Dialog for Intensity (I) and Source
        dialog = OperationParameterDialog(
            self.root,
            "Sharpening Filter",
            "Kernel Size (3x3 fixed)",
            "Intensity (I)",
            "3", # Default kernel size, even if fixed
            "1.0",
            is_smoothing=False
        )

        if dialog.result is None: return

        _, intensity, source_choice = dialog.result  # Only need intensity and source
        try:
            intensity = float(intensity)
        except ValueError:
            messagebox.showerror("Error", "Invalid intensity value.")
            return
        arr = self._get_array_for_operation(source_choice)
        if arr is None: return

        if arr.ndim == 3 and arr.shape[2] == 3:
            arr_gray = np.mean(arr, axis=2).astype(np.uint8)
        else:
            arr_gray = arr.copy().astype(np.uint8)

        # Fixed 3x3 kernels for standard sharpening
        laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        composite_laplacian_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        if choice.strip() == "1":
            kernel = laplacian_filter
            filter_name = "Laplacian Filter"
            laplace_result = convolve(arr_gray.astype(np.float32), kernel, mode='constant', cval=0.0)
            sharpened = arr_gray.astype(np.float32) + intensity * laplace_result

        elif choice.strip() == "2":
            kernel = composite_laplacian_filter
            filter_name = "Composite Laplacian Filter"
            conv_result = convolve(arr_gray.astype(np.float32), kernel, mode='constant', cval=0.0)
            sharpened = (1 - intensity) * arr_gray.astype(np.float32) + intensity * conv_result

        else:
            messagebox.showerror("Error", "Invalid filter choice.")
            return
         # Clip values to valid pixel range and update image
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        self.update_image(
            Image.fromarray(sharpened),
            name=f"Sharpened ({filter_name}, I={intensity:.2f})"
        )

    def histogram(self):
        """Show original and/or processed image histograms in two subplots as selected."""
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Dialog for source selection
        dialog = OperationParameterDialog(
            self.root,
            "Histogram Source",
            "Show Original?", "Show Processed?",
            True, True,
            show_parameters=False
        )
        if dialog.result is None:
            return

        # Always show both subplots (you can add checkboxes for more control if needed)
        show_original = True
        show_processed = True

        # Prepare arrays
        arr_orig = np.array(self.img.convert("RGB")) if self.img.mode == "RGB" else np.array(self.img.convert("L"))
        arr_proc = None
        if self.processed_img is not None:
            arr_proc = np.array(self.processed_img.convert("RGB")) if self.processed_img.mode == "RGB" else np.array(self.processed_img.convert("L"))

        # Convert to grayscale
        def to_gray(arr):
            if arr is None:
                return None
            if arr.ndim == 3 and arr.shape[2] == 3:
                return np.dot(arr[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            else:
                return arr.copy().astype(np.uint8)

        img_gray_orig = to_gray(arr_orig)
        img_gray_proc = to_gray(arr_proc)

        # Manual histogram calculation
        def manual_histogram(img_gray):
            if img_gray is None:
                return np.zeros(256, dtype=int)
            height, width = img_gray.shape
            histogram = np.zeros(256, dtype=int)
            for i in range(height):
                for j in range(width):
                    pixel_value = img_gray[i, j]
                    histogram[pixel_value] += 1
            return histogram

        hist_orig = manual_histogram(img_gray_orig)
        hist_proc = manual_histogram(img_gray_proc)

        # Plot in two subplots
        plt.figure("Histogram Comparison", figsize=(10, 4))
        if show_original:
            plt.subplot(1, 2, 1)
            plt.bar(range(256), hist_orig, color='gray', alpha=0.7)
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.title("Original Image Histogram")
        if show_processed and img_gray_proc is not None:
            plt.subplot(1, 2, 2)
            plt.bar(range(256), hist_proc, color='blue', alpha=0.7)
            plt.xlabel("Intensity")
            plt.ylabel("Frequency")
            plt.title("Processed Image Histogram")
        plt.tight_layout()
        plt.show()

    def resize(self):
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Use processed_img as default if it exists, otherwise original
        default_w = self.processed_img.width if self.processed_img is not None else self.img.width
        default_h = self.processed_img.height if self.processed_img is not None else self.img.height

        dialog = OperationParameterDialog(
            self.root, 
            "Resize Image", 
            "Width", "Height", 
            default_w, 
            default_h, 
            show_parameters=True # Has parameters (width, height)
        )
        if dialog.result is None: return

        new_w, new_h, source_choice = dialog.result
        img = self._get_image_for_operation(source_choice)
        if img is None: return

        try:
            new_w = int(new_w)
            new_h = int(new_h)

            if new_w > 0 and new_h > 0:
                resized = self.resize_image_manual(img, new_w, new_h)
                self.update_image(resized, name=f"Resized {new_w}x{new_h}")
            else:
                messagebox.showerror("Error", "Width and Height must be positive integers.")
        except ValueError:
            messagebox.showerror("Error", "Invalid input format for width/height.")


    def resize_image_manual(self, image, new_width, new_height):
        original_width, original_height = image.size
        resized_image = Image.new(image.mode, (new_width, new_height))
        for y in range(new_height):
            for x in range(new_width):
                orig_x = int(x * original_width / new_width)
                orig_y = int(y * original_height / new_height)
                orig_x = min(orig_x, original_width - 1)
                orig_y = min(orig_y, original_height - 1)
                resized_image.putpixel((x, y), image.getpixel((orig_x, orig_y)))
        return resized_image

    def threshold(self):
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        dialog = OperationParameterDialog(
            self.root, 
            "Threshold (Binarize)", 
            "Threshold Value (0-255)", "N/A", 
            128, 0, # Default threshold 128
            show_parameters=True, # Has one parameter
            is_smoothing=False # Not a smoothing operation
        )
        if dialog.result is None: return

        thresh, _, source_choice = dialog.result # Only need thresh and source
        arr = self._get_array_for_operation(source_choice)
        if arr is None: return

        try:
            thresh = int(thresh)
        except Exception:
            messagebox.showerror("Error", "Threshold must be an integer between 0 and 255.")
            return

        result = arr.copy()

        if arr.ndim == 2:
            result = np.where(result >= thresh, 255, 0)
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img_gray = np.dot(result[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            thresh_gray = np.where(img_gray >= thresh, 255, 0)
            result = np.stack([thresh_gray]*3, axis=-1)
            mode = "RGB"
        else:
            messagebox.showerror("Error", "Unsupported image format for thresholding.")
            return

        self.update_image(Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode=mode), name=f"Thresholded ({thresh})")

    def intensity_transform(self, choice):
        """Fixed Log and Gamma transforms (vectorized; handles RGB/grayscale)."""
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        # Prepare dialog
        if choice == "1":  # Log
            dialog = OperationParameterDialog(self.root, "Log Transform", "N/A", "N/A", 0, 0, show_parameters=False)
            proc_label = "Log Transformed"
        elif choice == "2":  # Gamma
            dialog = OperationParameterDialog(self.root, "Gamma Transform", "Gamma Value", "N/A", "2.0", 0, show_parameters=True)
            proc_label = "Gamma Transformed"
        else:
            return

        if dialog.result is None:
            return

        param1, _, source_choice = dialog.result

        # get numpy array for selected source (must be numpy array, not PIL Image)
        arr = self._get_array_for_operation(source_choice)
        if arr is None:
            return

        # Work in float for calculations
        arr_f = arr.astype(np.float32)

        try:
            if choice == "1":
                # Log transform: s = c * log(1 + r)
                max_val = np.max(arr_f)
                if max_val <= 0:
                    messagebox.showerror("Error", "Image has no positive pixel values for log transform.")
                    return
                c = 255.0 / np.log(1.0 + max_val)
                res_f = c * np.log(1.0 + arr_f)

            else:  # Gamma
                gamma = float(param1)
                if gamma <= 0:
                    messagebox.showerror("Error", "Gamma must be > 0.")
                    return
                res_f = ((arr_f / 255.0) ** gamma) * 255.0

        except Exception as e:
            messagebox.showerror("Error", f"Transform failed: {e}")
            return

        # Clip and convert to uint8
        res_uint8 = np.clip(res_f, 0, 255).astype(np.uint8)

        # Determine PIL mode and update
        if res_uint8.ndim == 3 and res_uint8.shape[2] == 3:
            mode = "RGB"
        else:
            mode = "L"

        self.proc_name = proc_label if choice == "1" else f"Gamma={param1}"
        self.update_image(Image.fromarray(res_uint8, mode=mode))

    def edge_detection(self, method):
        if self.img is None:
            messagebox.showerror("Error", "No image loaded!")
            return

        method = method.lower()
        intensity = 1.0  # Default intensity for edge detection

        if method in ["sobel", "prewitt", "roberts"]:
            # Only ask for intensity and source, not kernel size
            dialog = OperationParameterDialog(
                self.root,
                f"{method.title()} Edge Detection",
                "Intensity (I)", "N/A",
                "1.0", "0",
                is_smoothing=False,
                show_parameters=True
            )
            if dialog.result is None: return
            intensity, _, source_choice = dialog.result
            try:
                intensity = float(intensity)
            except Exception:
                intensity = 1.0
            arr = self._get_array_for_operation(source_choice)
            if arr is None: return
        elif method == "canny":
            dialog = OperationParameterDialog(
                self.root,
                "Canny Edge Detection ",
                "N/A", "N/A", 0, 0,
                show_parameters=False
            )
            if dialog.result is None: return
            _, _, source_choice = dialog.result
            arr = self._get_array_for_operation(source_choice)
            if arr is None: return
            messagebox.showwarning("Canny Not Implemented", "Canny Edge Detection")
            return
        else:
            arr = self._get_array_for_operation("processed")
            if arr is None: return

        # Convert to grayscale if needed
        if arr.ndim == 3:
            arr_gray = np.mean(arr, axis=2).astype(np.uint8)
        else:
            arr_gray = arr.astype(np.uint8)

        # Manual convolution using loops
        if method == "sobel":
            gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            ksize = 3
        elif method == "prewitt":
            gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            ksize = 3
        elif method == "roberts":
            gx = np.array([[1, 0], [0, -1]])
            gy = np.array([[0, 1], [-1, 0]])
            ksize = 2
        else:
            messagebox.showinfo("Error", "Invalid method name.")
            return

        pad = ksize // 2
        padded = np.pad(arr_gray, pad, mode='constant')
        result = np.zeros_like(arr_gray, dtype=np.float32)
        for i in range(pad, padded.shape[0] - pad):
            for j in range(pad, padded.shape[1] - pad):
                region = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
                if method == "roberts":
                    gx_val = np.sum(region[:2, :2] * gx)
                    gy_val = np.sum(region[:2, :2] * gy)
                else:
                    gx_val = np.sum(region * gx)
                    gy_val = np.sum(region * gy)
                result[i - pad, j - pad] = np.sqrt(gx_val ** 2 + gy_val ** 2)
        edges = intensity * result
        edges = np.clip(edges, 0, 255).astype(np.uint8)
        # Match RGB size if needed
        if arr.ndim == 3 and arr.shape[2] == 3:
            edges = np.stack([edges] * 3, axis=-1)
        self.proc_name = f"Edge Detection ({method.title()})"
        self.update_image(Image.fromarray(edges, mode="RGB" if arr.ndim == 3 else "L"))

    def undo_last_change(self, event=None):
        """Restore last processed image from history."""
        if not hasattr(self, "history") or len(self.history) == 0:
            messagebox.showinfo("Undo", "No actions to undo.")
            return
        try:
            last_img, last_name = self.history.pop()
            # set processed image to previous state (do not push current into history)
            self.processed_img = last_img
            self.proc_name = last_name
            self.display_processed(self.processed_img)
            self.update_proc_info()
        except Exception as e:
            messagebox.showerror("Undo Failed", f"Could not undo: {e}")

if __name__ == "__main__":
    # Ensure matplotlib is set to a GUI backend if available, otherwise fallback safely
    try:
        plt.switch_backend('TkAgg')
    except Exception:
        try:
            plt.switch_backend('Agg')
        except Exception:
            pass

    root = tk.Tk()
    app = ImageToolkit(root)  # Initialize the class instance 'app'
    root.mainloop()  # Start the main event loop
