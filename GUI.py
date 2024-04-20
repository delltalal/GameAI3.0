import tkinter as tk
import tkinter.messagebox as messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import configparser  # Import the configparser module
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tkinter import OptionMenu, StringVar
import subprocess
import cv2
import shutil


class GameAIInterface:

    def __init__(self, root):
        self.root = root
        self.root.title("GameAI Interface")
        self.root.configure(bg="#2e4b6e")  # Set background color

        self.texture_preview_window = None
        self.current_page = 0
        self.previews_per_page = 20
        self.page_number = StringVar()

        self.texture_directory = ""  # Initialize with an empty string
        self.fmv_file = ""
        self.tile_height = ""
        self.tile_width = ""
        self.general_threshold = ""

        self.img_height = 32
        self.img_width = 32
        self.class_names = ['Asphalt', 'Bark', 'Branch', 'Brick', 'Coal', 'Concrete', 'Creature', 'Debris', 'Fabric',
                            'Fur', 'Grass', 'Gravel', 'Marble', 'Moss', 'Rock', 'Roofing', 'Sand',
                            'Snow', 'Tile', 'Wood']
        #self.class_names = ['Blood', 'Brick', 'Cartoon', 'Concrete', 'Concrete_Painted', 'Covers', 'Cracks',
        #                    'Decorative', 'Dirt', 'Door', 'Fabric', 'Faces', 'Fingerprints', 'Graffiti', 'Grates',
        #                    'Ground', 'Ground_Grass', 'Hair', 'Icons', 'Leaves', 'Metal', 'Metal_Diamond-Metal',
        #                    'Plaster', 'Plaster_Damaged-Plaster', 'Rock', 'Rust', 'Rust_Rusted-Paint', 'Sand',
        #                    'Signs', 'Sprites', 'Stone', 'Stone_Stone-Walls', 'Text', 'VFX', 'Wood', 'Wood_Bark',
        #                    'Wood_FibreBoard', 'Wood_Painted', 'Wood_Planks', 'Wood_Shutters']
        #self.model = keras.models.load_model('models/GameAI_40classes_classifier_savedmodel.h5')
        self.model = keras.models.load_model('models/classifier.h5')

        self.texture_labels = {}  # Dictionary to store texture labels and buttons
        self.texture_model_selections = {}  # Dictionary to store the model selections for each texture
        self.texture_skip_states = {}  # Dictionary to store skip states for each texture

        self.config = configparser.ConfigParser()  # Initialize the configparser
        self.load_settings()  # Load settings from the configuration file
        self.create_ui()

    def create_ui(self):

        self.texture_frame = tk.Frame(self.root, bg="#2e4b6e")
        self.texture_frame.pack(pady=20)

        self.fmv_frame = tk.Frame(self.root, bg="#2e4b6e")
        self.fmv_frame.pack()

        # Create a frame to hold both buttons and align them
        button_frame = tk.Frame(self.root, bg="#2e4b6e")
        button_frame.pack(pady=20)

        button_width = 25  # Set the width for both buttons

        self.texture_button = tk.Button(button_frame, text="Select Textures Directory",
                                        command=self.browse_texture_directory, bg="#2e4b6e", fg="white",
                                        width=button_width)
        self.texture_button.pack(side="left", padx=10, pady=10)

        self.fmv_button = tk.Button(button_frame, text="Select an FMV to Upscale", command=self.browse_fmv_file,
                                    bg="#2e4b6e", fg="white", width=button_width)
        self.fmv_button.pack(side="left", padx=10, pady=10)

        self.settings_button = tk.Button(self.root, text="Settings", command=self.open_settings, bg="#2e4b6e",
                                         fg="white")
        self.settings_button.pack(pady=20)

        self.root.geometry("800x400")

    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.configure(bg="#2e4b6e")

        height_label = tk.Label(settings_window, text="Tile Height:", bg="#2e4b6e", fg="white")
        height_label.pack(padx=10, pady=10)
        height_entry = tk.Entry(settings_window, bg="#2e4b6e", fg="white")
        height_entry.insert(0, str(self.tile_height))  # Load saved value
        height_entry.pack(padx=10, pady=10)

        width_label = tk.Label(settings_window, text="Tile Width:", bg="#2e4b6e", fg="white")
        width_label.pack(padx=10, pady=10)
        width_entry = tk.Entry(settings_window, bg="#2e4b6e", fg="white")
        width_entry.insert(0, str(self.tile_width))  # Load saved value
        width_entry.pack(padx=10, pady=10)

        threshold_label = tk.Label(settings_window, text="General Upscaler Threshold:", bg="#2e4b6e", fg="white")
        threshold_label.pack(padx=10, pady=10)
        threshold_entry = tk.Entry(settings_window, bg="#2e4b6e", fg="white")
        threshold_entry.insert(0, str(self.general_threshold))  # Load saved value
        threshold_entry.pack(padx=10, pady=10)

        transparency_fix_label = tk.Label(settings_window, text="Fix Transparency", bg="#2e4b6e", fg="white")
        transparency_fix_label.pack(padx=10, pady=10)
        self.transparency_fix_var = tk.BooleanVar(value=self.transparency_fix)  # Initialize with loaded value
        transparency_fix_checkbox = tk.Checkbutton(settings_window, variable=self.transparency_fix_var, bg="#2e4b6e",
                                                   fg="black", selectcolor="#2e4b6e")
        transparency_fix_checkbox.pack(padx=10, pady=10)

        save_button = tk.Button(settings_window, text="Save",
                                command=lambda: self.save_settings(height_entry, width_entry, threshold_entry),
                                bg="#2e4b6e", fg="white")
        save_button.pack(pady=20)

        settings_window.geometry("400x400")

    def save_settings(self, height_entry, width_entry, threshold_entry):
        try:
            self.tile_height = int(height_entry.get())
            self.tile_width = int(width_entry.get())
            threshold_value = float(threshold_entry.get())
            self.transparency_fix = self.transparency_fix_var.get()

            if 100 >= threshold_value > 0:  # Ensure threshold is within the valid range
                self.general_threshold = threshold_value

                # Save the settings to the configuration file
                self.config['Settings'] = {
                    'TileHeight': str(self.tile_height),
                    'TileWidth': str(self.tile_width),
                    'GeneralThreshold': str(self.general_threshold),
                    'TransparencyFix': str(self.transparency_fix)
                }

                with open('settings.ini', 'w') as configfile:
                    self.config.write(configfile)

                print("Tile Height:", self.tile_height)
                print("Tile Width:", self.tile_width)
                print("General Upscaler Threshold:", self.general_threshold)

            elif threshold_value <= 0:
                messagebox.showinfo("Invalid Threshold", "Threshold value must be greater than 0.")
            else:
                messagebox.showinfo("Threshold Exceeded", "Threshold value cannot exceed 100.")

        except ValueError:
            print("Invalid input for tile size or threshold")

    def load_settings(self):
        if os.path.exists('settings.ini'):
            self.config.read('settings.ini')
            if 'Settings' in self.config:
                settings = self.config['Settings']
                self.tile_height = int(settings.get('TileHeight', ''))
                self.tile_width = int(settings.get('TileWidth', ''))
                self.general_threshold = float(settings.get('GeneralThreshold', ''))
                self.transparency_fix = self.config.getboolean('Settings', 'TransparencyFix', fallback=False)

    def browse_texture_directory(self):
        self.texture_directory = filedialog.askdirectory(title="Select Texture Directory")
        if self.texture_directory:
            print("Selected Texture Directory:", self.texture_directory)

            # Clear previous texture data
            self.texture_labels = {}
            self.texture_model_selections = {}

            self.load_texture_previews()
            self.auto_click_next_page()  # Add this line to start auto-clicking

    def auto_click_next_page(self):
        # Check if you are on the last page
        last_page = self.get_last_page()
        if self.current_page < last_page:
            self.show_next_page()
        else:
            # If on the last page, go back to page 0 (first page)
            self.current_page = 0
            self.load_texture_previews()
            return  # Add a return statement to stop the function

        self.root.after(0, self.auto_click_next_page)

    def load_texture_previews(self):
        if self.texture_preview_window:
            self.texture_preview_window.destroy()
        self.create_texture_preview_window()
        self.page_number.set(f"{self.current_page + 1}/{self.get_last_page() + 1}")

    def show_next_page(self):
        self.current_page += 1
        last_page = self.get_last_page()
        if self.current_page > last_page:
            self.current_page = last_page
        self.load_texture_previews()
        # Save the selected texture classes for the current page
        self.save_selected_texture_classes()
        self.save_current_texture_states()

    def save_selected_texture_classes(self):
        # Iterate over all textures and save their selected classes
        for filename in os.listdir(self.texture_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dds', '.tga')):
                class_var = self.texture_labels.get(filename, {}).get("class_var")
                if class_var:
                    selected_model = class_var.get()
                    if selected_model and selected_model != "Select Model":
                        self.texture_model_selections[filename] = selected_model

    def save_current_texture_states(self):
        for filename, labels in self.texture_labels.items():
            skip_var = labels.get("skip_var")
            if skip_var:
                self.texture_skip_states[filename] = skip_var.get()

    def show_prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
        self.load_texture_previews()
        self.save_current_texture_states()

    def close_texture_preview_window(self):
        if self.texture_preview_window:
            self.texture_preview_window.destroy()
            self.texture_preview_window = None

    def create_texture_preview_window(self):
        self.texture_preview_window = tk.Toplevel(self.root)
        self.texture_preview_window.protocol("WM_DELETE_WINDOW", self.texture_preview_window.destroy)

        self.texture_preview_window.title("Texture Previews")

        # Make the window fullscreen
        self.texture_preview_window.attributes('-fullscreen', True)

        self.texture_preview_window.canvas = tk.Canvas(self.texture_preview_window, bg="#2e4b6e")
        self.texture_preview_window.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(self.texture_preview_window, orient="vertical",
                                 command=self.texture_preview_window.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.texture_preview_window.canvas.configure(yscrollcommand=scrollbar.set)

        self.texture_preview_window.inner_frame = tk.Frame(self.texture_preview_window.canvas, bg="#2e4b6e")
        self.texture_preview_window.canvas.create_window((0, 0), window=self.texture_preview_window.inner_frame,
                                                         anchor="nw")

        # Bind mouse wheel event to canvas scrolling
        self.texture_preview_window.canvas.bind_all("<MouseWheel>", self.on_canvas_scroll)
        self.texture_preview_window.inner_frame.bind("<Configure>", self.on_frame_configure)

        # Adding an Exit button
        exit_button = tk.Button(self.texture_preview_window, text="Exit", command=self.close_texture_preview_window,
                                bg="#2e4b6e", fg="white")
        exit_button.pack(side="bottom", padx=10, pady=10)

        preview_size = 256  # Set the desired size for texture previews (both width and height)

        start_index = self.current_page * self.previews_per_page
        end_index = start_index + self.previews_per_page

        for index, filename in enumerate(sorted(os.listdir(self.texture_directory))):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dds', '.tga')):
                if start_index <= index < end_index:
                    texture_path = os.path.join(self.texture_directory, filename)
                    img = Image.open(texture_path)

                    # Calculate the scaling factor to fit the image within the preview size
                    scaling_factor = min(preview_size / img.width, preview_size / img.height)

                    # Resize the image to fit completely within the preview size while preserving the aspect ratio
                    img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)))

                    # Calculate the position to center the image within the square canvas
                    x_offset = (preview_size - img.width) // 2
                    y_offset = (preview_size - img.height) // 2

                    # Create a blank square canvas
                    preview_img = Image.new("RGB", (preview_size, preview_size), color="#2e4b6e")
                    preview_img.paste(img, (x_offset, y_offset))

                    preview_img = ImageTk.PhotoImage(preview_img)

                    texture_frame = tk.Frame(self.texture_preview_window.inner_frame, bg="#2e4b6e")
                    texture_frame.pack(side="top", padx=10, pady=10, fill="x")

                    label = tk.Label(texture_frame, image=preview_img, bg="#2e4b6e")
                    label.image = preview_img
                    label.pack(side="left")

                    # Inside the loop that loads texture previews
                    texture_class_label = tk.Label(texture_frame, text="Texture Class", bg="#2e4b6e", fg="white")
                    texture_class_label.pack(side="left", padx=10)

                    class_var = tk.StringVar()
                    class_var.set("Select Model")  # Set the initial value for the OptionMenu

                    class_button = OptionMenu(texture_frame, class_var, *self.class_names)
                    class_button.config(bg="#2e4b6e", fg="white")
                    class_button.pack(side="left", padx=10)

                    upscale_factor_label = tk.Label(texture_frame, text="Upscale Factor", bg="#2e4b6e", fg="white")
                    upscale_factor_label.pack(side="left", padx=10)
                    # Upscale factor options
                    # upscale_options = ["x4", "x16"]
                    upscale_options = ["x4"]
                    upscale_factor_var = tk.StringVar()
                    upscale_factor_var.set(upscale_options[0])  # default value is "x4"
                    upscale_factor_menu = tk.OptionMenu(texture_frame, upscale_factor_var, *upscale_options)
                    upscale_factor_menu.config(bg="#2e4b6e", fg="white")
                    upscale_factor_menu.pack(side="left", padx=10)

                    skip_label = tk.Label(texture_frame, text="Skip", bg="#2e4b6e", fg="white")
                    skip_label.pack(side="left", padx=10)
                    skip_var = tk.BooleanVar(value=self.texture_skip_states.get(filename, False))
                    skip_checkbox = tk.Checkbutton(texture_frame, variable=skip_var, bg="#2e4b6e",
                                                   fg="black", selectcolor="#2e4b6e")
                    skip_checkbox.pack(side="left", padx=10)

                    separator_frame = tk.Frame(self.texture_preview_window.inner_frame, bg="white")
                    separator_frame.pack(side="top", fill="both", expand=True)

                    # Store the label and button in the dictionary
                    self.texture_labels[filename] = {"class_label": texture_class_label, "class_var": class_var,
                                                     "skip_var": skip_var}

                    # Initialize the model selection based on the dictionary
                    if filename in self.texture_model_selections:
                        class_var.set(self.texture_model_selections[filename])
                    else:
                        class_var.set("Select Model")  # Set the initial value for the OptionMenu

                    if filename in self.texture_skip_states:
                        skip_var.set(self.texture_skip_states[filename])
                    else:
                        skip_var.set(False)  # Default value for the skip checkbox

        next_button = tk.Button(self.texture_preview_window, text="Next Page", bg="#2e4b6e", fg="white",
                                command=self.show_next_page)
        next_button.pack(side="bottom", padx=10, pady=10)

        prev_button = tk.Button(self.texture_preview_window, text="Previous Page", bg="#2e4b6e", fg="white",
                                command=self.show_prev_page)
        prev_button.pack(side="bottom", padx=10, pady=10)

        upscale_button = tk.Button(self.texture_preview_window, text="Extract Texture Materials", bg="#2e4b6e",
                                   fg="white", command=self.extract_texture_materials)
        upscale_button.pack(side="bottom", padx=10, pady=10)

        upscale_button = tk.Button(self.texture_preview_window, text="Upscale Texture Materials", bg="#2e4b6e",
                                   fg="white")
        # ,command=self.upscale_texture_materials
        upscale_button.pack(side="bottom", padx=10, pady=10)

        classify_button = tk.Button(self.texture_preview_window, text="Classify Full Textures", bg="#2e4b6e",
                                    fg="white", command=self.classify_textures)
        classify_button.pack(side="bottom", padx=10, pady=10)

        upscale_button = tk.Button(self.texture_preview_window, text="Upscale Full Textures", bg="#2e4b6e", fg="white",
                                   command=self.upscale_textures)
        upscale_button.pack(side="bottom", padx=10, pady=10)

        self.page_number_label = tk.Label(self.texture_preview_window, textvariable=self.page_number, bg="#2e4b6e",
                                          fg="white")  # Page number label
        self.page_number_label.pack(side="bottom", padx=10, pady=10)
        self.page_number_label.bind("<Button-1>", self.edit_page_number)  # Bind a click event to edit the page number

    def edit_page_number(self, event):
        page_number_str = self.page_number.get().split('/')[0]  # Get the current page number from the label
        self.page_number_label.pack_forget()  # Remove the label temporarily
        self.page_number_entry = tk.Entry(self.texture_preview_window, bg="#2e4b6e", fg="white")
        self.page_number_entry.insert(0, page_number_str)  # Set the initial text to the current page number
        self.page_number_entry.pack(side="bottom", padx=10, pady=10)
        self.page_number_entry.focus_set()  # Set focus to the entry field
        self.page_number_entry.bind("<FocusOut>", self.update_page_number)  # Bind an event to update the page number

    def update_page_number(self, event):
        new_page_number_str = self.page_number_entry.get()
        try:
            new_page_number = int(new_page_number_str)
            last_page = self.get_last_page()
            if 1 <= new_page_number <= last_page + 1:
                self.current_page = new_page_number - 1
                self.load_texture_previews()
            else:
                messagebox.showinfo("Invalid Page", f"Page number should be between 1 and {last_page + 1}")
        except ValueError:
            messagebox.showinfo("Invalid Input", "Please enter a valid page number.")

        self.page_number_entry.pack_forget()
        self.page_number.set(f"{self.current_page + 1}/{self.get_last_page() + 1}")
        self.page_number_label.pack(side="bottom", padx=10, pady=10)

    def get_last_page(self):
        num_texture_files = len([filename for filename in os.listdir(self.texture_directory) if
                                 filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dds', '.tga'))])
        return (num_texture_files - 1) // self.previews_per_page

    def on_frame_configure(self, event):
        self.texture_preview_window.canvas.configure(scrollregion=self.texture_preview_window.canvas.bbox("all"))

    def on_canvas_scroll(self, event):
        # Scroll the canvas when the mouse wheel is used
        self.texture_preview_window.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def browse_fmv_file(self):
        self.fmv_file = filedialog.askopenfilename(title="Select FMV File", filetypes=(
            ("Video files", "*.mp4 *.avi *.mkv"), ("All files", "*.*")))
        print("Selected FMV File:", self.fmv_file)

        if self.fmv_file:
            # Clear out the 'tmp' directory
            if os.path.exists('tmp'):
                shutil.rmtree('tmp')
            os.makedirs('tmp', exist_ok=True)
            os.makedirs("tmp/audio", exist_ok=True)
            os.makedirs("tmp/in_frames", exist_ok=True)

            # Extract the base name of the selected video file
            self.selected_video_base_name = os.path.splitext(os.path.basename(self.fmv_file))[0]
            video = cv2.VideoCapture(self.fmv_file)
            if video.isOpened():

                # Get the frame rate of the video
                self.fps = video.get(cv2.CAP_PROP_FPS)
                print(f"Frame Rate of the Video: {self.fps} FPS")

                # Create necessary directories if they don't exist
                os.makedirs("tmp/audio", exist_ok=True)
                os.makedirs("tmp/in_frames", exist_ok=True)

                # Extract audio from video
                extract_audio_command = f'ffmpeg -i "{self.fmv_file}" "tmp/audio/output.wav"'
                # Extract video frames
                ffmpeg_command = f'ffmpeg -i "{self.fmv_file}" -q:v 1 -vf "fps={self.fps}" tmp/in_frames/frame%08d.png'

                try:
                    # Extract Audio
                    subprocess.run(extract_audio_command, shell=True, check=True)
                    print("Audio extracted successfully.")
                    # Extract Frames
                    subprocess.run(ffmpeg_command, shell=True, check=True)
                    print("FMV file has been successfully converted into PNG frames.")
                    # Execute Real-Esrgan Interface
                    self.upscale_video_frames()
                    # Remove _out from tmp/out_frames PNGs suffix
                    self.remove_suffix()
                    # Merge video frames with and without audio
                    self.merge_frames_to_video()
                except subprocess.CalledProcessError as e:
                    print(f"Error during FMV file processing: {e}")

                video.release()  # Release the video file

    def upscale_textures(self):
        input_dir = self.texture_directory
        output_dir = "output"
        processing_dir = os.path.join('tmp')
        os.makedirs(processing_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        for filename, labels in self.texture_labels.items():
            # Check if 'skip' is selected for the texture
            skip_var = labels.get("skip_var")
            if skip_var and skip_var.get():
                print(f"Skipping texture: {filename} (marked as 'skip')")
                continue  # Skip this texture

            class_var = labels["class_var"].get()
            texture_path = os.path.join(input_dir, filename)
            rgb_path, alpha_path = None, None

            # Handle transparency and save RGB and Alpha parts
            if self.transparency_fix and Image.open(texture_path).mode == 'RGBA':
                img = Image.open(texture_path)
                rgb_img, alpha_img = img.split()[:3], img.split()[3]
                rgb_img = Image.merge("RGB", rgb_img)
                alpha_img = alpha_img.resize((alpha_img.width * 4, alpha_img.height * 4), Image.Resampling.BILINEAR)
                alpha_path = os.path.join(processing_dir, f'{filename}_alpha.png')
                alpha_img.save(alpha_path)
                rgb_path = os.path.join(processing_dir, f'{filename}_rgb.png')
                rgb_img.save(rgb_path)
                texture_path = rgb_path

            # Attempt to crop the image
            cropped = False
            crop_command = f'convert -crop {self.tile_width}x{self.tile_height} "{texture_path}" "{processing_dir}/{filename}_cropped_%d.png"'
            try:
                subprocess.run(crop_command, shell=True, check=True)
                cropped = any(
                    f.startswith(f'{filename}_cropped_') and f.endswith('.png') for f in os.listdir(processing_dir))
            except subprocess.CalledProcessError as e:
                print(f"Error during image cropping: {e}")

            # Process the cropped files if cropping happened, otherwise upscale the RGB image
            if cropped:
                for cropped_file in os.listdir(processing_dir):
                    if cropped_file.startswith(f'{filename}_cropped_') and cropped_file.endswith('.png'):
                        cropped_file_path = os.path.join(processing_dir, cropped_file)
                        if class_var in self.class_names:
                            upscale_command = f'python inference_realesrgan.py -n {class_var}x4 -i "{cropped_file_path}" -o "{processing_dir}"'
                            try:
                                subprocess.run(upscale_command, shell=True, check=True)
                                print(f"Upscaling cropped texture '{cropped_file}' using class '{class_var}'")
                            except subprocess.CalledProcessError as e:
                                print(f"Error upscaling cropped texture '{cropped_file}': {e}")

                # Merge the upscaled cropped images
                merge_tiles_command = f'montage "{processing_dir}/{filename}_cropped_*_out.png" -tile x -geometry +0+0 "{processing_dir}/{filename}_result.png"'
                try:
                    subprocess.run(merge_tiles_command, shell=True, check=True)
                    print(f"Merged cropped textures for '{filename}'")
                except subprocess.CalledProcessError as e:
                    print(f"Error during merging cropped textures for '{filename}': {e}")
            else:
                # Upscale the entire RGB image if not cropped
                upscale_command = f'python inference_realesrgan.py -n {class_var}x4 -i "{rgb_path}" -o "{processing_dir}"'
                try:
                    subprocess.run(upscale_command, shell=True, check=True)
                    print(f"Upscaled full RGB texture for '{filename}'")
                except subprocess.CalledProcessError as e:
                    print(f"Error upscaling full RGB texture '{filename}': {e}")

            # Combine with Alpha Channel if it exists
            output_file = f"{filename}_result.png" if cropped else f"{os.path.splitext(filename)[0]}_rgb_out.png"
            if alpha_path:
                output_filename = os.path.splitext(filename)[
                                      0] + '.png'  # Ensure the filename has only one .png extension
                combined_image_path = os.path.join(output_dir, output_filename)
                combine_command = f'convert "{processing_dir}/{output_file}" "{alpha_path}" -alpha Off -compose CopyOpacity -composite "{combined_image_path}"'
                try:
                    subprocess.run(combine_command, shell=True, check=True)
                    print(f"Combined RGB and Alpha for '{filename}'")
                except subprocess.CalledProcessError as e:
                    print(f"Error combining RGB and Alpha for '{filename}': {e}")

            # Clear out the 'tmp' directory
            shutil.rmtree(processing_dir)
            os.makedirs(processing_dir, exist_ok=True)

    def extract_texture_materials(self):
        input_dir = self.texture_directory
        output_dir = "tmp"
        os.makedirs(output_dir, exist_ok=True)
        scale_factor = 8

        for filename, labels in self.texture_labels.items():
            # Check if 'skip' is selected for the texture
            skip_var = labels.get("skip_var")
            if skip_var and skip_var.get():
                print(f"Skipping texture: {filename} (marked as 'skip')")
                continue  # Skip this texture

            texture_path = os.path.join(input_dir, filename)
            texture_processing_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(texture_processing_dir, exist_ok=True)

            # Handle transparency and save RGB and Alpha parts
            if self.transparency_fix and Image.open(texture_path).mode == 'RGBA':
                img = Image.open(texture_path)
                rgb_img, alpha_img = img.split()[:3], img.split()[3]
                rgb_img = Image.merge("RGB", rgb_img)
                alpha_img = alpha_img.resize((alpha_img.width * 4, alpha_img.height * 4), Image.Resampling.BILINEAR)
                alpha_path = os.path.join(texture_processing_dir, f'{filename}_alpha.png')
                alpha_img.save(alpha_path)
                rgb_path = os.path.join(texture_processing_dir, f'{filename}_rgb.png')
                rgb_img.save(rgb_path)
                texture_path = rgb_path

            # Read the original dimensions of the texture
            with Image.open(texture_path) as img:
                original_width, original_height = img.size

            # Calculate new dimensions for cropping
            crop_width = original_width // scale_factor
            crop_height = original_height // scale_factor

            # Attempt to crop the image
            cropped = False
            crop_command = f'convert -crop {crop_width}x{crop_height} +repage "{texture_path}" "{texture_processing_dir}/{filename}_cropped_%08d.png"'

            try:
                subprocess.run(crop_command, shell=True, check=True)
                cropped_files = [f for f in os.listdir(texture_processing_dir) if
                                 f.startswith(f'{filename}_cropped_') and f.endswith('.png')]
                cropped = len(cropped_files) > 0
            except subprocess.CalledProcessError as e:
                print(f"Error during image cropping: {e}")

            if cropped:
                cropped_count = len(cropped_files)
                print(f"Number of cropped images for '{filename}': {cropped_count}")  # Print the count

                classified_tiles = {}
                for cropped_file in cropped_files:
                    cropped_file_path = os.path.join(texture_processing_dir, cropped_file)
                    img = tf.keras.utils.load_img(cropped_file_path, target_size=(self.img_height, self.img_width))
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, axis=0)
                    predictions = self.model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])
                    predicted_class = self.class_names[np.argmax(score)]

                    class_dir = os.path.join(texture_processing_dir, predicted_class)
                    os.makedirs(class_dir, exist_ok=True)
                    new_file_path = os.path.join(class_dir, cropped_file)
                    os.rename(cropped_file_path, new_file_path)
                    classified_tiles.setdefault(predicted_class, []).append(new_file_path)

                for class_name, files in classified_tiles.items():
                    class_dir = os.path.join(texture_processing_dir, class_name)
                    result_path = os.path.join(class_dir, f'{filename}_result.png')  # Construct the path first

                    # Calculate the number of white images to generate
                    num_white_images = cropped_count - len(files)

                    # Get the existing indices from files
                    existing_indices = set(int(file.split("_")[-1].split(".")[0]) for file in files)

                    # Generate white images for missing indices
                    missing_indices = [i for i in range(cropped_count) if i not in existing_indices]

                    for i in missing_indices:
                        white_img = Image.new('RGB', (crop_width, crop_height), color=(255, 255, 255))
                        white_img_path = os.path.join(class_dir, '{0}_cropped_{1:08d}.png'.format(filename, i))
                        white_img.save(white_img_path)

                    merge_tiles_command = f'montage "{class_dir}/{filename}_cropped_*.png" -tile x -geometry +0+0 "{result_path}"'  # Use the constructed path
                    #print(merge_tiles_command)
                    try:
                        subprocess.run(merge_tiles_command, shell=True, check=True)
                        print(f"Merged cropped textures for '{class_name}' in '{filename}'")
                    except subprocess.CalledProcessError as e:
                        print(f"Error during merging cropped textures for '{class_name}' in '{filename}': {e}")

                # Clear the processing directory for the next texture
                # for file in os.listdir(texture_processing_dir):
                    # shutil.move(os.path.join(texture_processing_dir, file), output_dir)

                # Delete the unique texture processing directory
                # shutil.rmtree(texture_processing_dir)

    def merge_and_save_images(self, filename, processing_dir, output_dir):
        base_name = os.path.splitext(filename)[0]
        rgb_image_path = os.path.join(processing_dir, f'{filename}_rgb_out.png')
        alpha_image_path = os.path.join(processing_dir, f'{filename}_alpha.png')

        # Updated output image path to use the original filename only
        output_image_path = os.path.join(output_dir, filename)

        if os.path.exists(rgb_image_path) and os.path.exists(alpha_image_path):
            rgb_image = Image.open(rgb_image_path).convert("RGB")
            alpha_image = Image.open(alpha_image_path).convert("L")
            merged_image = Image.merge("RGBA", (*rgb_image.split(), alpha_image))
            merged_image.save(output_image_path)
            print(f"Merged and saved: {output_image_path}")
        else:
            print(f"RGB or Alpha image not found for {filename}")

    def classify_textures(self):
        if self.texture_directory:
            for filename in os.listdir(self.texture_directory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dds', '.tga')):
                    texture_path = os.path.join(self.texture_directory, filename)
                    img = Image.open(texture_path)
                    img = tf.keras.utils.load_img(texture_path, target_size=(self.img_height, self.img_width))
                    img_rgb = img.convert("RGB")
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, axis=0)

                    predictions = self.model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])

                    predicted_class = self.class_names[np.argmax(score)]
                    confidence = 100 * np.max(score)

                    # Update the class button text and store the model selection
                    if filename in self.texture_labels:
                        if confidence > self.general_threshold:
                            selected_model = predicted_class
                            self.texture_labels[filename]["class_var"].set(selected_model)
                            # Store the selected model in the dictionary
                            self.texture_model_selections[filename] = selected_model
                        else:
                            selected_model = "General"
                            self.texture_labels[filename]["class_var"].set(selected_model)
                            # Store the selected model in the dictionary
                            self.texture_model_selections[filename] = selected_model

                    print(
                        f"Texture '{filename}' is classified as "
                        f"{predicted_class} with a {confidence:.2f} percent confidence."
                    )

                    category = self.class_names[np.argmax(score)]

    def upscale_video_frames(self):
        video_upscale_command = f'python inference_realesrgan.py -n PS2VideoV2x4 -i tmp/in_frames -o tmp/out_frames'
        subprocess.run(video_upscale_command, shell=True, check=True)

    def remove_suffix(self):
        out_frames_dir = "tmp/out_frames"
        for filename in os.listdir(out_frames_dir):
            if filename.endswith("_out.png"):
                old_file_path = os.path.join(out_frames_dir, filename)
                new_file_name = filename.replace("_out", "")
                new_file_path = os.path.join(out_frames_dir, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{old_file_path}' to '{new_file_path}'")

    def merge_frames_to_video(self):
        output_video_path = f"F:/AI/GameAI3.0/output/{self.selected_video_base_name}_output.mp4"
        final_video_path = f"F:/AI/GameAI3.0/output/{self.selected_video_base_name}_final.mp4"
        frames_path = "F:/AI/GameAI3.0/tmp/out_frames/frame%08d.png"
        audio_path = "tmp/audio/output.wav"  # Path to the extracted audio file

        # Command to merge frames into a video
        frames_to_video_command = f'ffmpeg -framerate {self.fps} -i "{frames_path}" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "{output_video_path}"'

        # Command to add audio to the video
        add_audio_command = f'ffmpeg -i "{output_video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental "{final_video_path}"'

        try:
            subprocess.run(frames_to_video_command, shell=True, check=True)
            print("Video frames merged successfully.")
            subprocess.run(add_audio_command, shell=True, check=True)
            print("Audio and video merged successfully.")

            # Clear out the 'tmp' directory
            shutil.rmtree('tmp')
            print("Temporary files cleared.")

        except subprocess.CalledProcessError as e:
            print(f"Error during video processing: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GameAIInterface(root)
    root.mainloop()
