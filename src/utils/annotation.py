"""
Speech to text annotation with 
GUI and bounding box drawing ability
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk
import speech_recognition as sr
import json
from src.utils.image_loader import ImageLoader

class Annotator:
    def __init__(self, root=None):
        """Initialize the Annotator with a tkinter root window"""
        if root is None:
            self.root = tk.Tk()
            self.root.title("Image Annotation Tool")
        else:
            self.root = root
            
        self.current_image_index = 0
        self.images = []
        self.image_paths = []
        self.annotations = {}
        self.current_boxes = []
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface components"""
        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for buttons
        self.button_panel = tk.Frame(self.main_frame)
        self.button_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Load images button
        self.load_button = tk.Button(self.button_panel, text="Load Images", 
                                    command=self.load_images, width=20)
        self.load_button.pack(pady=5)
        
        # Draw bounding box button
        self.draw_button = tk.Button(self.button_panel, text="Draw Bounding Box", 
                                    command=self.toggle_drawing, width=20)
        self.draw_button.pack(pady=5)
        
        # Speech to text button
        self.speech_button = tk.Button(self.button_panel, text="Speech to Text", 
                                      command=self.speech_to_text, width=20)
        self.speech_button.pack(pady=5)
        
        # Navigation buttons
        self.nav_frame = tk.Frame(self.button_panel)
        self.nav_frame.pack(pady=10)
        
        self.prev_button = tk.Button(self.nav_frame, text="Previous", 
                                    command=self.prev_image, width=9)
        self.prev_button.pack(side=tk.LEFT, padx=2)
        
        self.next_button = tk.Button(self.nav_frame, text="Next", 
                                    command=self.next_image, width=9)
        self.next_button.pack(side=tk.LEFT, padx=2)
        
        # Save annotations button
        self.save_button = tk.Button(self.button_panel, text="Save Annotations", 
                                    command=self.save_annotations, width=20)
        self.save_button.pack(pady=5)
        
        # Image display area
        self.canvas_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Annotation text display
        self.annotation_frame = tk.Frame(self.main_frame)
        self.annotation_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.annotation_label = tk.Label(self.annotation_frame, text="Annotations:")
        self.annotation_label.pack(anchor=tk.W)
        
        self.annotation_text = tk.Text(self.annotation_frame, height=4, width=50)
        self.annotation_text.pack(fill=tk.X, expand=True)
        
        # Bind canvas events for drawing
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, 
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_images(self):
        """Load images from a directory"""
        folder_path = filedialog.askdirectory(title="Select Image Directory")
        if not folder_path:
            return
            
        self.image_paths = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.image_paths.append(os.path.join(folder_path, file))
        
        if not self.image_paths:
            messagebox.showinfo("Info", "No images found in the selected directory")
            return
            
        self.current_image_index = 0
        self.annotations = {}
        self.display_current_image()
        self.status_var.set(f"Loaded {len(self.image_paths)} images")
    
    def display_current_image(self):
        """Display the current image on the canvas"""
        if not self.image_paths or self.current_image_index >= len(self.image_paths):
            return
            
        # Clear canvas
        self.canvas.delete("all")
        self.current_boxes = []
        
        # Load and display image
        img_path = self.image_paths[self.current_image_index]
        img = Image.open(img_path)
        
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
            img_width, img_height = img.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Display existing annotations if any
        img_name = os.path.basename(img_path)
        if img_name in self.annotations:
            # Display bounding boxes
            for box in self.annotations[img_name].get('boxes', []):
                x1, y1, x2, y2 = box
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
                self.current_boxes.append((x1, y1, x2, y2))
            
            # Display text annotation
            text = self.annotations[img_name].get('text', '')
            self.annotation_text.delete(1.0, tk.END)
            self.annotation_text.insert(tk.END, text)
        else:
            self.annotation_text.delete(1.0, tk.END)
        
        self.status_var.set(f"Image {self.current_image_index + 1}/{len(self.image_paths)}: {img_name}")
    
    def toggle_drawing(self):
        """Toggle drawing mode"""
        self.drawing = not self.drawing
        if self.drawing:
            self.draw_button.config(relief=tk.SUNKEN)
            self.status_var.set("Drawing mode: ON - Click and drag to draw a bounding box")
        else:
            self.draw_button.config(relief=tk.RAISED)
            self.status_var.set("Drawing mode: OFF")
    
    def on_mouse_down(self, event):
        """Handle mouse button press"""
        if not self.drawing:
            return
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = None
    
    def on_mouse_move(self, event):
        """Handle mouse movement while button is pressed"""
        if not self.drawing:
            return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        
        # Delete previous rectangle and draw new one
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, cur_x, cur_y, outline="red", width=2
        )
    
    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if not self.drawing:
            return
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Ensure coordinates are in the right order (top-left to bottom-right)
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        # Add the box to current boxes
        self.current_boxes.append((x1, y1, x2, y2))
        
        # Update annotations
        self.update_annotations()
    
    def speech_to_text(self):
        """Record speech and convert to text annotation"""
        self.status_var.set("Listening... Speak now")
        self.root.update()
        
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                
                # Add to current text
                current_text = self.annotation_text.get(1.0, tk.END).strip()
                if current_text:
                    new_text = current_text + " " + text
                else:
                    new_text = text
                
                self.annotation_text.delete(1.0, tk.END)
                self.annotation_text.insert(tk.END, new_text)
                
                # Update annotations
                self.update_annotations()
                self.status_var.set("Speech recognized")
            except sr.WaitTimeoutError:
                self.status_var.set("No speech detected")
            except sr.UnknownValueError:
                self.status_var.set("Could not understand audio")
            except sr.RequestError:
                self.status_var.set("Could not request results; check your network connection")
    
    def update_annotations(self):
        """Update the annotations dictionary with current data"""
        if not self.image_paths:
            return
            
        img_name = os.path.basename(self.image_paths[self.current_image_index])
        text = self.annotation_text.get(1.0, tk.END).strip()
        
        self.annotations[img_name] = {
            'boxes': self.current_boxes,
            'text': text,
            'path': self.image_paths[self.current_image_index]
        }
    
    def next_image(self):
        """Navigate to the next image"""
        if not self.image_paths:
            return
            
        # Update annotations for current image
        self.update_annotations()
        
        # Move to next image
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_current_image()
        else:
            messagebox.showinfo("Info", "This is the last image")
    
    def prev_image(self):
        """Navigate to the previous image"""
        if not self.image_paths:
            return
            
        # Update annotations for current image
        self.update_annotations()
        
        # Move to previous image
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
        else:
            messagebox.showinfo("Info", "This is the first image")
    
    def save_annotations(self):
        """Save annotations to a JSON file"""
        if not self.annotations:
            messagebox.showinfo("Info", "No annotations to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Annotations"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w') as f:
                json.dump(self.annotations, f, indent=4)
            self.status_var.set(f"Annotations saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {str(e)}")
    
    def run(self):
        """Run the application main loop"""
        self.root.mainloop()


def launch_annotator():
    """Launch the annotator as a standalone application"""
    annotator = Annotator()
    annotator.run()


if __name__ == "__main__":
    launch_annotator()
