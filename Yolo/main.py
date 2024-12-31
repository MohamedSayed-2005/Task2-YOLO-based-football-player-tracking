import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from collections import defaultdict
import sys
import os
import torch
from threading import Thread
from queue import Queue
import time


class VideoStream(Thread):
    def __init__(self, video_path, max_queue_size=30):
        Thread.__init__(self, daemon=True)
        self.video_path = video_path
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.stop_flag = False
        self.fps = 0

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1 / self.fps

        while not self.stop_flag:
            if self.frame_queue.qsize() < self.frame_queue.maxsize:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame_queue.put(frame)
                time.sleep(frame_delay)  # Maintain video FPS
            else:
                time.sleep(0.001)  # Prevent CPU overload

        cap.release()

    def stop(self):
        self.stop_flag = True

    def read(self):
        return False if self.frame_queue.empty() else self.frame_queue.get()


class FrameProcessor(Thread):
    def __init__(self, frame_queue, result_queue, model, resize_factor):
        Thread.__init__(self, daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = model
        self.resize_factor = resize_factor
        self.running = True

        # Create processing buffer
        self.process_buffer = []
        self.buffer_size = 5

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is None:
                    break

                # Buffer frames for batch processing
                self.process_buffer.append(frame)

                if len(self.process_buffer) >= self.buffer_size:
                    # Process batch of frames
                    batch_results = self.process_batch(self.process_buffer)

                    # Send results
                    for frame, results in zip(self.process_buffer, batch_results):
                        self.result_queue.put((frame, results))

                    self.process_buffer = []
            else:
                time.sleep(0.001)

    def process_batch(self, frames):
        processed_frames = []
        for frame in frames:
            height, width = frame.shape[:2]
            processed = cv2.resize(frame, (int(width * self.resize_factor),
                                           int(height * self.resize_factor)))
            processed_frames.append(processed)

        # Batch process with YOLO
        results = self.model.track(processed_frames, persist=True, classes=[0],
                                   stream=True, augment=False)

        return list(results)

    def stop(self):
        self.running = False


class FootballPlayerTracker:
    def __init__(self, model_path=None):
        try:
            # Initialize YOLO with optimized settings
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                print("Downloading YOLO model...")
                self.model = YOLO('yolov8n.pt')

            # Optimize model settings
            self.model.conf = 0.3
            self.model.iou = 0.4
            self.model.max_det = 50

            # Enable GPU acceleration with mixed precision
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cuda':
                torch.backends.cudnn.benchmark = True
                self.model.to(self.device).half()  # Use FP16 for GPU
            else:
                self.model.to(self.device)

            # Processing queues with larger sizes
            self.frame_queue = Queue(maxsize=60)
            self.result_queue = Queue(maxsize=60)
            self.display_queue = Queue(maxsize=30)

            self.player_tracks = defaultdict(list)
            self.frame_dimensions = None
            self.paused = False
            self.current_player = None
            self.process_every_n_frames = 2
            self.frame_count = 0
            self.resize_factor = 0.5
            self.display_width = 800

            # Initialize UI
            self.root = tk.Tk()
            self.root.title("Football Player Tracker")
            self.setup_ui()

            # Performance monitoring
            self.last_frame_time = time.time()
            self.frame_times = []

            print("Initialization successful!")

        except Exception as e:
            print(f"Error initializing tracker: {str(e)}")
            sys.exit(1)

    def process_video(self, video_path):
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
            return False

        try:
            # Initialize video stream thread
            self.video_stream = VideoStream(video_path)
            self.video_stream.start()

            # Initialize processor thread
            self.processor = FrameProcessor(self.frame_queue, self.result_queue,
                                            self.model, self.resize_factor)
            self.processor.start()

            # Start display loop
            self.update_display()

            return True

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False

    def update_display(self):
        if not self.paused:
            # Read frame from video stream
            frame = self.video_stream.read()
            if frame is not None:
                self.frame_count += 1

                # Process frame
                if self.frame_count % self.process_every_n_frames == 0:
                    if self.frame_queue.qsize() < self.frame_queue.maxsize - 1:
                        self.frame_queue.put(frame)

                # Process results
                while not self.result_queue.empty():
                    frame, results = self.result_queue.get()
                    if results and results.boxes is not None and hasattr(results.boxes, 'id'):
                        self.process_results(frame, results)

                # Update FPS counter
                current_time = time.time()
                self.frame_times.append(current_time - self.last_frame_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)
                fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.last_frame_time = current_time

                # Display frame
                self.display_frame(frame)

                # Update UI elements less frequently
                if self.frame_count % (self.process_every_n_frames * 4) == 0:
                    if len(self.player_tracks) > 0:
                        self.update_player_list()
                    if self.current_player is not None:
                        self.update_heatmap(self.current_player)

        # Schedule next update
        self.root.after(1, self.update_display)

    def display_frame(self, frame):
        try:
            # Use PIL for efficient image conversion and resizing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # Calculate new dimensions while maintaining aspect ratio
            ratio = self.display_width / frame_pil.width
            display_height = int(frame_pil.height * ratio)

            # Resize using BILINEAR for better quality/speed trade-off
            frame_pil = frame_pil.resize((self.display_width, display_height),
                                         Image.Resampling.BILINEAR)

            # Convert to PhotoImage
            frame_tk = ImageTk.PhotoImage(frame_pil)
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk

        except Exception as e:
            print(f"Error displaying frame: {str(e)}")

    # Rest of the class methods remain the same...

    def create_soccer_field(self):
        field = np.ones((600, 800, 3))
        field[:, :] = [0.2, 0.5, 0.2]  # Green color

        def draw_line(img, start, end, color=(1, 1, 1), thickness=2):
            cv2.line(img,
                     (int(start[0] * img.shape[1]), int(start[1] * img.shape[0])),
                     (int(end[0] * img.shape[1]), int(end[1] * img.shape[0])),
                     color, thickness)

        # Field outline
        draw_line(field, (0.05, 0.05), (0.95, 0.05))  # Top
        draw_line(field, (0.05, 0.95), (0.95, 0.95))  # Bottom
        draw_line(field, (0.05, 0.05), (0.05, 0.95))  # Left
        draw_line(field, (0.95, 0.05), (0.95, 0.95))  # Right

        # Center line
        draw_line(field, (0.5, 0.05), (0.5, 0.95))

        # Center circle
        cv2.circle(field,
                   (int(0.5 * field.shape[1]), int(0.5 * field.shape[0])),
                   int(0.1 * field.shape[0]),
                   (1, 1, 1), 2)

        # Penalty areas
        draw_line(field, (0.05, 0.25), (0.25, 0.25))
        draw_line(field, (0.25, 0.25), (0.25, 0.75))
        draw_line(field, (0.25, 0.75), (0.05, 0.75))
        draw_line(field, (0.95, 0.25), (0.75, 0.25))
        draw_line(field, (0.75, 0.25), (0.75, 0.75))
        draw_line(field, (0.75, 0.75), (0.95, 0.75))

        return field

    def setup_ui(self):
        # Create main container with three columns
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left panel - Player List and Controls
        left_frame = ttk.Frame(main_container)
        main_container.add(left_frame, weight=1)

        # Add file selection button
        file_frame = ttk.Frame(left_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(file_frame, text="Open Video File",
                   command=self.select_video_file).pack(fill=tk.X)

        # Player list controls
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(controls_frame, text="Show Selected",
                   command=self.update_visible_players).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Show All",
                   command=self.show_all_players).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Pause/Resume",
                   command=self.toggle_pause).pack(side=tk.LEFT, padx=2)

        # Player listbox
        self.player_listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE, height=30)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical",
                                  command=self.player_listbox.yview)
        self.player_listbox.configure(yscrollcommand=scrollbar.set)

        self.player_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Center panel - Video Display
        center_frame = ttk.Frame(main_container)
        main_container.add(center_frame, weight=6)

        self.video_label = ttk.Label(center_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right panel - Heatmap Display
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=2)

        # Create figure for heatmap
        self.fig, self.ax = plt.subplots(figsize=(4, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # FPS display
        self.fps_label = ttk.Label(self.root, text="FPS: 0")
        self.fps_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Status bar
        self.status_label = ttk.Label(self.root, text="Status: Ready")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Bind listbox selection to heatmap update
        self.player_listbox.bind('<<ListboxSelect>>', self.on_select_player)

    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if file_path:
            self.process_video(file_path)

    def process_video(self, video_path):
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found!")
            return False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video file!")
                return False

            # Optimize video capture
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.frame_dimensions = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

            frame_times = []
            last_time = time.time()

            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    self.frame_count += 1

                    # Skip frames for processing
                    if self.frame_count % self.process_every_n_frames == 0:
                        if self.frame_queue.qsize() < 3:  # Only add if queue not full
                            self.frame_queue.put(frame)

                    # Process results
                    while not self.result_queue.empty():
                        frame, results = self.result_queue.get()
                        if results and results[0].boxes is not None and hasattr(results[0].boxes, 'id'):
                            self.process_results(frame, results[0])

                    # Calculate and display FPS
                    current_time = time.time()
                    frame_times.append(current_time - last_time)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    fps = 1.0 / (sum(frame_times) / len(frame_times))
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                    last_time = current_time

                    # Display frame
                    self.display_frame(frame)

                    # Update UI less frequently
                    if self.frame_count % (self.process_every_n_frames * 2) == 0:
                        if len(self.player_tracks) > 0:
                            self.update_player_list()
                        if self.current_player is not None:
                            self.update_heatmap(self.current_player)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                self.root.update()

            cap.release()
            return True

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return False

    def process_results(self, frame, results):
        boxes = results.boxes.xywh.cpu()
        track_ids = results.boxes.id.cpu()
        scores = results.boxes.conf.cpu()

        # Scale boxes back to original size
        boxes[:, 0:4] /= self.resize_factor

        # Update tracking data
        for box, track_id, score in zip(boxes, track_ids, scores):
            if score > 0.3:  # Increased threshold
                x, y = int(box[0]), int(box[1])
                track_id = int(track_id)
                self.player_tracks[track_id].append((x, y, len(self.player_tracks[track_id])))

                # Draw bounding box and ID (simplified)
                x, y, w, h = map(int, box)
                cv2.rectangle(frame,
                              (x - int(w / 2), y - int(h / 2)),
                              (x + int(w / 2), y + int(h / 2)),
                              (0, 255, 0), 1)

    def display_frame(self, frame):
        # Convert and resize frame efficiently
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Calculate new dimensions
        ratio = self.display_width / frame_pil.width
        display_height = int(frame_pil.height * ratio)

        # Resize using NEAREST for speed
        frame_pil = frame_pil.resize((self.display_width, display_height),
                                     Image.Resampling.NEAREST)

        frame_tk = ImageTk.PhotoImage(frame_pil)
        self.video_label.configure(image=frame_tk)
        self.video_label.image = frame_tk

    def update_player_list(self):
        current_items = set(self.player_listbox.get(0, tk.END))
        current_players = {f"Player {id}" for id in self.player_tracks.keys()}

        # Add new players
        for player in current_players - current_items:
            self.player_listbox.insert(tk.END, player)

    def update_visible_players(self):
        selection = self.player_listbox.curselection()
        if selection:
            player_id = int(self.player_listbox.get(selection[0]).split()[1])
            self.current_player = player_id
            self.update_heatmap(player_id)

    def show_all_players(self):
        self.player_listbox.select_set(0, tk.END)
        self.update_visible_players()

    def toggle_pause(self):
        self.paused = not self.paused
        status = "Paused" if self.paused else "Running"
        self.status_label.config(text=f"Status: {status}")


    def on_select_player(self, event=None):
        selection = self.player_listbox.curselection()
        if selection:
            player_id = int(self.player_listbox.get(selection[0]).split()[1])
            self.current_player = player_id
            self.update_heatmap(player_id)

    def update_heatmap(self, player_id):
        if player_id not in self.player_tracks or not self.player_tracks[player_id]:
            return

        self.ax.clear()
        positions = self.player_tracks[player_id]
        x_pos = [p[0] for p in positions]
        y_pos = [p[1] for p in positions]

        # Display soccer field background
        self.ax.imshow(self.field_img, extent=[0, self.frame_dimensions[0], 0, self.frame_dimensions[1]])

        # Create heatmap with transparency
        heatmap, xedges, yedges = np.histogram2d(
            x_pos, y_pos,
            bins=50,
            range=[[0, self.frame_dimensions[0]], [0, self.frame_dimensions[1]]]
        )

        # Plot heatmap with transparency
        self.ax.imshow(heatmap.T, cmap='hot', origin='lower', alpha=0.6)
        self.ax.set_title(f'Player {player_id} Heatmap')
        self.canvas.draw()


def main():
    try:
        # Create the tracker instance
        tracker = FootballPlayerTracker()

        # Start the main event loop
        tracker.root.mainloop()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()