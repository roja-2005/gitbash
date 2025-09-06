import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import os
import requests
from PIL import Image
from io import BytesIO
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

class GestureController:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.frame_width = 1280
        self.frame_height = 720

        if not os.path.exists('shirts'):
            os.makedirs('shirts')

        # Load shirt filenames from directory
        self.shirts = [f for f in os.listdir('shirts') if f.lower().endswith('.png')]
        self.current_shirt_index = len(self.shirts) - 1 if self.shirts else 0
        self.shirt_scale = 2.0
        self.size_multiplier = 0.9
        self.prev_gesture = None
        self.gesture_cooldown = 0

        self.load_shirts()

    def get_chromedriver_path(self):
        # Check for chromedriver executable in the script directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        chromedriver_exe = 'chromedriver.exe' if os.name == 'nt' else 'chromedriver'
        path = os.path.join(base_dir, chromedriver_exe)
        if not os.path.exists(path):
            messagebox.showerror("ChromeDriver Missing",
                f"ChromeDriver executable not found at:\n{path}\n\n"
                "Please download the correct version from:\n"
                "https://sites.google.com/chromium.org/driver/")
            raise FileNotFoundError(f"ChromeDriver not found at {path}")
        return path

    def add_amazon_product(self, url):
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument(
                'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

            chrome_driver_path = self.get_chromedriver_path()
            service = Service(chrome_driver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(url)

            wait = WebDriverWait(driver, 10)
            img_element = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, '#landingImage, #imgBlkFront, .a-dynamic-image')
            ))

            img_url = img_element.get_attribute('src') or img_element.get_attribute('data-old-hires')

            if img_url:
                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                }

                img_response = requests.get(img_url, headers=headers)
                img = Image.open(BytesIO(img_response.content))
                img = img.convert('RGBA')

                # Make white background transparent
                data = img.getdata()
                new_data = []
                for item in data:
                    if item[0] > 240 and item[1] > 240 and item[2] > 240:
                        new_data.append((255, 255, 255, 0))
                    else:
                        new_data.append(item)
                img.putdata(new_data)

                filename = f'amazon_shirt_{len(os.listdir("shirts"))}.png'
                filepath = os.path.join('shirts', filename)
                img.save(filepath, 'PNG')

                # Reload shirt list and images
                self.shirts = [f for f in os.listdir('shirts') if f.lower().endswith('.png')]
                self.current_shirt_index = len(self.shirts) - 1
                self.load_shirts()
                driver.quit()
                return True

            driver.quit()
            return False

        except Exception as e:
            print(f"Error adding Amazon product: {str(e)}")
            if 'driver' in locals():
                driver.quit()
            return False

    def load_shirts(self):
        self.shirt_images = []
        for shirt in self.shirts:
            try:
                path = os.path.join('shirts', shirt)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.shirt_images.append(img)
                else:
                    print(f"Failed to load image: {shirt}")
            except Exception as e:
                print(f"Error loading {shirt}: {str(e)}")

    def detect_hand_gesture(self, hand_landmarks):
        try:
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]

            thumb_index_dist = hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
            index_middle_dist = hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
            middle_ring_dist = hypot(middle_tip.x - ring_tip.x, middle_tip.y - ring_tip.y)

            if self.gesture_cooldown > 0:
                self.gesture_cooldown -= 1
                return

            if thumb_index_dist < 0.05:
                if self.prev_gesture != 'increase':
                    self.size_multiplier = min(2.0, self.size_multiplier + 0.1)
                    self.gesture_cooldown = 10
                    self.prev_gesture = 'increase'
            elif middle_ring_dist < 0.05:
                if self.prev_gesture != 'decrease':
                    self.size_multiplier = max(0.5, self.size_multiplier - 0.1)
                    self.gesture_cooldown = 10
                    self.prev_gesture = 'decrease'
            elif index_middle_dist < 0.05:
                if self.prev_gesture != 'change':
                    if self.shirt_images:
                        self.current_shirt_index = (self.current_shirt_index + 1) % len(self.shirt_images)
                    self.gesture_cooldown = 10
                    self.prev_gesture = 'change'
            else:
                self.prev_gesture = None

        except Exception as e:
            print(f"Error in hand gesture detection: {str(e)}")

    def overlay_shirt(self, frame, landmarks):
        try:
            if not self.shirt_images:
                return frame

            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]

            shoulder_width = int(hypot(
                (right_shoulder.x - left_shoulder.x) * self.frame_width,
                (right_shoulder.y - left_shoulder.y) * self.frame_height
            ))

            torso_height = int(hypot(
                ((left_hip.x + right_hip.x)/2 - (left_shoulder.x + right_shoulder.x)/2) * self.frame_width,
                ((left_hip.y + right_hip.y)/2 - (left_shoulder.y + right_shoulder.y)/2) * self.frame_height
            ))

            shirt_img = self.shirt_images[self.current_shirt_index]
            shirt_width = int(shoulder_width * self.shirt_scale * self.size_multiplier)
            shirt_height = int(torso_height * 1.1)

            center_x = int((left_shoulder.x + right_shoulder.x) * self.frame_width / 2)
            center_y = int(left_shoulder.y * self.frame_height) - int(shirt_height * 0.13)

            resized_shirt = cv2.resize(shirt_img, (shirt_width, shirt_height))

            x1 = max(0, center_x - shirt_width // 2)
            y1 = max(0, center_y)
            x2 = min(frame.shape[1], x1 + shirt_width)
            y2 = min(frame.shape[0], y1 + shirt_height)

            if resized_shirt.shape[2] == 4:
                shirt_region = resized_shirt[:y2 - y1, :x2 - x1]
                alpha = shirt_region[:, :, 3] / 255.0
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        frame[y1:y2, x1:x2, c] * (1 - alpha) +
                        shirt_region[:, :, c] * alpha
                    ).astype(np.uint8)

            return frame
        except Exception as e:
            print(f"Error in overlay_shirt: {str(e)}")
            return frame

    def process_frame(self, frame):
        try:
            if frame is None:
                return None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_height, self.frame_width = frame.shape[:2]

            pose_results = self.pose.process(rgb_frame)
            hands_results = self.hands.process(rgb_frame)

            if pose_results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        self.detect_hand_gesture(hand_landmarks)

                frame = self.overlay_shirt(frame, pose_results.pose_landmarks)

            return frame

        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            return frame

    def get_current_state(self):
        if not self.shirts:
            return {'current_shirt': 'None', 'scale': int(self.size_multiplier * 100)}
        return {
            'current_shirt': self.shirts[self.current_shirt_index],
            'scale': int(self.size_multiplier * 100)
        }

def main():
    def launch_tryon():
        root.destroy()
        controller = GestureController()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, controller.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, controller.frame_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = controller.process_frame(frame)
            if processed_frame is not None:
                state = controller.get_current_state()
                cv2.putText(
                    processed_frame,
                    f"Shirt: {state['current_shirt']} | Scale: {state['scale']}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow('Virtual Try-On', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    root = tk.Tk()
    root.title("Virtual Try-On System")
    root.geometry("600x400")

    controller = GestureController()

    frame = ttk.Frame(root, padding="20")
    frame.pack(fill=tk.BOTH, expand=True)

    title_label = ttk.Label(frame, text="Virtual Try-On System", font=('Helvetica', 16, 'bold'))
    title_label.pack(pady=10)

    instructions = ttk.Label(frame, text="1. Browse Amazon for shirts\n2. Copy the product URL\n3. Paste URL below and click Add\n4. Click Start Try-On when ready")
    instructions.pack(pady=10)

    ttk.Label(frame, text="Amazon Product URL:").pack(pady=5)
    url_entry = ttk.Entry(frame, width=50)
    url_entry.pack(pady=5)

    def add_product():
        url = url_entry.get()
        if url:
            success = controller.add_amazon_product(url)
            if success:
                messagebox.showinfo("Success", "Product added successfully!")
                url_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Error", "Failed to add product")

    def open_amazon():
        webbrowser.open("https://www.amazon.com/s?k=shirts")

    button_frame = ttk.Frame(frame)
    button_frame.pack(pady=20)

    ttk.Button(button_frame, text="Browse Amazon", command=open_amazon).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Add Product", command=add_product).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Start Try-On", command=launch_tryon).pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    main()
