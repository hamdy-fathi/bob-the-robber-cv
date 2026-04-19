import asyncio
import json
import threading
import time
import csv
import math
import argparse
import sys
import urllib.request
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
from websockets.server import serve

HOST = "127.0.0.1"
PORT = 8765
SEND_FPS = 20

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Added medical metrics to the state payload
latest_state = {
    "move": "idle", 
    "action": False, 
    "connected": False, 
    "tracked": False, 
    "x": 0.5, 
    "y": 0.5,
    "tremor_score": 0.0,
    "pinch_amplitude": 0.0,
    "flicks_count": 0,
    "total_pinches": 0
}
state_lock = threading.Lock()
stop_event = threading.Event()

os_keymap = {
    "left": "Key.left",
    "right": "Key.right",
    "up": "Key.up",
    "down": "Key.down",
    "action": "Key.space",
    "auto_hold_enabled": False,
    "auto_hold_key": "w"
}

clients = set()
clients_lock = asyncio.Lock()

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

def ensure_model_file():
    if MODEL_PATH.exists():
        return MODEL_PATH
    print("Downloading MediaPipe hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return MODEL_PATH

def launch_gui():
    root = tk.Tk()
    root.title("KineticLink Control Mapper")
    root.geometry("450x400")
    
    options = [
        "Key.left", "Key.right", "Key.up", "Key.down", 
        "Key.space", "Key.enter", "Key.shift", "Key.ctrl", "Key.alt",
        "w", "a", "s", "d", "e", "q", "f", "r"
    ]
    
    tk.Label(root, text="Map Rehab Gestures to Game Keys:", font=("Helvetica", 12, "bold")).pack(pady=10)
    
    frame = tk.Frame(root)
    frame.pack(pady=10)
    
    def create_row(row_idx, label_text, default_val):
        tk.Label(frame, text=label_text).grid(row=row_idx, column=0, padx=10, pady=5, sticky="e")
        cb = ttk.Combobox(frame, values=options)
        cb.set(default_val)
        cb.grid(row=row_idx, column=1, padx=10, pady=5)
        return cb
        
    left_cb = create_row(0, "Sweep Left:", "Key.left")
    right_cb = create_row(1, "Sweep Right:", "Key.right")
    up_cb = create_row(2, "Flick Up:", "Key.up")
    down_cb = create_row(3, "Sweep Down:", "Key.down")
    action_cb = create_row(4, "Pinch (Action):", "Key.space")
    
    ttk.Separator(frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='ew', pady=15)
    
    auto_hold_var = tk.BooleanVar(value=False)
    tk.Checkbutton(frame, text="Auto-Hold Key (Gas Pedal):", variable=auto_hold_var).grid(row=6, column=0, padx=10, pady=5, sticky="e")
    
    auto_hold_cb = ttk.Combobox(frame, values=options)
    auto_hold_cb.set("w")
    auto_hold_cb.grid(row=6, column=1, padx=10, pady=5)
    
    def on_start():
        global os_keymap
        os_keymap["left"] = left_cb.get()
        os_keymap["right"] = right_cb.get()
        os_keymap["up"] = up_cb.get()
        os_keymap["down"] = down_cb.get()
        os_keymap["action"] = action_cb.get()
        os_keymap["auto_hold_enabled"] = auto_hold_var.get()
        os_keymap["auto_hold_key"] = auto_hold_cb.get()
        root.destroy()
        
    def on_close():
        stop_event.set()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_close)
    tk.Button(root, text="Start Tracker", font=("Helvetica", 12), bg="#4CAF50", fg="white", command=on_start).pack(pady=20)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()

def smooth_move(move_history):
    if not move_history:
        return "idle"
    counts = Counter(move_history)
    return counts.most_common(1)[0][0]

# REHAB MAPPING: Require large arm movements (tracked via wrist coordinates) to combat rigidity
def detect_move(cx, cy):
    horizontal = "idle"
    vertical = "idle"

    # Adjusted for a better balance: Requires movement, but not extreme stretching.
    # Center 'idle' zone is between 35% and 65% of the screen.
    if cx < 0.35:
        horizontal = "left"
    elif cx > 0.65:
        horizontal = "right"

    # Remove static 'up' movement (now handled by Flick Detection)
    if cy > 0.65:
        vertical = "down"

    if horizontal != "idle" and vertical != "idle":
        return f"{horizontal}+{vertical}"
    if horizontal != "idle":
        return horizontal
    if vertical != "idle":
        return vertical
    return "idle"

def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = []
    for landmark in hand_landmarks:
        points.append((int(landmark.x * w), int(landmark.y * h)))

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (90, 255, 90), 2)

    for point in points:
        cv2.circle(frame, point, 3, (0, 200, 255), -1)

# CSV Logger Thread
def csv_logger_loop():
    csv_filename = f"rehab_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Game_Move", "Action_Pinch", "Tremor_Score", "Pinch_Amplitude", "Wrist_Flicks", "Total_Pinches"])
        
        print(f"Started logging rehab data to {csv_filename}")
        
        while not stop_event.is_set():
            time.sleep(1.0) # Log every second
            with state_lock:
                writer.writerow([
                    datetime.now().strftime('%H:%M:%S'),
                    latest_state["move"],
                    latest_state["action"],
                    f"{latest_state['tremor_score']:.4f}",
                    f"{latest_state['pinch_amplitude']:.4f}",
                    latest_state["flicks_count"],
                    latest_state["total_pinches"]
                ])
                f.flush()

def vision_loop():
    global latest_state

    try:
        model_path = ensure_model_file()
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as error:
        print(f"ERROR: failed to initialize MediaPipe Tasks API: {error}")
        stop_event.set()
        return

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        stop_event.set()
        return

    move_history = deque(maxlen=5)
    
    # MEDICAL TRACKING VARIABLES
    tremor_positions = deque(maxlen=30) # Store index tip positions for ~1 second
    wrist_y_history = deque(maxlen=5)   # Store wrist Y for flick velocity tracking
    current_tremor_score = 0.0
    
    current_max_pinch = 0.0
    recorded_pinch_amplitude = 0.0
    
    total_flicks = 0
    flick_cooldown = 0
    hand_size_history = deque(maxlen=10) # Sliding window to track hand size
    
    total_pinches = 0
    previous_action = False

    cv2.namedWindow("KineticLink Rehab Tracker", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("KineticLink Rehab Tracker", cv2.WND_PROP_TOPMOST, 1)

    with mp_vision.HandLandmarker.create_from_options(options) as hand_landmarker:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            move = "idle"
            action = False
            tracked = False
            x_pos = 0.5
            y_pos = 0.5

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                draw_hand(frame, hand_landmarks)

                wrist = hand_landmarks[0]
                index_tip = hand_landmarks[8]
                thumb_tip = hand_landmarks[4]
                
                tracked = True
                x_pos = wrist.x
                y_pos = wrist.y

                # Rehab: Track the WRIST for movement to force large arm sweeps
                move = detect_move(wrist.x, wrist.y)
                
                # Flick Detection (Wrist Extension -> Wrist Flexion based on user photos)
                # Instead of a strict state machine, we track the 2D size of the hand on screen.
                # When the wrist flexes forward, the distance from wrist to fingertips shrinks drastically!
                middle_tip = hand_landmarks[12]
                current_hand_size = math.hypot(wrist.x - middle_tip.x, wrist.y - middle_tip.y)
                hand_size_history.append(current_hand_size)
                
                if flick_cooldown > 0:
                    flick_cooldown -= 1
                    if flick_cooldown > 10:
                        move = "up" # Hold the 'up' command for 5 frames to guarantee it registers over WebSocket
                    else:
                        move = "idle" # PREVENT overlapping movement while recovering from a flick!
                elif len(hand_size_history) == 10:
                    max_recent_size = max(hand_size_history)
                    
                    # If the hand was open (size > 0.08) and suddenly shrinks by 35% or more (Flexion):
                    if max_recent_size > 0.08 and current_hand_size < (max_recent_size * 0.65):
                        move = "up"
                        total_flicks += 1
                        flick_cooldown = 15 # 15 frames total buffer
                        hand_size_history.clear() # Reset history
                
                if move != "up":
                    move_history.append(move)
                    move = smooth_move(move_history)
                else:
                    move_history.clear()

                # MEDICAL TRACKING: Pinch Distance (Hypokinesia / Tapping)
                pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                
                # Prevent foreshortening during a forward flick from triggering an accidental pinch
                if flick_cooldown > 0 or current_hand_size < 0.08:
                    action = False
                else:
                    action = pinch_dist < 0.05
                
                if action and not previous_action:
                    total_pinches += 1
                previous_action = action
                
                if action:
                    # When they pinch, lock in their max opening amplitude before the pinch
                    recorded_pinch_amplitude = current_max_pinch
                    current_max_pinch = 0.0 # reset for the next pinch cycle
                else:
                    # When opening hand, track how wide they can stretch their fingers
                    current_max_pinch = max(current_max_pinch, pinch_dist)

                # MEDICAL TRACKING: Tremor (Resting Micro-oscillations)
                tremor_positions.append((index_tip.x, index_tip.y))
                if move == "idle" and len(tremor_positions) > 10:
                    # Calculate frame-to-frame path length (high-frequency jitter)
                    total_jitter = 0.0
                    for i in range(1, len(tremor_positions)):
                        dx = tremor_positions[i][0] - tremor_positions[i-1][0]
                        dy = tremor_positions[i][1] - tremor_positions[i-1][1]
                        total_jitter += math.hypot(dx, dy)
                    
                    # Average frame-to-frame travel distance, scaled up for readability
                    avg_jitter = (total_jitter / len(tremor_positions)) * 1000
                    
                    # Deadzone filter: ignore normal webcam ML model tracking noise
                    if avg_jitter < 1.5:
                        current_tremor_score = 0.0
                    else:
                        current_tremor_score = avg_jitter
                else:
                    current_tremor_score = 0.0 # Don't track tremor while the patient is actively moving their arm

                h, w, _ = frame.shape
                # Draw wrist tracker for movement
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                cv2.line(frame, (w // 2, h // 2), (cx, cy), (0, 255, 0), 2)
                
                # Draw pinch tracker line
                px, py = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                cv2.line(frame, (px, py), (tx, ty), (255, 0, 255) if action else (255, 255, 255), 2)

            else:
                # TRACKING LOST! 
                # MediaPipe loses tracking when the hand flexes forward (extreme foreshortening).
                # We can interpret a tracking loss in the CENTER of the screen as a successful flick!
                if flick_cooldown > 0:
                    flick_cooldown -= 1
                    if flick_cooldown > 10:
                        move = "up"
                    else:
                        move = "idle"
                else:
                    if len(hand_size_history) >= 3:
                        max_recent_size = max(hand_size_history)
                        last_x, last_y = latest_state["x"], latest_state["y"]
                        # If hand was large, and tracking was lost away from the screen edges, it's a flick!
                        if max_recent_size > 0.08 and (0.1 < last_x < 0.9) and (0.1 < last_y < 0.9):
                            move = "up"
                            total_flicks += 1
                            flick_cooldown = 15
                            
                if move != "up":
                    move_history.append("idle")
                    move = smooth_move(move_history)
                else:
                    move_history.clear()
                    
                tremor_positions.clear()
                hand_size_history.clear()
                current_tremor_score = 0.0

            with state_lock:
                latest_state = {
                    "move": move,
                    "action": action,
                    "connected": True,
                    "tracked": tracked,
                    "x": x_pos,
                    "y": y_pos,
                    "tremor_score": current_tremor_score,
                    "pinch_amplitude": recorded_pinch_amplitude,
                    "flicks_count": total_flicks,
                    "total_pinches": total_pinches
                }

            cv2.putText(frame, f"Move (Wrist): {move}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 255, 40), 2)
            cv2.putText(frame, f"Pinch (Action): {action}", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255) if action else (200, 200, 200), 2)
            cv2.putText(frame, f"Tremor Score: {current_tremor_score:.2f}", (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            cv2.putText(frame, f"Pinch Amplitude: {recorded_pinch_amplitude:.2f}", (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Flicks: {total_flicks} | Pinches: {total_pinches}", (10, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("KineticLink Rehab Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27: # 'q' or ESC key
                stop_event.set()
                break

    cap.release()
    cv2.destroyAllWindows()


async def handler(websocket):
    async with clients_lock:
        clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        async with clients_lock:
            clients.discard(websocket)


async def broadcaster():
    while not stop_event.is_set():
        await asyncio.sleep(1 / SEND_FPS)
        with state_lock:
            payload = json.dumps(latest_state)

        async with clients_lock:
            ws_clients = list(clients)

        if not ws_clients:
            continue

        dead = []
        for ws in ws_clients:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with clients_lock:
                for ws in dead:
                    clients.discard(ws)


async def main():
    print(f"Starting MediaPipe WebSocket server at ws://{HOST}:{PORT}")
    async with serve(handler, HOST, PORT):
        await broadcaster()

def str_to_pynput_key(key_str):
    from pynput.keyboard import Key, KeyCode
    if key_str.startswith("Key."):
        attr = key_str.split(".")[1]
        if hasattr(Key, attr):
            return getattr(Key, attr)
    return KeyCode.from_char(key_str)

def os_keyboard_loop():
    try:
        from pynput.keyboard import Key, Controller
    except ImportError:
        print("ERROR: pynput is required for --os-keyboard mode. Run: pip install pynput")
        stop_event.set()
        return

    keyboard = Controller()
    
    current_pressed_moves = set()
    current_pressed_action = False
    
    print("Started OS Keyboard simulation mode! Your physical keyboard is now being controlled by the AI.")
    
    key_left = str_to_pynput_key(os_keymap["left"])
    key_right = str_to_pynput_key(os_keymap["right"])
    key_up = str_to_pynput_key(os_keymap["up"])
    key_down = str_to_pynput_key(os_keymap["down"])
    key_action = str_to_pynput_key(os_keymap["action"])
    
    key_auto_hold = None
    if os_keymap["auto_hold_enabled"]:
        key_auto_hold = str_to_pynput_key(os_keymap["auto_hold_key"])
        
    current_auto_hold_pressed = False
    
    while not stop_event.is_set():
        time.sleep(1 / SEND_FPS) # Match standard FPS rate
        
        with state_lock:
            target_move = latest_state["move"]
            target_action = latest_state["action"]
            is_tracked = latest_state["tracked"]
            
        # Auto-hold logic (Fatigue therapy)
        if key_auto_hold:
            if is_tracked and not current_auto_hold_pressed:
                keyboard.press(key_auto_hold)
                current_auto_hold_pressed = True
            elif not is_tracked and current_auto_hold_pressed:
                keyboard.release(key_auto_hold)
                current_auto_hold_pressed = False
            
        # Parse moves (e.g. "left+down" or "idle")
        if target_move == "idle":
            target_moves_set = set()
        else:
            target_moves_set = set(target_move.split("+"))
            
        # Release old keys
        for m in current_pressed_moves - target_moves_set:
            if m == "left": keyboard.release(key_left)
            elif m == "right": keyboard.release(key_right)
            elif m == "up": keyboard.release(key_up)
            elif m == "down": keyboard.release(key_down)
            
        # Press new keys
        for m in target_moves_set - current_pressed_moves:
            if m == "left": keyboard.press(key_left)
            elif m == "right": keyboard.press(key_right)
            elif m == "up": keyboard.press(key_up)
            elif m == "down": keyboard.press(key_down)
            
        current_pressed_moves = target_moves_set
        
        # Action (Pinch) -> Spacebar
        if target_action and not current_pressed_action:
            keyboard.press(key_action)
            current_pressed_action = True
        elif not target_action and current_pressed_action:
            keyboard.release(key_action)
            current_pressed_action = False
            
    # Safety cleanup: ensure all keys are released on exit
    for m in current_pressed_moves:
        if m == "left": keyboard.release(key_left)
        elif m == "right": keyboard.release(key_right)
        elif m == "up": keyboard.release(key_up)
        elif m == "down": keyboard.release(key_down)
    if current_pressed_action:
        keyboard.release(key_action)
    if current_auto_hold_pressed and key_auto_hold:
        keyboard.release(key_auto_hold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KineticLink Rehab Tracker")
    parser.add_argument("--os-keyboard", action="store_true", help="Enable OS-level keyboard simulation for any game")
    args = parser.parse_args()

    if args.os_keyboard:
        launch_gui()
        if stop_event.is_set():
            sys.exit(0)

    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()
    
    logger_thread = threading.Thread(target=csv_logger_loop, daemon=True)
    logger_thread.start()
    
    if args.os_keyboard:
        keyboard_thread = threading.Thread(target=os_keyboard_loop, daemon=True)
        keyboard_thread.start()
        try:
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            keyboard_thread.join(timeout=2)
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            
    vision_thread.join(timeout=2)
    logger_thread.join(timeout=2)
