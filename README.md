# KineticLink: Parkinson's Rehab Tracker

**KineticLink** is a specialized, gamified physical therapy tool designed specifically for Parkinson's Disease rehabilitation. Built as a modification to an HTML5 game, it replaces traditional keyboard controls with an advanced, invisible computer vision layer powered by Google MediaPipe. 

Instead of tapping keys, patients must perform specific, clinically-relevant biomotor exercises to control the game. While the patient plays, the Python backend silently runs advanced mathematical analyses to log medical metrics like Resting Tremor severity and Amplitude Decrements.

## 🩺 Therapeutic Control Mapping

The game controls are intentionally designed to combat the primary motor symptoms of Parkinson's:

*   **Combating Rigidity (Arm Sweeps):** To move the character Left, Right, or Down, the patient must perform large, sweeping arm movements. The AI tracks the *Wrist* coordinate, forcing the patient to exceed normal ranges of motion to trigger the game event.
*   **Combating Bradykinesia (Finger Tapping/Pinch):** To interact with objects (e.g., unlocking doors), the patient must perform the standard clinical "Finger Tapping" test by pinching their Index Finger and Thumb together.
*   **Reflex & Speed (Whole-Hand Flicks):** To move the character UP, the patient must perform a rapid, forward wrist flexion (a "flick"). The AI measures the frame-to-frame size of the hand; if the hand visually shrinks by over 35% in a split-second, the flick is registered.

## 📊 Invisible Medical Logging

While the game provides a fun, distracting frontend for the patient, the Python backend acts as a strict medical tracker. As soon as the server is started, it automatically generates a unique `rehab_session_YYYYMMDD.csv` file.

Every second, it logs:
1.  **Game Move:** The patient's current movement state.
2.  **Action Pinch:** Whether the patient successfully completed a pinch gesture.
3.  **Tremor Score:** When the patient holds their arm still, the AI tracks the high-frequency micro-jitter (frame-to-frame pathing length) of their index finger, filtering out webcam static and slow drifting to generate an accurate Resting Tremor magnitude score.
4.  **Pinch Amplitude:** Measures how wide the patient extends their fingers before a pinch, tracking for amplitude decrement (Hypokinesia).
5.  **Wrist Flicks:** A total count of successful, high-velocity wrist flexions.

## 🛠️ Setup & Installation

### Requirements
*   Python 3.8+
*   A functional Webcam

### Installation
1. Clone or download this repository.
2. Install the required Python dependencies:
   ```bash
   pip install mediapipe opencv-python websockets
   ```

### Running the Tracker

**Mode 1: Web Game (Default)**
1. Start the backend computer vision server:
   ```bash
   python mediapipe_controller.py
   ```
   *Note: On the first run, the script will automatically download the required MediaPipe ML model (`hand_landmarker.task`).*
2. Open `index.html` in any modern web browser to launch the gamified frontend.
3. Click **"Enable Python MediaPipe"** on the browser UI to connect.

**Mode 2: Universal OS Keyboard (Play Any Game)**
You can use the tracker to play any desktop game or emulator on your computer. In this mode, the AI physically injects keyboard presses into your operating system!
1. Start the server with the keyboard flag:
   ```bash
   python mediapipe_controller.py --os-keyboard
   ```
2. A **KineticLink Control Mapper** settings window will appear on your screen.
3. Use the dropdown menus to map the 5 clinical gestures (Pinch, Flick, Sweeps) to whatever keys your desktop game uses (e.g., WASD, Arrow Keys, Space).
4. Click **"Start Tracker"**. The camera window will open, and your AI movements will now trigger your custom keystrokes!

### Exiting
To safely shut down the tracker and finalize the CSV log file, make sure the Python video window is in focus and press either the **`Q`** key or the **`ESC`** key.

## 📈 Motion Performance Dashboard

After collecting data, visualize your performance metrics using the built-in dashboard:

### Automatic Dashboard Display
- **After Game Completion**: The dashboard automatically appears when you finish a game session
- **Real-time Analytics**: View your performance data immediately after completing exercises
- **Auto-load Latest Data**: The dashboard automatically loads your most recent rehab session data

### Manual Access
1. Start the local server:
   ```bash
   npm start
   ```
2. Open your browser and navigate to `http://localhost:3000/dashboard.html`
3. Or click the "View Performance Dashboard" link from the main game page

### Dashboard Features
- **📊 Interactive Charts**: Time-series visualizations for all motion metrics
- **🎨 Beautiful Design**: Modern, colorful interface with smooth animations
- **📈 Performance Stats**: Key metrics summary (session time, averages, totals)
- **🔄 Auto-loading**: Automatically detects and loads the latest CSV file
- **📱 Responsive**: Works on desktop and mobile devices

The dashboard provides a comprehensive view of your motion performance for analysis and tracking progress.