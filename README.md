# Face-Depth Point Cloud Generator

Detects faces in video, estimates depth for each face, generates point clouds from face depth data, and overlays the depth heatmap back onto the original video.

## What it does

Instead of generating point clouds on entire scenes, this project narrows the scope by only detecting faces in video, applying depth-estimations for each face region only, generating 3D point clouds from those depth maps. The rest of the scene is left untouched, only detected faces get depth processing and visualization.

## How it works

1. **Frame capture & skipping.** OpenCV reads the source video, using `cap.grab()` to skip undecoded frames and only decoding frames that will actually be processed, avoiding unnecessary overhead.
2. **Face detection.** Each processed frame is passed through `face_recognition` to locate and crop faces.
3. **Depth inference.** Every cropped face is passed through a HuggingFace `depth-estimation` pipeline running Depth-Anything V2 (Small), producing a per-face depth map.
4. **Point cloud generation.** For each face's depth map, a 3D point cloud is generated, converting 2D depth into spatial coordinates that represent the face region in 3D space.
5. **Temporal smoothing.** Each face's depth map is smoothed against its previous frame using exponential moving average, reducing flicker.
6. **Heatmap blending.** The smoothed depth map is converted to a JET colormap and alpha-blended back onto the original face region in the video.
7. **Video encoding.** The blended heat-depth-map frames are written to output video at a framerate scaled to the source capture rate.

Note: this is near-real-time, not hard real-time. The processing is bound by inference speed, so if inference can't keep pace with the camera or source video, the live preview will lag slightly behind.

## Tech stack

* Python
* OpenCV for video I/O, frame decoding/encoding, and frame skipping
* face_recognition for face detection
* HuggingFace Transformers for the depth-estimation model pipeline
* Depth-Anything V2 for monocular depth estimation
* NumPy for depth map processing and point cloud generation

## Setup

Clone the repository:

```bash
git clone https://github.com/MurphyAmos/Face-Depth-Point-Cloud-Generator.git
cd Face-Depth-Point-Cloud-Generator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Hugging Face token:

```bash
export HF_TOKEN="your_token_here"      # macOS/Linux
setx HF_TOKEN "your_token_here"        # Windows
```

## Usage

Update the video source path in the script, then run:

```bash
python main.py
```

Output video is written to `test.mp4` in the working directory. A `test` flag is available for demo mode, which generates faux depth maps without running the real model, useful for quick visual & functionality checks.

## Known limitations & Next Fixes

* **Point cloud generation is per-face, not integrated into a single unified 3D space.** Each face gets its own point cloud, but they're not registered or merged together. Unifying multiple faces into a single 3D coordinate system is the natural next step.
* **Face tracking is index-based, not identity-based.** Temporal smoothing is keyed to a face's position in the detection list each frame, not persistent identity, so if face order shifts, smoothing state can briefly blend across different faces.
* **No UI or export for point clouds.** Point clouds are generated but not currently saved or visualized independently of the video output.

## Fixed & Updates

* **Per-face depth estimation and point cloud generation.** Each detected face now gets its own depth map and corresponding 3D point cloud, enabling spatial analysis of face regions specifically.
* **Frame skipping efficiency.** Uses `cap.grab()` for skipped frames instead of decode-then-discard, reducing unnecessary overhead.
* **Temporal smoothing per face.** Exponential moving average applied independently to each face's depth map, reducing frame-to-frame flicker.

## Motivation

After building face-aware depth overlay as a 2D visualization, the natural next question was whether that depth data could be converted into actual 3D spatial representations. Point clouds are the bridge between monocular depth estimation and real 3D understanding, and generating them per-face is a step toward building more sophisticated 3D scene understanding from 2D video.
