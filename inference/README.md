##  Inference Pipeline

This project supports both **one-stage** (YOLO-only) and **two-stage** (YOLO + classifier) inference pipelines for vehicle detection and classification. Several scripts were developed to perform live inference, benchmark model performance, and visualize results.

###  Inference Scripts Overview

| Script                          | Description |
|---------------------------------|-------------|
| `inference.py`                  | Performs live inference using only the YOLO11n object detection model. Annotates frames with bounding boxes and displays them in real time. |
| `2stage_inference.py`           | Implements a two-stage inference pipeline: detects objects using YOLO11n, then classifies each detection using a trained classifier model. |
| `2stagemetrics.py`              | Runs the full two-stage inference pipeline and logs performance metrics such as average FPS, inference time per frame, and processing latency. |
| `classifier_metric_inference.py` | Evaluates and logs inference metrics specifically for the classifier model, excluding the YOLO detection stage. |
| `Inference_data.py`            | Records detailed runtime data including CPU/GPU usage, FPS, and inference speed. Also saves annotated output as video and logs runtime data in `.json` format. |

---

### ▶ Running Inference

After completing training (see `train/README.md`), ensure your trained model file (`.pt`) is placed in the inference directory.

1. Move the model (e.g., `your_model.pt`) to the root directory of the inference pipeline.
2. Rename it to `global_model.pt` or update the corresponding file path in the scripts.

#### One-Stage Inference (YOLO only):

```bash
python3 inference.py
```

#### Two-Stage Inference (YOLO + Classifier):

```bash
python3 2stage_inference.py
```

#### Benchmark Two-Stage Pipeline:

```bash
python3 2stagemetrics.py
```

#### Benchmark Classifier Only:

```bash
python3 classifier_metric_inference.py
```

#### Log Inference Runtime + Save Output Video:

```bash
python3 Inference_data.py
```

---

###  Output

- Inference videos with bounding boxes and class labels are saved to the output directory (defined in each script).
- Logs containing FPS, CPU/GPU usage, and inference timing are saved as `.json` file.
- Example:
  ```
  output/
  ├── annotated_video.mp4
  ├── metrics_log.json
  ```

---

###  Two-Stage Pipeline Logic

1. **YOLO11n** detects bounding boxes for all vehicles in a frame.
2. Each cropped region is passed to a **custom classifier** model.
3. Final labels  are added on top of the original YOLO output.




