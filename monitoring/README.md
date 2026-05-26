# Prediction Logging & Drift Monitoring Schema

This directory contains the prediction logs and reference files used to monitor data drift in production.

## File Breakdown
* **`reference_predictions.csv`**: Baseline prediction distributions computed from the clean validation set during model registration.
* **`prediction_log.csv`**: Live production prediction metrics logged by the FastAPI inference server (`serve.py`) in real time.

---

## Column Descriptions

| Column Name | Type | Description |
| :--- | :--- | :--- |
| **`timestamp`** | String | The exact ISO 8601 timestamp of when the inference took place. |
| **`num_detections`** | Integer | The total count of defect bounding boxes detected on the PCB. |
| **`avg_confidence`** | Float | The average model confidence score (0.0 to 1.0) across all detections on this image. See details below. |
| **`max_confidence`** | Float | The highest confidence score (0.0 to 1.0) among all detections on this image. See details below. |
| **`pred_open`** | Integer | Count of **open circuit** defects detected. |
| **`pred_short`** | Integer | Count of **short circuit** defects detected. |
| **`pred_mousebite`** | Integer | Count of **mousebite** defects detected. |
| **`pred_spur`** | Integer | Count of **spur** defects detected. |
| **`pred_spurious_copper`** | Integer | Count of **spurious copper** defects detected. |
| **`pred_pin_hole`** | Integer | Count of **pin hole** defects detected. |
| **`avg_bbox_area`** | Float | Average normalized area of the bounding boxes relative to the image size. |
| **`pass_fail`** | Integer | Binary decision: `0` = PASS (0 defects detected), `1` = FAIL (1+ defects detected). |

---

## Understanding Model Confidence Scores

When YOLO detects a defect on a PCB, it outputs a bounding box and a **confidence score** between `0.0` (0%) and `1.0` (100%), which represents how sure the model is that the detected region contains that specific defect.

### Average Confidence (`avg_confidence`)
This is the mathematical average of all confidence scores on a single PCB image.
* **Example**: If a PCB has 3 defects detected with confidence scores of `0.45` (45%), `0.70` (70%), and `0.92` (92%), the `avg_confidence` is:
  $$\text{avg\_confidence} = \frac{0.45 + 0.70 + 0.92}{3} = 0.69 \text{ (69\%)}$$
* **MLOps Value**: If the average confidence in production begins to drop over time (e.g. from 80% to 50%), it suggests that the model is becoming generally uncertain about its detections—a clear indicator of **model/concept drift** (often caused by changes in camera lighting, angle, or new PCB board layouts).

### Max Confidence (`max_confidence`)
This is the confidence score of the single most certain defect detected on the PCB.
* **Example**: In the example above with scores of `0.45`, `0.70`, and `0.92`, the `max_confidence` is **`0.92` (92%)**.
* **MLOps Value**: Tracking the max confidence tells us whether the model is still capable of making highly certain predictions. If both the `avg_confidence` and `max_confidence` drop together, it indicates a severe performance decay. If `max_confidence` remains high but `avg_confidence` drops, it means the model is making many weak, low-certainty detections.

---

## Drift Detection Features

The Evidently AI drift monitor uses the following features from `prediction_log.csv`:

| Feature | Why it is used |
| :--- | :--- |
| `avg_confidence` | Reflects how familiar the model finds the incoming images. A drop signals the model is seeing patterns outside its training distribution. |
| `avg_bbox_area` | Reflects the physical scale of defects. A shift indicates changes in camera setup, PCB layout, or defect type — all signs of input distribution change. |
| `pass_fail` | Captures the aggregate failure rate. A shift in the proportion of failed boards indicates a population-level change that justifies retraining. |
