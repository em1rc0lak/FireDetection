# Fire Detection using YCbCr Color Space

An extended implementation of the fire detection algorithm proposed by **Çelik & Ma (2008)**, enhanced with motion and flicker detection for improved accuracy in video streams.

## Overview

This project implements a real-time fire pixel detection system using YCbCr color space analysis combined with temporal features (motion and flicker). It processes video frames to identify fire regions based on predefined color rules and dynamic pixel behavior.

## Features

- **YCbCr color-based fire detection** using 5 rule conditions
- **Motion detection** via frame differencing
- **Flicker detection** using running average comparison
- **Real-time bounding box** around detected fire regions
- **Multi-video batch processing**

## Detection Rules

The algorithm applies the following conditions (R1–R5):

| Rule | Condition |
|------|-----------|
| R1 | Y ≥ Cb |
| R2 | Cr ≥ Cb |
| R3 | Y ≥ Ȳ, Cb ≤ C̄b, Cr ≥ C̄r |
| R4 | \|Cb − Cr\| ≥ τ |
| R5 | Cb bounded by polynomial functions of Cr |

## Temporal Features

- **Motion Mask**: Detects pixel changes between consecutive frames
- **Flicker Mask**: Identifies intensity fluctuations using exponential moving average

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Usage

```bash
python fire_detection.py
```

Place input videos in the `fire_videos/` folder (named `1.mp4`, `2.mp4`, etc.). Press `q` to skip to the next video.

## Configuration

```python
ADD_MOTION_AND_FLICKER = True  # Enable/disable temporal analysis
```

## Reference

> Extended from: Çelik, T., & Ma, K. K. (2008). *Computer vision based fire detection in color images.*

## License

MIT License
