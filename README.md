# Plant Disease Classification - Training Guide

This project allows you to train a **plant disease classification model** from your labeled image dataset, save the trained model, and convert it into **TensorFlow Lite** format for mobile or embedded deployment.

---

## Directory Structure

```
root/
├── FullDataset/
|   ├── class_1/
│   ├── class_2/
|   └── ... (dataset before splitting)
├── data/
│   ├── train/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
│   ├── test/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
│   ├── val/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
│   └── ... (dataset after splitting)
├── split.py
├── train.py
├── test.py
├── convert.py
├── model/
|   ├── model1
│   ├── model2
|   └── ... (all saved models)
```
---

## How to Run

1. Place your full dataset inside the `/FullDataset` directory (before splitting).  
Example: Each **class** should have its own folder containing respective images.
   ```
   FullDataset/
   ├── class_1/
   │   ├── img1.jpg
   │   ├── img2.jpg
   ├── class_2/
   │   ├── img3.jpg
   │   ├── img4.jpg
   ```

3. Run `split.py` to split into `train/`, `test/` and `val/`:
   ```bash
   python split.py
   ```
   This will automatically create `train/`, `test/` and `val/` folders inside `data/`.

4. Run `train.py` to train the model:
   ```bash
   python train.py
   ```
   This will save the trained `.h5` or `.keras` model file.

5. Run `test.py` to test the trained model:
   ```bash
   python test.py
   ```

6. Run `convert.py` to convert the trained model into `.tflite`:
   ```bash
   python convert.py
   ```
   This will create a `.tflite` model file ready for mobile deployment.

---

## Parameters You Can Tweak

| Parameter | Purpose | Recommendation |
|:---------|:--------|:---------------|
| `batch_size` | Samples per batch | 16, 32, 64 based on RAM |
| `epochs` | Number of training cycles | 15-45 depending on dataset |
| `learning_rate` | Optimizer step size | 0.001 initially, change upto 0.00001 |
| `image_size` | Resize images to this size | (299, 299) standard for InceptionV3 |
| `augmentation` | Random transformations | Recommended for small datasets |
| `model_type` | CNN architecture | Simple CNN, InceptionV3, etc. |

