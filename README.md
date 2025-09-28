# ASL Sign Language Recognition

This project implements a **deep learning-based American Sign Language (ASL) alphabet recognition system**. The model can classify 29 classes: 26 letters (A-Z) plus `space`, `del`, and `nothing`. The system is trained using transfer learning with EfficientNetB0 and strong data augmentation techniques including **CutMix**. Mixed precision training is also used to speed up training and reduce memory usage.

---

## Features

- Recognizes 29 ASL classes: A-Z, space, delete, and nothing
- Uses **MobileNet v2** pre-trained on ImageNet
- Strong **data augmentation**:
  - Rotation, zoom, shear, brightness adjustment
  - Horizontal flip
  - CutMix augmentation
- Mixed precision training for faster performance
- Real-time predictions possible (with webcam integration)
- Saved trained model in `.h5` format

---

## Dataset

The model uses the [ASL Alphabet dataset](https://www.kaggle.com/grassknoted/asl-alphabet) structured as:

```
asl_alphabet_train/
├── A/
├── B/
├── ...
├── nothing/
```

- Training and validation folders should follow the same structure.
- Each subfolder contains images corresponding to that class.

---

## Requirements

```bash
tensorflow>=2.13
numpy
opencv-python
matplotlib
```

You can also install from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Training

The training notebook (`Training.ipynb`) performs:

1. Loading the dataset from folders.
2. Applying strong data augmentation and CutMix.
3. Building an EfficientNetB0 transfer learning model.
4. Training with mixed precision.
5. Saving the final trained model.

Example of starting training:

```python
history = model.fit(
    train_cutmix_gen,
    validation_data=val_gen,
    epochs=20
)
```

---

## Inference

To make predictions on new images:

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("asl_transfer_efficientnet_cutmix_mixedprecision.h5")

img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (96, 96))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
pred_class = np.argmax(pred)
print("Predicted class:", pred_class)
```

---

## Model Files

- `asl_final_model.h5` → original ASL model
- `asl_transfer_model_final.h5` → transfer learning model
- `asl_transfer_efficientnet_cutmix_mixedprecision.h5` → final trained model with CutMix and mixed precision

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Author

**Mohamed Makrani**
