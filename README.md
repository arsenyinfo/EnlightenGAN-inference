# EnlightenGAN-inference

Very minimalistic wrapper for [EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN) inference. 
It uses carefully converted pretrained weights (+ baked in preprocessing) from the original repo and only requires `onnxruntime` as inference engine.   

## Installation

`pip install git+https://github.com/arsenyinfo/EnlightenGAN-inference`

## Usage

```python
from enlighten_inference import EnlightenOnnxModel
import cv2

img = cv2.imread('/path/to/image.jpg')
model = EnlightenOnnxModel()

processed = model.predict(img)
``` 
