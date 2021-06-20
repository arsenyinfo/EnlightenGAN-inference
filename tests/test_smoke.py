import cv2

from enlighten_inference import EnlightenOnnxModel, get_relative_path


def test_smoke():
    img = cv2.imread(get_relative_path(__file__, 'chicken.jpeg'))
    model = EnlightenOnnxModel()

    processed = model.predict(img)

    assert img.shape == processed.shape
    assert img.mean() < processed.mean()
