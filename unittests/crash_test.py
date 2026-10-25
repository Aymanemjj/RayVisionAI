import unittest
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory

MODEL_PATH = "../artifacts/ft_model.keras"
TEST_DATA_PATH = "../data/test_data"
MIN_CORRECT = 7


class CrashTest(unittest.TestCase):

    def test_responsive_model(self):
        model = load_model(MODEL_PATH)

        data = self.get_images()
        data = self.convert_to_pandas(data)
        data["image"] = data["image"].map(self.scale_vector)

        X_test = np.stack(data["image"].to_numpy())
        y_test = data["label"].to_numpy().astype(int)

        probs = model.predict(X_test).ravel()
        preds = (probs > 0.5).astype(int)

        self.assertEqual(preds.shape, y_test.shape)
        self.assertTrue(np.isfinite(probs).all())

        correct = int((preds == y_test).sum())
        self.assertGreaterEqual(
            correct, MIN_CORRECT,
            f"Only {correct}/{len(y_test)} correct, needed {MIN_CORRECT}",
        )

    @staticmethod
    def convert_to_pandas(dataset):
        ds_labels = []
        for image, labels in dataset:
            ds_labels.append({"image": image.numpy(), "label": labels.numpy()[0]})
        new_df = pd.DataFrame.from_dict(ds_labels)
        return new_df

    @staticmethod
    def scale_vector(vecotre):
        return vecotre / 255.0

    @staticmethod
    def get_images(path=TEST_DATA_PATH):
        dataset = image_dataset_from_directory(path, label_mode='binary', batch_size=None,
                                               color_mode="rgb", image_size=(180, 180),
                                               crop_to_aspect_ratio=False, shuffle=True)
        return dataset


if __name__ == '__main__':
    unittest.main()