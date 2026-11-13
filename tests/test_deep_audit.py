"""Check scene matching against known transformed and unrelated images."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, "Optional OpenCV audit tool is not installed")
class SceneMatchingTests(unittest.TestCase):
    def test_interior_cutout_is_distinguished_from_frame_and_grayscale(self):
        from scripts.audit_dataset_deep import scan_one_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (320, 240), (80, 140, 90))
            draw = ImageDraw.Draw(image)
            draw.rectangle((60, 60, 100, 100), fill="black")
            draw.rectangle((0, 0, 319, 239), outline="black", width=8)
            image.save(root / "cutout.png")
            image.convert("L").convert("RGB").save(root / "grayscale.png")
            row = {"path": "cutout.png", "id": "cutout.png", "split": "train"}
            result = scan_one_artifacts((root, row))
            self.assertEqual(1, result["interior_black_rectangles"])
            self.assertEqual(1, result["large_black_border_components"])
            self.assertFalse(result["near_grayscale"])
            row.update(path="grayscale.png", id="grayscale.png")
            self.assertTrue(scan_one_artifacts((root, row))["near_grayscale"])

    def test_perspective_rotation_jpeg_match_and_unrelated_negative(self):
        from scripts.audit_dataset_deep import extract_one, verify_pair

        rng = np.random.default_rng(123)
        images = []
        for _ in range(2):
            image = Image.new("RGB", (384, 320), "#c0c5b5")
            draw = ImageDraw.Draw(image)
            for index in range(100):
                x, y = rng.integers(10, 280, 2)
                radius = int(rng.integers(4, 25))
                color = tuple(int(v) for v in rng.integers(20, 240, 3))
                draw.ellipse((x, y, x + radius, y + radius), fill=color, outline="black")
                draw.text((x, y), str(index), fill="white")
            images.append(np.asarray(image))
        transform = np.array([[0.95, 0.03, 4], [-0.02, 0.96, 8], [0.00003, 0.00004, 1]], np.float32)
        transformed = cv2.warpPerspective(images[0], transform, (384, 320))
        transformed = cv2.rotate(transformed, cv2.ROTATE_180)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            all_desc, all_points, rows, offsets = [], [], [], [0]
            for i, image in enumerate((images[0], transformed, images[1])):
                name = f"scene_{i}.jpg"
                Image.fromarray(image).save(root / name, quality=30 if i == 1 else 92)
                row = {"path": name, "id": name, "label": 0, "split": "train"}
                _, desc, points, _, _, row = extract_one((root, row, 448, 240))
                rows.append(row); all_desc.append(desc); all_points.append(points)
                offsets.append(offsets[-1] + len(desc))
            data = {"descriptors": np.concatenate(all_desc), "points": np.concatenate(all_points),
                    "offsets": np.array(offsets)}
            hashes = np.full((3, 3), 20, np.uint8)
            similarity = np.zeros((3, 3), np.float32)
            positive = verify_pair((0, 1), data, rows, hashes, similarity, similarity)
            negative = verify_pair((0, 2), data, rows, hashes, similarity, similarity)
            self.assertIsNotNone(positive)
            self.assertTrue(positive["strict_scene"])
            self.assertTrue(negative is None or not negative["broad_scene"])


if __name__ == "__main__":
    unittest.main()
