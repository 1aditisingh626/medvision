import unittest


def validate_four_class_output(output, class_names):
    """Small pure validation contract mirrored by the app's Keras inference."""
    import numpy as np
    values = np.asarray(output)
    if values.shape != (1, 4):
        raise ValueError(f"Expected (1, 4), got {values.shape}")
    if len(class_names) != 4:
        raise ValueError("Exactly four class labels are required")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite scores")
    return class_names[int(np.argmax(values[0]))]


class KidneyOutputMappingTests(unittest.TestCase):
    def test_four_class_output_uses_supplied_index_mapping(self):
        labels = ["Stone", "Tumor", "Normal", "Cyst"]
        self.assertEqual(validate_four_class_output([[0.1, 0.2, 0.6, 0.1]], labels), "Normal")

    def test_binary_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected"):
            validate_four_class_output([[0.8, 0.2]], ["Cyst", "Normal", "Stone", "Tumor"])

    def test_nonfinite_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_four_class_output([[0.1, 0.2, float("nan"), 0.7]], ["Cyst", "Normal", "Stone", "Tumor"])


if __name__ == "__main__":
    unittest.main()
