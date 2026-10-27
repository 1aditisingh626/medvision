# Model files required

Place these trained files in this folder (do not commit them unless you intentionally use Git LFS):

- `best_model.keras` — four-class kidney classifier
- `densenet121_3class_best.pth` — original DenseNet121 chest classifier checkpoint

The kidney classifier additionally needs `kidney_model_config.json`, which must be populated from its training artifacts. Do not guess class index order or preprocessing. See the root README.
