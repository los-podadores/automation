# Photographic simulation fixtures

These are real photographic crops, not generated pictures or YOLO annotations.

- Person: the user's `PODADORAS.mp4`, frame at 80 seconds, pixels
  `(550, 80)` to `(760, 478)`. Provided for this project.
- Dog: the existing local ROD test image `IMG_20296.jpg`, pixels `(165, 0)`
  to `(515, 640)`. Dataset citation: Zandi et al., ROD-Dataset (2026),
  DOI `10.34740/kaggle/dsv/16214289`; original dataset citations are in
  `datasets/rod/citations.txt`. No additional redistribution rights are claimed.

`sources.json` records provenance and SHA-256 of the prepared JPEG files.
`prepare_posters.py` regenerates the crops from the original local sources.
`panel.obj` is the project's own double-sided UV-mapped rectangular mesh.

The dog panel is deliberately enlarged to 2 m high for this camera test. These
two fixed pictures are demonstration fixtures, not an independent accuracy
benchmark or a validation of real-world pedestrian/animal safety.
