MX Truncation Research: split presentation figures

Open MX_Truncation_Research_SVHN_CNN8_Midterm.ipynb in Jupyter or VS Code.
All four figures are already embedded. Run All recreates them and exports
PNG at 300 dpi, SVG, CSV and JSON to the relative figures/ directory.
Requirements: Python 3 and Matplotlib. No training, datasets or checkpoints.

Figures:
1. shared_scale_accuracy: four original accuracy references + four scale variants.
2. shared_scale_storage: first four original storage rows + remaining scale variants.
3. group_size_accuracy: four original accuracy references + groups 16/32/64 at x64.
4. group_size_storage: first four original storage rows + groups 16/32/64 at x64.

Reference rows follow EACH original figure. The storage figure originally
has no separate direct joint-QAT bar. Its first four rows are dedicated,
post-training truncation, truncation-aware x1, and fixed x32. This differs
from the four accuracy references and preserves the original representation.

Group size 32 at x64 reuses the existing fixed_scale64 record, with a clear
label. All original accuracy values, byte counts and provenance are unchanged.
No unverified 8x8 result has been added. Details are documented in the notebook.
