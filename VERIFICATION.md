# October 6, 2026 verification

Bundled Python on Windows: `python -B -m unittest -v`: 12 tests passed, exit 0. CLI `python -B scorecard.py sample.json --out demo`: two cases processed, pooled IoU 0.7142857142857143. Outputs JSON, HTML and two computed SVGs. Invalid output-directory test deliberately prints a rejection error. No learned-model inference or physical robot test performed.
