# GlomExtractor
Implementation of the methods described in "GlomExtractor: a versatile tool for extracting glomerular patches from whole slide kidney biospy images"

<p align="center">
    <img src="src/img/Figure1.png" width="600" alt="Figure 1: Workflow illustration of the code">
</p>


## To run the code
### Create the env
```bash
conda env create -f src/environment.yml
```
### Run
```bash
python main "../INPUT/"
```

## Folder Structure
### Dataset mode (-d):
    INPUT/
        Images/
            <slide files or images>
        Annotations/
            <annotation files with same stem as image>

> Example:
    >>INPUT/Images/case_001.svs
    >>
    >>INPUT/Annotations/case_001.json

### Single file mode (-f)
> Example: INPUT/Images/case_001.svs
    >> The code will look for any matching annotation:
    >>
    >>INPUT/Annotations/case_001.*

## Command-line options

### Paths and output
- `path` (positional): Dataset folder (contains `Images/` + `Annotations/`) or a single slide/image path.
- `-o, --out_folder`: Output root folder.  
  Default: `../OUTPUT/`

### Export settings
- `-w, --patch_size`: Patch width in pixels used for extraction (and for white/blur metrics).  
  Default: `1000`
- `--out_size`: Size of exported PNGs (images/masks/markups).  
  Default: `224`
- `--no_masks`: Disable saving masks (masks are ON by default).
- `--no_markups`: Disable saving markups (markups are ON by default).

### CSV
- `--no_csv`: Disable saving `annotations_full.csv` (CSV is ON by default).

### Filtering thresholds (used for the `Filtered*` output folders)
- `--min_area`: `FilteredSmall` keeps `area >= min_area`.  
  Default: `20000`
- `--max_area`: `FilteredLarge` keeps `area <= max_area`.  
  Default: `1200000`
- `--max_white`: `FilteredWhite` keeps `percWhite <= max_white`.  
  Default: `0.75`
- `--min_lap`: `FilteredBlur` keeps `LaplacianVariance >= min_lap`.  
  Default: `400`
- `--min_circ_q`: `FilteredCircularity` keeps `circularity >= quantile(min_circ_q)` computed from the full dataset.  
  Default: `0.10`
- `--min_ch_q`: `FilteredCHmetric` keeps `chMetric >= quantile(min_ch_q)` computed from the full dataset.  
  Default: `0.10`




