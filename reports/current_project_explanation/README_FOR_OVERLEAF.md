# README for Overleaf

This folder is an Overleaf-ready documentation package for the current MLSS exoplanet habitability classifier project.

## Contents

- `current_project_report.md`: Main explanatory report in Markdown.
- `references.bib`: Bibliography entries for official documentation and XGBoost references.
- `tables/model_comparison.csv`: Lightweight CSV version of the visible final model comparison table.
- `tables/model_comparison.md`: Markdown version of the visible final model comparison table.
- `figures/`: PNG figures extracted from saved notebook outputs only.
- `current_project_report.zip`: Zip archive of this report package.

## What is not included

This package does not include raw data, model binaries, notebook checkpoints, caches, or generated training outputs. It does not modify the original notebook and does not regenerate model results.

## Using with Overleaf

The main report is Markdown. Two practical options:

1. Convert `current_project_report.md` to LaTeX with Pandoc before uploading to Overleaf.
2. Upload the Markdown, figures, and `references.bib` to Overleaf and adapt them into an existing LaTeX template.

The image links in the Markdown are relative to this folder, for example:

```text
figures/02_target_class_distribution.png
```

If converting with Pandoc, keep `references.bib` in the same folder and use the citation keys already present in the report.

## Caveat

This is explanatory documentation for the current notebook. It is not the enhanced/reproducible version of the project. README creation, workflow refactoring, requirements files, tests, CLI scripts, and modeling improvements are intentionally reserved for later steps.
