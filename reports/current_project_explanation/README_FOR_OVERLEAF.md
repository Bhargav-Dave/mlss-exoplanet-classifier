# README for Overleaf

This folder contains the early/current-project Overleaf report for the MLSS exoplanet habitability classifier. It explains the original notebook workflow and the historical notebook results before the project was fully modularized.

For the final enhanced report with the reproducible pipeline results, use:

```text
reports/enhanced_project_overleaf/
```

## What to upload

Upload this zip file to Overleaf:

```text
mlss-exoplanet-classifier-early-report.zip
```

The zip is structured so that `main.tex` is at the root of the archive. When opened, the zip root should contain:

```text
main.tex
references.bib
README_FOR_OVERLEAF.md
figures/
tables/
```

## Main document

The project should compile from:

```text
main.tex
```

If Overleaf does not auto-detect the main file, select `main.tex` as the main document in the Overleaf project menu.

## Compiler and references

This report uses simple pdfLaTeX-compatible packages and BibTeX-style references:

- Compiler: pdfLaTeX
- Bibliography file: `references.bib`
- Bibliography command in `main.tex`: `\bibliography{references}`
- Bibliography style: `plain`

On Overleaf, the normal compile button should run the needed LaTeX/BibTeX passes automatically. If references appear as question marks after the first compile, compile again.

## Assets

Figures live in:

```text
figures/
```

Tables live in:

```text
tables/
```

The LaTeX document uses relative paths such as:

```text
figures/01_feature_boxplots_before_outlier_removal.png
```

## Scope

This early report is useful for understanding the original notebook, target construction, features, model families, and historical notebook comparison table. It does not include the final enhanced-pipeline results.

## What is not included

This package does not include raw data, model binaries, notebook checkpoints, caches, generated training outputs, or local Codex planning reports. It does not modify the original notebook.
