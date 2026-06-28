# README for Overleaf

This folder is now a real Overleaf-ready LaTeX project for the early MLSS exoplanet habitability classifier report.

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

## Figures and tables

Figures are stored in:

```text
figures/
```

Tables are stored in:

```text
tables/
```

The LaTeX document uses relative paths such as:

```text
figures/01_feature_boxplots_before_outlier_removal.png
```

## What is not included

This package does not include raw data, model binaries, notebook checkpoints, caches, generated training outputs, or local Codex planning reports. It does not modify the original notebook and does not regenerate model results.

## Caveat

This is an early explanatory report for the current notebook. It is not the enhanced/reproducible version of the full project. README creation for the GitHub repository, workflow refactoring, requirements files, tests, CLI scripts, and modeling improvements are intentionally reserved for later steps.
