# README for Overleaf

Upload this zip file to Overleaf:

```text
mlss-exoplanet-classifier-enhanced-report.zip
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

The main document is:

```text
main.tex
```

If Overleaf does not auto-detect it, set `main.tex` as the main document in the Overleaf project menu.

## Compiler and references

Use pdfLaTeX. The report uses BibTeX-style references:

- bibliography file: `references.bib`
- bibliography command: `\bibliography{references}`
- bibliography style: `plain`

Overleaf usually runs the needed LaTeX/BibTeX passes automatically. If citations appear as question marks after the first compile, compile again.

## Assets

Figures live in:

```text
figures/
```

Tables and copied classification-report text files live in:

```text
tables/
```

The report uses relative paths such as `figures/roc_curve.png`.

## Exclusions

This package does not include raw data, model binaries, caches, notebook checkpoints, generated training code caches, or local Codex reports. The raw Excel files remain local-only in the repository and ignored by Git.
