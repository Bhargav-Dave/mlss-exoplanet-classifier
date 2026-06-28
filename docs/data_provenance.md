# Data Provenance

The original notebook describes the data as Kepler Public Data Repository / Kepler Object of Interest (KOI) cumulative exoplanet data combined with separate habitable and non-habitable planet lists.

This repository expects three local raw files:

```text
data/raw/cumulative-exoplanets.xlsx
data/raw/habitable_planets_detailed_list.xlsx
data/raw/non_habitable_planets_confirmed_detailed_list.xlsx
```

Raw data is intentionally not committed to GitHub. The `data/raw/` directory is ignored except for `.gitkeep`.

The exact provenance, version, download date, and license terms of the three local Excel files are unknown from the current repository contents unless visible inside the files themselves. Future work should document where each file came from, when it was downloaded, and what license or terms apply.

NASA Exoplanet Archive provides KOI table column documentation, including definitions for many KOI columns used in this project. The repository does not claim more provenance detail than the original notebook and local files support.

The current workflow loads the raw files locally, constructs the supervised target label from the two supplied label lists, and merges labels onto selected KOI features by `kepoi_name`.
