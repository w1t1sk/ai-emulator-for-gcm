# Environment Reproduction

This repo includes two conda environment files derived from the actual local environment used for the main circular-padding emulator runs.

## Files

- `environment.yml`
  The full export of the `emulator` environment. This is the closest match to the exact environment used on the original machine.

- `environment.from-history.yml`
  A lighter spec based on the conda install history. It is easier to recreate than the full export, and it now also includes the pip-installed packages present in the original working environment.

- `requirements.txt`
  A lightweight pip-style dependency list. Keep this as a convenience only, not as the primary reproducibility file.

## Recommended Usage

For the closest possible match to the original setup:

```bash
conda env create -f environment.yml
conda activate emulator
```

If the environment already exists:

```bash
conda env update -n emulator -f environment.yml --prune
conda activate emulator
```

## When To Use The Lighter File

Use `environment.from-history.yml` if:

- the full environment file is too strict for a different Linux machine
- you want a cleaner starting point
- you plan to adapt package versions while keeping the same major stack
- you still want the non-conda pip packages captured in the repo

Command:

```bash
conda env create -f environment.from-history.yml
conda activate emulator
```

## Notes

- The full `environment.yml` includes Linux/CUDA-specific build pins because it was exported from the original conda environment.
- The lighter `environment.from-history.yml` keeps the manually selected conda packages and also carries the pip-installed packages from the original environment.
- The environment files were renamed to `emulator` for repository consistency.
- For this repo, the conda YAML files are the preferred way to recreate the environment.
