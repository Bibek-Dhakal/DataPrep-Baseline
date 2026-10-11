# Data

Datasets are never committed. Download them into a local `data/` folder at the repository root (git-ignored) and point
the CLI at the file. The pipeline works with the documented example below and with any CSV that meets the
[requirements](#using-your-own-dataset).

## Example: Adult Census Income (classification)

| Item         | Value                                                                                                                                                        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Source       | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/2/adult                                                                                 |
| Download via | OpenML `adult`, version 2: https://www.openml.org/d/1590                                                                                                     |
| License      | CC BY 4.0 (as listed by UCI; check the source page for current terms)                                                                                        |
| Size         | About 48k rows, 14 feature columns                                                                                                                           |
| Target       | `class` (`<=50K` or `>50K`, a string label)                                                                                                                  |
| Why it fits  | Numeric and categorical columns, real missing values (`workclass`, `occupation`, `native-country`), skewed numeric features (`capital-gain`, `capital-loss`) |

Download and save as CSV (works in bash and PowerShell, needs network access once):

```bash
python -c "from pathlib import Path; from sklearn.datasets import fetch_openml; Path('data').mkdir(exist_ok=True); fetch_openml('adult', version=2, as_frame=True).frame.to_csv('data/adult.csv', index=False)"
```

Run the pipeline:

```bash
python preprocess_and_model.py --data data/adult.csv --target class --drop-cols fnlwgt --output-dir outputs/adult
```

`fnlwgt` is a census sampling weight rather than a person attribute, so dropping it is optional but common. The
pipeline also runs without `--drop-cols`.

## Using your own dataset

Any CSV works if it meets these requirements:

- It has a header row and one row per record.
- One column is the target. Pass it with `--target`.
- Remaining columns are features: numeric columns are scaled, every other column is treated as categorical.
- Rows with a missing target are dropped automatically. Missing feature values are imputed inside the pipeline.
- Task type is auto-detected (non-numeric target or 10 or fewer unique numeric values means classification). Override
  with `--task classification` or `--task regression`.

Before running, check the dataset for columns that should not be features, and exclude them with `--drop-cols`:

| Column type                          | Why to drop it                                              |
|--------------------------------------|-------------------------------------------------------------|
| Row IDs, customer IDs, order numbers | Unique per row; they add noise or explode one-hot encoding. |
| Columns derived from the target      | They leak the answer and give unrealistic scores.           |
| Free text, URLs, file paths          | Not meaningful as categories.                               |

The pipeline logs a warning for non-float columns with almost one unique value per row (likely IDs). It cannot detect
every case, so review the columns yourself.

```bash
python preprocess_and_model.py --data data/mydata.csv --target price --task regression --drop-cols Id,PID
```

### Datasets that need cleaning first

- Numbers stored as text (for example blank strings in a numeric column) are loaded as text and treated as
  categorical. Convert them to numeric in the CSV before running.
- Numeric codes that are really categories (for example a zip code or a class code) are scaled as numbers.
- Date and datetime columns are not parsed. Derive numeric features (year, month) or drop them.

## Other candidates

| Dataset                           | Task           | Notes                                                                        |
|-----------------------------------|----------------|------------------------------------------------------------------------------|
| Ames Housing (target `SalePrice`) | Regression     | Many categoricals and missing values. Drop ID columns such as `Id` or `PID`. |
| Telco Customer Churn              | Classification | Drop `customerID`. Convert `TotalCharges` to numeric first.                  |

Check each dataset's license on its source page before use or redistribution.
