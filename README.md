# Student Performance Prediction - First Progress Report Code

This codebase matches the **first progress report** stage of the project.

## What is already implemented
- Project structure and Python environment setup
- Dataset loading
- Basic data inspection
- Missing-value checks
- Categorical/numerical column detection
- Preprocessing pipeline setup
- Label creation for classification
- Train/test split scaffolding
- Exploratory data analysis outputs (summary text + plots)

## What is intentionally not finished yet
To stay aligned with the written progress report, the full model comparison stage is **not finished yet**.
The next step is to train and evaluate:
- Logistic Regression (baseline)
- Decision Tree
- Random Forest

A starter file for that stage is included as `train_models.py`.

## Expected dataset
Place your dataset CSV inside the `data/` folder and update the filename in the command if needed.

Example expected columns (flexible):
- studytime
- failures
- absences
- G1, G2, G3
- school, sex, address, famsize, Pstatus, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic

This project is designed to work with common student performance datasets such as the UCI student performance dataset.

## Run order
1. Put your CSV in `data/`
2. Run EDA:
   ```bash
   python src/eda.py --input data/student-mat.csv --output outputs
   ```
3. Build the cleaned dataset and preprocessing artifacts:
   ```bash
   python src/preprocess.py --input data/student-mat.csv --output outputs
   ```
4. Later, train baseline models:
   ```bash
   python src/train_models.py --input outputs/processed_student_data.csv --target performance_label
   ```

## Notes
- The script creates a **classification label** from `G3` if a target column is not already present.
- By default, `G3 >= 10` becomes label `1` and `G3 < 10` becomes label `0`.
- You can adjust this threshold later depending on your final project design.
