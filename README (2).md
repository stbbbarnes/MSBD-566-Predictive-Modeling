# Predicting Heart Disease with Social Determinants of Health (NHANES 2021–2023)

**Authors:** Sheretta Barnes and Christina Ndayisenga

Machine-learning project that tests whether adding social determinants of health (SDoH), and then clinical measurements, improves prediction of self-reported heart disease in U.S. adults.

## Research question

Does adding social context (income, education, food security, insurance, usual source of care) to traditional risk factors improve heart-disease prediction? What does adding clinical measures (BMI, HDL, total cholesterol) add on top of that?

## Data

- **Source:** CDC National Center for Health Statistics, National Health and Nutrition Examination Survey (NHANES), August 2021–August 2023.
- **Files:** 12 component files (demographics, questionnaires, examination and laboratory data) merged on the participant ID (`SEQN`).
- **Outcome (`heart_disease`):** 1 if the participant reports any of congestive heart failure (`MCQ160B`), coronary heart disease (`MCQ160C`), angina (`MCQ160D`) or heart attack (`MCQ160E`); 0 if all four are "no"; otherwise excluded.
- **Sample flow:** 11,933 participants → 7,770 with a valid outcome → 6,377 with complete questionnaire and SDoH data → 4,671 with complete clinical data (429 cases, about 9% prevalence).

NHANES data are public: https://wwwn.cdc.gov/nchs/nhanes/ . Download the `.xpt` files and place them where the notebook expects them.

## Models

Three nested feature sets, each containing the one before it:

| Model | Features |
|---|---|
| 1. Traditional | Age, sex, race/ethnicity, hypertension, diabetes, smoking, sedentary time |
| 2. + SDoH | Model 1 + education, income-to-poverty ratio, food security, insurance, usual source of care |
| 3. + Clinical | Model 2 + BMI, HDL cholesterol, total cholesterol |

Algorithms: logistic regression, decision tree, random forest, XGBoost.

## Methods

1. **Cleaning:** survey special codes (7, 9, 77, 99, 7777, 9999) set to missing; SAS near-zero values set to 0; three-level smoking variable (never, former, current); dummy coding with `drop_first`.
2. **Class imbalance (about 10:1):** class weights, `scale_pos_weight` and SMOTE were compared.
3. **Stage 1, 5-fold stratified CV (seed 42):** 4 algorithms × 3 models with fixed settings. Scaling is fit inside each fold. A cutoff for 80% recall is chosen on the training fold. Metrics come from pooled out-of-fold predictions.
4. **Stage 2, nested CV (5 outer, 3 inner folds):** grid search for a regularized random forest (`min_samples_leaf` 20/30/50, `max_depth` 4/6/8) and XGBoost (`max_depth` 3/5, `learning_rate` .05/.1, `n_estimators` 50/100).
5. **Diagnostics:** ROC-AUC, PR-AUC, Brier score, calibration curves, 1,000-resample bootstrap 95% CIs, AUC by sex and race/ethnicity, SHAP, probability histograms, decision curve analysis.

## Key results (4,671 participants, out-of-fold)

Nested CV:

| Algorithm | Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Logistic regression | 1 | 0.834 | 0.309 | 0.072 |
| Logistic regression | 2 | 0.836 | 0.307 | 0.072 |
| Logistic regression | 3 | 0.849 | 0.326 | 0.071 |
| Regularized random forest | 1 | 0.824 | 0.291 | 0.158 |
| Regularized random forest | 2 | 0.827 | 0.291 | 0.151 |
| Regularized random forest | 3 | 0.851 | 0.340 | 0.129 |
| Tuned XGBoost | 1 | 0.821 | 0.283 | 0.164 |
| Tuned XGBoost | 2 | 0.822 | 0.292 | 0.163 |
| Tuned XGBoost | 3 | 0.847 | 0.329 | 0.142 |

Takeaways:

- Clinical measures help most (+0.014 to +0.024 ROC-AUC); adding SDoH adds only +0.001 to +0.003, and bootstrap intervals overlap.
- With regularization, the three algorithms end up within 0.004 ROC-AUC of each other on Model 3.
- Only logistic regression is well calibrated. The class-weighted forest and XGBoost over-predict risk and need recalibration before their outputs are read as probabilities.
- Performance varies by subgroup (lowest for Mexican American participants).

## How to run

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib jupyter
jupyter notebook Complete_Heart_disease_code.ipynb
```

Use **Kernel → Restart & Run All** so every result comes from the same data.

## Known limitations and issues

- Cross-sectional, self-reported outcome; complete-case analysis (about 30% missing labs); only 429 cases in the clinical sample.
- Survey strata and PSUs are not used in variance estimation.
- In the nested-CV function the recall cutoff is chosen on the same fold it is scored on, so out-of-fold sensitivity is identical (0.8019) for every model. Treat it as an artifact, not a result. Stage 1 chooses cutoffs on training folds.
- Cell 5 of an earlier run printed different cohort sizes (5,958 / 4,376) than the later results (6,377 / 4,671). Re-run all cells to make them agree.
- Cells 111, 112 and 114 repeat the same Model 3 nested-CV run and can be deleted.

## Future work

Multiple imputation for missing labs; pool NHANES cycles for more cases and external validation; use the full survey design; recalibrate class-weighted models and choose cutoffs out-of-fold; add housing and neighborhood SDoH measures; paired tests of Model 2 vs Model 1.

## References

- Braveman, P., & Gottlieb, L. (2014). The social determinants of health: It's time to consider the causes of the causes. *Public Health Reports, 129*(Suppl 2), 19–31.
- Heidenreich, P. A., et al. (2011). Forecasting the future of cardiovascular disease in the United States. *Circulation, 123*(8), 933–944.
- CDC National Center for Health Statistics. National Health and Nutrition Examination Survey, 2021–2023.
