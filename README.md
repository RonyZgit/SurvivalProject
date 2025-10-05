# Survival Project

In this project, I analyzed survival data from a clinical study on Primary Biliary Cirrhosis (PBC), a chronic liver disease. Our objective is to compare traditional survival models, such as the Cox Proportional Hazards model, with modern machine learning approaches, including Random Survival Forests and deep learning-based survival model (DeepHit). The dataset includes time-to-event information (death or liver transplant) and various clinical covariates for over 300 patients. Model performance by using standard metrics such as the Concordance Index (C-index) and Integrated Brier Score (IBS). 
Our analysis revealed that the treatment variable (D-penicillamine vs. placebo) did not have a significant impact on patient survival, aligning with clinical findings that questioned the drug’s efficacy. Also, Random Survival Forests achieved the best overall performance, with the lowest Integrated Brier Score and the highest C-index, suggesting strong predictive ability and good calibration.

The analysis contains:
1. Explanatory data analysis including visualizations.
2. Survival models building:
  Cox Proportional Hazards model.
  Cumulative Incidence Function (CIF) for competing risks analysis.
  Random Survival Forests (RSF).
  Deep learning-based survival models.
3. Models analysis and comparisons.
4. Conclusions

R code and graphs:
https://RonyZgit.github.io/SurvivalProject/R_code_analysis_of_PBC_dataset.html
