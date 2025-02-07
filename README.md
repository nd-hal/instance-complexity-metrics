# instance-complexity-metrics

1. Download the instance-level prediction (i.e., "ledger") data from [Zenodo](https://zenodo.org/records/14834909)
    * Warning: downloading a large file (4.0 GB) can take up to 25min
    * Alternatively, researchers can use the top few rows shown on Zenodo to create their own ledgers from instance-level model outputs
3. Download the input data (i.e., DataCVFolds.zip and Data_SurveyPlusDemographics.txt) from the [FairPsych NLP repository](https://github.com/nd-hal/fair-psych-nlp/tree/main/Data) and place Data_SurveyPlusDemographics.txt into the DataCVFolds/ directory.
    * i.e., the full path to the demographics file should be data/DataCVFolds/Data_SurveyPlusDemographics.txt
4. Add both ledger_len.csv and DataCVFolds/ to the data/ directory of this instance-complexity-metrics/ repository
5. Run the cells in Artifact_Preprocessing.ipynb
6. Navigate to artifact_code/ and run code in each folder corresponding to desired table / figure to replicate the results

**NOTE;** The clinical depression data with the depression detection task / 'wer' (Word Error Rate) score variable is not publicly available according to the data collection IRB.  Please reach out to the authors to discuss data access.