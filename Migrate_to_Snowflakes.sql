USE ROLE ACCOUNTADMIN;
USE DATABASE SNOWFLAKE_LEARNING_DB;
USE SCHEMA PUBLIC;

CREATE OR REPLACE GIT REPOSITORY EY_PROJECT_REPO
  API_INTEGRATION = GITHUB
  ORIGIN = 'https://github.com/ahmdtrdi/water-quality-prediction.git';

ALTER GIT REPOSITORY EY_PROJECT_REPO FETCH;

CREATE OR REPLACE STAGE EXTERNAL_DATA_STAGE
  DIRECTORY = (ENABLE = TRUE)
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

COPY FILES
  INTO @EXTERNAL_DATA_STAGE/
  FROM @EY_PROJECT_REPO/branches/main/data/01-raw/water_quality_training_dataset.csv;

COPY FILES
  INTO @EXTERNAL_DATA_STAGE/
  FROM @EY_PROJECT_REPO/branches/main/data/01-raw/submission_template.csv;

COPY FILES
  INTO @EXTERNAL_DATA_STAGE/
  FROM @EY_PROJECT_REPO/branches/main/data/01-raw/landsat_features_training.csv;

COPY FILES
  INTO @EXTERNAL_DATA_STAGE/
  FROM @EY_PROJECT_REPO/branches/main/data/01-raw/landsat_features_validation.csv;

COPY FILES
  INTO @EXTERNAL_DATA_STAGE/
  FROM @EY_PROJECT_REPO/branches/main/data/01-raw/terraclimate_features_training.csv;

COPY FILES
  INTO @EXTERNAL_DATA_STAGE/
  FROM @EY_PROJECT_REPO/branches/main/data/01-raw/terraclimate_features_validation.csv;

-- 6. Verifikasi keberhasilan
LIST @EXTERNAL_DATA_STAGE; 