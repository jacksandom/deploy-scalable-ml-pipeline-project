# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- This model uses a Random Forest Classifier to predict salary bands based on US census data
- The model is developed as part of the Udacity ML DevOps Nanodegree

## Intended Use
- This project is for the purpose of the Udacity project only
- It is an example of how ML models can be deployed as a FastAPI application with CI/CD fundamentals through GitHub Actions and Heroku

## Training Data
- The training data is 80% of the original census data set

## Evaluation Data
- The evaluation data is 20% of the original census data set

## Metrics
- On the test dataset: precision was **0.73**, recall was **0.62** and fbeta was **0.67**

## Ethical Considerations
- Evaluation was performed on different slices by education to look for model bias. More investigation is needed to understand this for other categories within the data set. 

## Caveats and Recommendations
- This model should not be used to form any conclusions about the subject matter as it is only for demonstration purposes.
