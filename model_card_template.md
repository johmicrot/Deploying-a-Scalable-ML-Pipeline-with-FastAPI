# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf



## Model Details

Model Type: Random Forest Classifier
Developed By: John Rothman
Model Version: 1.0
Date: Nov 2024

## Intended Use

Primary Use Case: Predicting whether an individual's income exceeds $50,000 per year based on census data.
Users: Data scientists, researchers, and policymakers interested in socioeconomic trends.
Domains: Socioeconomic studies, labor economics, demographic analysis.
Usage Limitations: Not intended for making decisions affecting individuals without proper ethical considerations. Should not be used as the sole basis for employment, credit, or insurance decisions.

## Training Data

Source: The model was trained on the UCI Adult Census dataset.
Number of Instances: 
Dataset Characteristics:

    Number of Instances: 48842.
    Features: A mix of categorical and numerical features such as age, workclass, education, marital-status, occupation, relationship, race, sex, hours-per-week, and native-country.
    Target Variable: Salary bracket (<=50K or >50K).


## Evaluation Data

Source: A subset of the dataset reserved for testing, comprising approximately 20% of the original data.
Data Splitting: The data was split into training and testing sets using an 80-20 split.

## Metrics

Overall Model Performance

    Precision: x
    Recall: x
    F1 Score: x


Performance on Slices of Data

The model's performance was evaluated on slices of data where categorical features are held constant. Below are examples of performance metrics for specific slices:
Education Level

    Education: Bachelors
        Precision: x
        Recall: x
        F1 Score: x
    Education: HS-grad
        Precision: x
        Recall: x
        F1 Score: x

Gender

    Sex: Female
        Precision: x
        Recall: x
        F1 Score: x
    Sex: Male
        Precision: x
        Recall: x
        F1 Score: x

## Ethical Considerations

Bias and Fairness: The model may reflect existing societal biases present in the training data, such as gender or racial disparities. It's important to assess and mitigate any unfair biases in predictions.
Privacy: The data contains sensitive personal information. Ensure compliance with data protection laws like GDPR when handling and deploying the model.
Transparency: Users should understand how the model makes predictions. Consider providing explanations for individual predictions when deploying the model.
Accountability: Decisions based on the model should involve human oversight. The model should not be the sole basis for high-stakes decisions affecting individuals.

## Caveats and Recommendations

Generalization: The model may not perform well on data from different distributions or populations not represented in the training data.
Data Quality: Any inaccuracies or biases in the training data can affect model performance.
Regular Updates: The model should be retrained periodically with new data to maintain performance over time.
Ethical Use: The model should be used responsibly, keeping in mind its limitations and the potential impact on individuals and groups.