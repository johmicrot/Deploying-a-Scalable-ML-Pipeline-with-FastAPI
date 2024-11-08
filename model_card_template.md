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

Performance on Slices of Data

```
Feature              Value                                    Count  Precision     Recall         F1
-----------------------------------------------------------------------------------------------
workclass            ?                                          389     0.6667     0.0476     0.0889
workclass            Federal-gov                                191     1.0000     0.1286     0.2278
workclass            Local-gov                                  387     1.0000     0.1273     0.2258
workclass            Private                                  4,578     1.0000     0.1179     0.2109
workclass            Self-emp-inc                               212     1.0000     0.3136     0.4774
workclass            Self-emp-not-inc                           498     1.0000     0.1338     0.2360
workclass            State-gov                                  254     1.0000     0.2192     0.3596
workclass            Without-pay                                  4        N/A        N/A        N/A
education            10th                                       183     1.0000     0.0000     0.0000
education            11th                                       225     1.0000     0.0909     0.1667
education            12th                                        98     1.0000     0.0000     0.0000
education            1st-4th                                     23     1.0000     1.0000     1.0000
education            5th-6th                                     62     1.0000     0.0000     0.0000
education            7th-8th                                    141     1.0000     0.0000     0.0000
education            9th                                        115     1.0000     0.0000     0.0000
education            Assoc-acdm                                 198     1.0000     0.0851     0.1569
education            Assoc-voc                                  273     1.0000     0.0159     0.0312
education            Bachelors                                1,053     0.9889     0.1978     0.3296
education            Doctorate                                   77     1.0000     0.2632     0.4167
education            HS-grad                                  2,085     1.0000     0.0145     0.0286
education            Masters                                    369     1.0000     0.2464     0.3953
education            Preschool                                   10     1.0000     1.0000     1.0000
education            Prof-school                                116     1.0000     0.3929     0.5641
education            Some-college                             1,485     1.0000     0.0650     0.1220
marital-status       Divorced                                   920     1.0000     0.0000     0.0000
marital-status       Married-AF-spouse                            4        N/A        N/A        N/A
marital-status       Married-civ-spouse                       2,950     0.9954     0.1649     0.2829
marital-status       Married-spouse-absent                       96     1.0000     0.0000     0.0000
marital-status       Never-married                            2,126     1.0000     0.0000     0.0000
marital-status       Separated                                  209     1.0000     0.0000     0.0000
marital-status       Widowed                                    208     1.0000     0.0000     0.0000
occupation           ?                                          389     0.6667     0.0476     0.0889
occupation           Adm-clerical                               726     1.0000     0.0625     0.1176
occupation           Armed-Forces                                 3        N/A        N/A        N/A
occupation           Craft-repair                               821     1.0000     0.0608     0.1146
occupation           Exec-managerial                            838     1.0000     0.1889     0.3178
occupation           Farming-fishing                            193     1.0000     0.0714     0.1333
occupation           Handlers-cleaners                          273     1.0000     0.0833     0.1538
occupation           Machine-op-inspct                          378     1.0000     0.0213     0.0417
occupation           Other-service                              667     1.0000     0.0000     0.0000
occupation           Priv-house-serv                             26     1.0000     1.0000     1.0000
occupation           Prof-specialty                             828     1.0000     0.2219     0.3633
occupation           Protective-serv                            136     1.0000     0.0238     0.0465
occupation           Sales                                      729     1.0000     0.1250     0.2222
occupation           Tech-support                               189     1.0000     0.0392     0.0755
occupation           Transport-moving                           317     1.0000     0.0781     0.1449
relationship         Husband                                  2,590     0.9953     0.1803     0.3054
relationship         Not-in-family                            1,702     1.0000     0.0000     0.0000
relationship         Other-relative                             178     1.0000     0.0000     0.0000
relationship         Own-child                                1,019     1.0000     0.0000     0.0000
relationship         Unmarried                                  702     1.0000     0.0000     0.0000
relationship         Wife                                       322     1.0000     0.0420     0.0805
race                 Amer-Indian-Eskimo                          71     1.0000     0.1000     0.1818
race                 Asian-Pac-Islander                         193     1.0000     0.1452     0.2535
race                 Black                                      599     1.0000     0.0769     0.1429
race                 Other                                       55     1.0000     0.3333     0.5000
race                 White                                    5,595     0.9950     0.1401     0.2455
sex                  Female                                   2,126     1.0000     0.0258     0.0502
sex                  Male                                     4,387     0.9953     0.1577     0.2723
native-country       ?                                          125     1.0000     0.3548     0.5238
native-country       Cambodia                                     3        N/A        N/A        N/A
native-country       Canada                                      22     1.0000     0.2500     0.4000
native-country       China                                       18     1.0000     0.1250     0.2222
native-country       Columbia                                     6        N/A        N/A        N/A
native-country       Cuba                                        19     1.0000     0.0000     0.0000
native-country       Dominican-Republic                           8        N/A        N/A        N/A
native-country       Ecuador                                      5        N/A        N/A        N/A
native-country       El-Salvador                                 20     1.0000     0.0000     0.0000
native-country       England                                     14     1.0000     0.0000     0.0000
native-country       France                                       5        N/A        N/A        N/A
native-country       Germany                                     32     1.0000     0.2308     0.3750
native-country       Greece                                       7        N/A        N/A        N/A
native-country       Guatemala                                   13     1.0000     1.0000     1.0000
native-country       Haiti                                        6        N/A        N/A        N/A
native-country       Honduras                                     4        N/A        N/A        N/A
native-country       Hong                                         8        N/A        N/A        N/A
native-country       Hungary                                      3        N/A        N/A        N/A
native-country       India                                       21     1.0000     0.0000     0.0000
native-country       Iran                                        12     1.0000     0.0000     0.0000
native-country       Ireland                                      5        N/A        N/A        N/A
native-country       Italy                                       14     1.0000     0.2500     0.4000
native-country       Jamaica                                     13     1.0000     1.0000     1.0000
native-country       Japan                                       11     1.0000     0.0000     0.0000
native-country       Laos                                         4        N/A        N/A        N/A
native-country       Mexico                                     114     1.0000     0.3333     0.5000
native-country       Nicaragua                                    7        N/A        N/A        N/A
native-country       Peru                                         5        N/A        N/A        N/A
native-country       Philippines                                 35     1.0000     0.1250     0.2222
native-country       Poland                                      14     1.0000     0.0000     0.0000
native-country       Portugal                                     6        N/A        N/A        N/A
native-country       Puerto-Rico                                 22     1.0000     0.1667     0.2857
native-country       Scotland                                     3        N/A        N/A        N/A
native-country       South                                       13     1.0000     0.0000     0.0000
native-country       Taiwan                                      11     1.0000     0.5000     0.6667
native-country       Thailand                                     5        N/A        N/A        N/A
native-country       Trinadad&Tobago                              3        N/A        N/A        N/A
native-country       United-States                            5,870     0.9948     0.1338     0.2359
native-country       Vietnam                                      5        N/A        N/A        N/A
native-country       Yugoslavia                                   2        N/A        N/A        N/A
```

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