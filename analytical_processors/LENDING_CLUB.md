
# Conclusions
Model Accuracy with Keras

    accuracy-0.71, Test AUC Score: 0.7058

Model Accuracy with XG Boost

    accuracy-0.69, Test AUC Score: 0.6253
 
## Here are th conclusion with Keras model.
A precision of 71% means that out of all the positive predictions the model made, 71% were actually correct. This indicates that the model is fairly good at identifying true positives while keeping false positives relatively low.
An AUC score of 0.70 suggests that the model has a decent ability to distinguish between classes. A score of 0.70 means that in 70% of the cases, the model can correctly differentiate between positive and negative classes. While not perfect, it's a solid starting point, and there's room for improvement.
Overall, the model is reasonably effective but could benefit from further tuning or additional data to improve both precision and the AUC score. You might also want to look at other metrics like recall and F1-score to get a more comprehensive view of the model's performance.


## Run Predictions API
```commandline
$ cd analytical_processors/fastapi
$ python3 lending_club_predictions.py
```

```commandline
curl --location --request POST 'http://127.0.0.1:5000/predict' \
--header 'Content-Type: application/json' \
--data-raw '{
    "credit.policy": [
        1,1,1,1,1,1,0
    ],
    "purpose": [
        "all_other","credit_card","debt_consolidation","educational","home_improvement","major_purchase","small_business"
    ],
    "int.rate": [
        0.12,0.11,0.13,0.14,0.10,0.09,0.15
    ],
    "installment": [
        278.00,890.00,234.43,234.23,221.55,454.99,223.90
    ],
    "log.annual.inc": [
        11.08,12.09,13.45,12.43,13.56,9.23,14.8
    ],
    "dti": [
        14.49,12.09,13.45,12.43,13.56,9.23,14.8
    ],
    "fico": [
        720,800,790,678,780,770,678
    ],
    "days.with.cr.line": [
        3260,3432,4332,6765,2334,6677,4556
    ],
    "revol.bal": [
        43623,89089,89878,56547,34544,76556,100000
    ],
    "revol.util": [
        86.7,76.7,90.7,89.7,66.7,76.7,96.7
    ],
    "inq.last.6mths": [
        0,1,0,2,4,1,2
    ],
    "delinq.2yrs": [
        1,0,0,0,0,0,1
    ],
    "pub.rec": [
        0,0,1,0,0,0,0
    ],
    "not.fully.paid": [
        0,0,0,0,0,0,1
    ]
}'

Response
[
    {
        "default_probability_percent": 48.309627533,
        "predicted_class": 0
    },
    {
        "default_probability_percent": 53.461856842,
        "predicted_class": 1
    },
    {
        "default_probability_percent": 12.5070114136,
        "predicted_class": 0
    },
    {
        "default_probability_percent": 55.815612793,
        "predicted_class": 1
    },
    {
        "default_probability_percent": 27.3426151276,
        "predicted_class": 0
    },
    {
        "default_probability_percent": 62.6190261841,
        "predicted_class": 1
    },
    {
        "default_probability_percent": 81.0511932373,
        "predicted_class": 1
    }
]
```
## EDA, Model Training & Development
```commandline
/opt/homebrew/Cellar/python@3.12/3.12.9/venv/bin/python /Users/vijaypeddimsetti/PycharmProjects/public/analytical_processors/lending_club.py
-----------------------------------------------------------------------------------------------------------
Shape : 
(9578, 14)
-----------------------------------------------------------------------------------------------------------
Data type : 
credit.policy          int64
purpose               object
int.rate             float64
installment          float64
log.annual.inc       float64
dti                  float64
fico                   int64
days.with.cr.line    float64
revol.bal              int64
revol.util           float64
inq.last.6mths         int64
delinq.2yrs            int64
pub.rec                int64
not.fully.paid         int64
dtype: object
-----------------------------------------------------------------------------------------------------------
Row labels : 
RangeIndex(start=0, stop=9578, step=1)
-----------------------------------------------------------------------------------------------------------
Column names : 
Index(['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'], dtype='object')
-----------------------------------------------------------------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9578 entries, 0 to 9577
Data columns (total 14 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   credit.policy      9578 non-null   int64  
 1   purpose            9578 non-null   object 
 2   int.rate           9578 non-null   float64
 3   installment        9578 non-null   float64
 4   log.annual.inc     9578 non-null   float64
 5   dti                9578 non-null   float64
 6   fico               9578 non-null   int64  
 7   days.with.cr.line  9578 non-null   float64
 8   revol.bal          9578 non-null   int64  
 9   revol.util         9578 non-null   float64
 10  inq.last.6mths     9578 non-null   int64  
 11  delinq.2yrs        9578 non-null   int64  
 12  pub.rec            9578 non-null   int64  
 13  not.fully.paid     9578 non-null   int64  
dtypes: float64(6), int64(7), object(1)
memory usage: 1.0+ MB
Data info : 
None
-----------------------------------------------------------------------------------------------------------
Describe data : 
       credit.policy     int.rate  installment  log.annual.inc          dti         fico  days.with.cr.line     revol.bal   revol.util  inq.last.6mths  delinq.2yrs      pub.rec  not.fully.paid
count    9578.000000  9578.000000  9578.000000     9578.000000  9578.000000  9578.000000        9578.000000  9.578000e+03  9578.000000     9578.000000  9578.000000  9578.000000     9578.000000
mean        0.804970     0.122640   319.089413       10.932117    12.606679   710.846314        4560.767197  1.691396e+04    46.799236        1.577469     0.163708     0.062122        0.160054
std         0.396245     0.026847   207.071301        0.614813     6.883970    37.970537        2496.930377  3.375619e+04    29.014417        2.200245     0.546215     0.262126        0.366676
min         0.000000     0.060000    15.670000        7.547502     0.000000   612.000000         178.958333  0.000000e+00     0.000000        0.000000     0.000000     0.000000        0.000000
25%         1.000000     0.103900   163.770000       10.558414     7.212500   682.000000        2820.000000  3.187000e+03    22.600000        0.000000     0.000000     0.000000        0.000000
50%         1.000000     0.122100   268.950000       10.928884    12.665000   707.000000        4139.958333  8.596000e+03    46.300000        1.000000     0.000000     0.000000        0.000000
75%         1.000000     0.140700   432.762500       11.291293    17.950000   737.000000        5730.000000  1.824950e+04    70.900000        2.000000     0.000000     0.000000        0.000000
max         1.000000     0.216400   940.140000       14.528354    29.960000   827.000000       17639.958330  1.207359e+06   119.000000       33.000000    13.000000     5.000000        1.000000
-----------------------------------------------------------------------------------------------------------
At least one null value method null : 
credit.policy        0
purpose              0
int.rate             0
installment          0
log.annual.inc       0
dti                  0
fico                 0
days.with.cr.line    0
revol.bal            0
revol.util           0
inq.last.6mths       0
delinq.2yrs          0
pub.rec              0
not.fully.paid       0
dtype: int64
-----------------------------------------------------------------------------------------------------------
At least one null value method na : 
credit.policy        0
purpose              0
int.rate             0
installment          0
log.annual.inc       0
dti                  0
fico                 0
days.with.cr.line    0
revol.bal            0
revol.util           0
inq.last.6mths       0
delinq.2yrs          0
pub.rec              0
not.fully.paid       0
dtype: int64
-----------------------------------------------------------------------------------------------------------
Column which has null is: 
Series([], dtype: int64)
-----------------------------------------------------------------------------------------------------------
Duplicate data: 
0       False
1       False
2       False
3       False
4       False
        ...  
9573    False
9574    False
9575    False
9576    False
9577    False
Length: 9578, dtype: bool
-----------------------------------------------------------------------------------------------------------
Duplicate rows: 
Empty DataFrame
Columns: [credit.policy, purpose, int.rate, installment, log.annual.inc, dti, fico, days.with.cr.line, revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec, not.fully.paid]
Index: []
-----------------------------------------------------------------------------------------------------------
   credit.policy             purpose  int.rate  installment  log.annual.inc    dti  fico  days.with.cr.line  revol.bal  revol.util  inq.last.6mths  delinq.2yrs  pub.rec  not.fully.paid
0              1  debt_consolidation    0.1189       829.10       11.350407  19.48   737        5639.958333      28854        52.1               0            0        0               0
1              1         credit_card    0.1071       228.22       11.082143  14.29   707        2760.000000      33623        76.7               0            0        0               0
2              1  debt_consolidation    0.1357       366.86       10.373491  11.63   682        4710.000000       3511        25.6               1            0        0               0
3              1  debt_consolidation    0.1008       162.34       11.350407   8.10   712        2699.958333      33667        73.2               1            0        0               0
4              1         credit_card    0.1426       102.92       11.299732  14.97   667        4066.000000       4740        39.5               0            1        0               0
-----------------------------------------------------------------------------------------------------------
Shape : 
(9578, 20)
-----------------------------------------------------------------------------------------------------------
Data type : 
credit.policy                   int64
int.rate                      float64
installment                   float64
log.annual.inc                float64
dti                           float64
fico                            int64
days.with.cr.line             float64
revol.bal                       int64
revol.util                    float64
inq.last.6mths                  int64
delinq.2yrs                     int64
pub.rec                         int64
not.fully.paid                  int64
purpose_all_other               int64
purpose_credit_card             int64
purpose_debt_consolidation      int64
purpose_educational             int64
purpose_home_improvement        int64
purpose_major_purchase          int64
purpose_small_business          int64
dtype: object
-----------------------------------------------------------------------------------------------------------
Row labels : 
RangeIndex(start=0, stop=9578, step=1)
-----------------------------------------------------------------------------------------------------------
Column names : 
Index(['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid', 'purpose_all_other', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_major_purchase', 'purpose_small_business'], dtype='object')
-----------------------------------------------------------------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9578 entries, 0 to 9577
Data columns (total 20 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   credit.policy               9578 non-null   int64  
 1   int.rate                    9578 non-null   float64
 2   installment                 9578 non-null   float64
 3   log.annual.inc              9578 non-null   float64
 4   dti                         9578 non-null   float64
 5   fico                        9578 non-null   int64  
 6   days.with.cr.line           9578 non-null   float64
 7   revol.bal                   9578 non-null   int64  
 8   revol.util                  9578 non-null   float64
 9   inq.last.6mths              9578 non-null   int64  
 10  delinq.2yrs                 9578 non-null   int64  
 11  pub.rec                     9578 non-null   int64  
 12  not.fully.paid              9578 non-null   int64  
 13  purpose_all_other           9578 non-null   int64  
 14  purpose_credit_card         9578 non-null   int64  
 15  purpose_debt_consolidation  9578 non-null   int64  
 16  purpose_educational         9578 non-null   int64  
 17  purpose_home_improvement    9578 non-null   int64  
 18  purpose_major_purchase      9578 non-null   int64  
 19  purpose_small_business      9578 non-null   int64  
dtypes: float64(6), int64(14)
memory usage: 1.5 MB
Data info : 
None
-----------------------------------------------------------------------------------------------------------
Describe data : 
       credit.policy     int.rate  installment  log.annual.inc          dti         fico  days.with.cr.line     revol.bal   revol.util  inq.last.6mths  delinq.2yrs      pub.rec  not.fully.paid  purpose_all_other  purpose_credit_card  purpose_debt_consolidation  purpose_educational  purpose_home_improvement  purpose_major_purchase  purpose_small_business
count    9578.000000  9578.000000  9578.000000     9578.000000  9578.000000  9578.000000        9578.000000  9.578000e+03  9578.000000     9578.000000  9578.000000  9578.000000     9578.000000        9578.000000          9578.000000                 9578.000000          9578.000000               9578.000000             9578.000000             9578.000000
mean        0.804970     0.122640   319.089413       10.932117    12.606679   710.846314        4560.767197  1.691396e+04    46.799236        1.577469     0.163708     0.062122        0.160054           0.243370             0.131760                    0.413134             0.035811                  0.065671                0.045625                0.064627
std         0.396245     0.026847   207.071301        0.614813     6.883970    37.970537        2496.930377  3.375619e+04    29.014417        2.200245     0.546215     0.262126        0.366676           0.429139             0.338248                    0.492422             0.185829                  0.247720                0.208682                0.245880
min         0.000000     0.060000    15.670000        7.547502     0.000000   612.000000         178.958333  0.000000e+00     0.000000        0.000000     0.000000     0.000000        0.000000           0.000000             0.000000                    0.000000             0.000000                  0.000000                0.000000                0.000000
25%         1.000000     0.103900   163.770000       10.558414     7.212500   682.000000        2820.000000  3.187000e+03    22.600000        0.000000     0.000000     0.000000        0.000000           0.000000             0.000000                    0.000000             0.000000                  0.000000                0.000000                0.000000
50%         1.000000     0.122100   268.950000       10.928884    12.665000   707.000000        4139.958333  8.596000e+03    46.300000        1.000000     0.000000     0.000000        0.000000           0.000000             0.000000                    0.000000             0.000000                  0.000000                0.000000                0.000000
75%         1.000000     0.140700   432.762500       11.291293    17.950000   737.000000        5730.000000  1.824950e+04    70.900000        2.000000     0.000000     0.000000        0.000000           0.000000             0.000000                    1.000000             0.000000                  0.000000                0.000000                0.000000
max         1.000000     0.216400   940.140000       14.528354    29.960000   827.000000       17639.958330  1.207359e+06   119.000000       33.000000    13.000000     5.000000        1.000000           1.000000             1.000000                    1.000000             1.000000                  1.000000                1.000000                1.000000
-----------------------------------------------------------------------------------------------------------
At least one null value method null : 
credit.policy                 0
int.rate                      0
installment                   0
log.annual.inc                0
dti                           0
fico                          0
days.with.cr.line             0
revol.bal                     0
revol.util                    0
inq.last.6mths                0
delinq.2yrs                   0
pub.rec                       0
not.fully.paid                0
purpose_all_other             0
purpose_credit_card           0
purpose_debt_consolidation    0
purpose_educational           0
purpose_home_improvement      0
purpose_major_purchase        0
purpose_small_business        0
dtype: int64
-----------------------------------------------------------------------------------------------------------
At least one null value method na : 
credit.policy                 0
int.rate                      0
installment                   0
log.annual.inc                0
dti                           0
fico                          0
days.with.cr.line             0
revol.bal                     0
revol.util                    0
inq.last.6mths                0
delinq.2yrs                   0
pub.rec                       0
not.fully.paid                0
purpose_all_other             0
purpose_credit_card           0
purpose_debt_consolidation    0
purpose_educational           0
purpose_home_improvement      0
purpose_major_purchase        0
purpose_small_business        0
dtype: int64
-----------------------------------------------------------------------------------------------------------
Column which has null is: 
Series([], dtype: int64)
-----------------------------------------------------------------------------------------------------------
Duplicate data: 
0       False
1       False
2       False
3       False
4       False
        ...  
9573    False
9574    False
9575    False
9576    False
9577    False
Length: 9578, dtype: bool
-----------------------------------------------------------------------------------------------------------
Duplicate rows: 
Empty DataFrame
Columns: [credit.policy, int.rate, installment, log.annual.inc, dti, fico, days.with.cr.line, revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec, not.fully.paid, purpose_all_other, purpose_credit_card, purpose_debt_consolidation, purpose_educational, purpose_home_improvement, purpose_major_purchase, purpose_small_business]
Index: []
-----------------------------------------------------------------------------------------------------------
   credit.policy  int.rate  installment  log.annual.inc    dti  fico  days.with.cr.line  revol.bal  revol.util  inq.last.6mths  delinq.2yrs  pub.rec  not.fully.paid  purpose_all_other  purpose_credit_card  purpose_debt_consolidation  purpose_educational  purpose_home_improvement  purpose_major_purchase  purpose_small_business
0              1    0.1189       829.10       11.350407  19.48   737        5639.958333      28854        52.1               0            0        0               0                  0                    0                           1                    0                         0                       0                       0
1              1    0.1071       228.22       11.082143  14.29   707        2760.000000      33623        76.7               0            0        0               0                  0                    1                           0                    0                         0                       0                       0
2              1    0.1357       366.86       10.373491  11.63   682        4710.000000       3511        25.6               1            0        0               0                  0                    0                           1                    0                         0                       0                       0
3              1    0.1008       162.34       11.350407   8.10   712        2699.958333      33667        73.2               1            0        0               0                  0                    0                           1                    0                         0                       0                       0
4              1    0.1426       102.92       11.299732  14.97   667        4066.000000       4740        39.5               0            1        0               0                  0                    1                           0                    0                         0                       0                       0

Class Distribution:
Class 0: 8045 samples (83.99%)
Class 1: 1533 samples (16.01%)

🚨 Data is IMBALANCED.
-----------------------------------------------------------------------------------------------------------
Shape : 
(9578, 27)
-----------------------------------------------------------------------------------------------------------
Data type : 
credit.policy                 int64
purpose                      object
int.rate                    float64
installment                 float64
log.annual.inc              float64
dti                         float64
fico                          int64
days.with.cr.line           float64
revol.bal                     int64
revol.util                  float64
inq.last.6mths                int64
delinq.2yrs                   int64
pub.rec                       int64
not.fully.paid                int64
credit.policy_zscore        float64
int.rate_zscore             float64
installment_zscore          float64
log.annual.inc_zscore       float64
dti_zscore                  float64
fico_zscore                 float64
days.with.cr.line_zscore    float64
revol.bal_zscore            float64
revol.util_zscore           float64
inq.last.6mths_zscore       float64
delinq.2yrs_zscore          float64
pub.rec_zscore              float64
not.fully.paid_zscore       float64
dtype: object
-----------------------------------------------------------------------------------------------------------
Row labels : 
RangeIndex(start=0, stop=9578, step=1)
-----------------------------------------------------------------------------------------------------------
Column names : 
Index(['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid', 'credit.policy_zscore', 'int.rate_zscore', 'installment_zscore', 'log.annual.inc_zscore', 'dti_zscore', 'fico_zscore', 'days.with.cr.line_zscore', 'revol.bal_zscore', 'revol.util_zscore', 'inq.last.6mths_zscore', 'delinq.2yrs_zscore', 'pub.rec_zscore', 'not.fully.paid_zscore'], dtype='object')
-----------------------------------------------------------------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9578 entries, 0 to 9577
Data columns (total 27 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   credit.policy             9578 non-null   int64  
 1   purpose                   9578 non-null   object 
 2   int.rate                  9578 non-null   float64
 3   installment               9578 non-null   float64
 4   log.annual.inc            9578 non-null   float64
 5   dti                       9578 non-null   float64
 6   fico                      9578 non-null   int64  
 7   days.with.cr.line         9578 non-null   float64
 8   revol.bal                 9578 non-null   int64  
 9   revol.util                9578 non-null   float64
 10  inq.last.6mths            9578 non-null   int64  
 11  delinq.2yrs               9578 non-null   int64  
 12  pub.rec                   9578 non-null   int64  
 13  not.fully.paid            9578 non-null   int64  
 14  credit.policy_zscore      9578 non-null   float64
 15  int.rate_zscore           9578 non-null   float64
 16  installment_zscore        9578 non-null   float64
 17  log.annual.inc_zscore     9578 non-null   float64
 18  dti_zscore                9578 non-null   float64
 19  fico_zscore               9578 non-null   float64
 20  days.with.cr.line_zscore  9578 non-null   float64
 21  revol.bal_zscore          9578 non-null   float64
 22  revol.util_zscore         9578 non-null   float64
 23  inq.last.6mths_zscore     9578 non-null   float64
 24  delinq.2yrs_zscore        9578 non-null   float64
 25  pub.rec_zscore            9578 non-null   float64
 26  not.fully.paid_zscore     9578 non-null   float64
dtypes: float64(19), int64(7), object(1)
memory usage: 2.0+ MB
Data info : 
None
-----------------------------------------------------------------------------------------------------------
Describe data : 
       credit.policy     int.rate  installment  log.annual.inc          dti         fico  days.with.cr.line     revol.bal   revol.util  inq.last.6mths  delinq.2yrs      pub.rec  not.fully.paid  credit.policy_zscore  int.rate_zscore  installment_zscore  log.annual.inc_zscore    dti_zscore   fico_zscore  days.with.cr.line_zscore  revol.bal_zscore  revol.util_zscore  inq.last.6mths_zscore  delinq.2yrs_zscore  pub.rec_zscore  not.fully.paid_zscore
count    9578.000000  9578.000000  9578.000000     9578.000000  9578.000000  9578.000000        9578.000000  9.578000e+03  9578.000000     9578.000000  9578.000000  9578.000000     9578.000000          9.578000e+03     9.578000e+03        9.578000e+03           9.578000e+03  9.578000e+03  9.578000e+03              9.578000e+03      9.578000e+03       9.578000e+03           9.578000e+03        9.578000e+03    9.578000e+03           9.578000e+03
mean        0.804970     0.122640   319.089413       10.932117    12.606679   710.846314        4560.767197  1.691396e+04    46.799236        1.577469     0.163708     0.062122        0.160054         -9.495664e-17    -4.747832e-17       -4.896202e-17           1.348681e-15 -7.121748e-17  2.848699e-16             -5.934790e-17     -1.186958e-17       4.154353e-17           2.373916e-17        1.186958e-17    5.638051e-17           1.780437e-17
std         0.396245     0.026847   207.071301        0.614813     6.883970    37.970537        2496.930377  3.375619e+04    29.014417        2.200245     0.546215     0.262126        0.366676          1.000052e+00     1.000052e+00        1.000052e+00           1.000052e+00  1.000052e+00  1.000052e+00              1.000052e+00      1.000052e+00       1.000052e+00           1.000052e+00        1.000052e+00    1.000052e+00           1.000052e+00
min         0.000000     0.060000    15.670000        7.547502     0.000000   612.000000         178.958333  0.000000e+00     0.000000        0.000000     0.000000     0.000000        0.000000         -2.031603e+00    -2.333347e+00       -1.465366e+00          -5.505403e+00 -1.831405e+00 -2.603373e+00             -1.754970e+00     -5.010888e-01      -1.613049e+00          -7.169889e-01       -2.997301e-01   -2.370032e-01          -4.365239e-01
25%         1.000000     0.103900   163.770000       10.558414     7.212500   682.000000        2820.000000  3.187000e+03    22.600000        0.000000     0.000000     0.000000        0.000000          4.922223e-01    -6.980686e-01       -7.501161e-01          -6.078650e-01 -7.836264e-01 -7.597422e-01             -6.971993e-01     -4.066715e-01      -8.340853e-01          -7.169889e-01       -2.997301e-01   -2.370032e-01          -4.365239e-01
50%         1.000000     0.122100   268.950000       10.928884    12.665000   707.000000        4139.958333  8.596000e+03    46.300000        1.000000     0.000000     0.000000        0.000000          4.922223e-01    -2.011729e-02       -2.421486e-01          -5.259710e-03  8.472466e-03 -1.013026e-01             -1.685393e-01     -2.464259e-01      -1.720737e-02          -2.624704e-01       -2.997301e-01   -2.370032e-01          -4.365239e-01
75%         1.000000     0.140700   432.762500       11.291293    17.950000   737.000000        5730.000000  1.824950e+04    70.900000        2.000000     0.000000     0.000000        0.000000          4.922223e-01     6.727340e-01        5.489849e-01           5.842340e-01  7.762382e-01  6.888249e-01              4.682925e-01      3.956625e-02       8.306913e-01           1.920481e-01       -2.997301e-01   -2.370032e-01          -4.365239e-01
max         1.000000     0.216400   940.140000       14.528354    29.960000   827.000000       17639.958330  1.207359e+06   119.000000       33.000000    13.000000     5.000000        1.000000          4.922223e-01     3.492564e+00        2.999368e+00           5.849627e+00  2.520962e+00  3.059207e+00              5.238382e+00      3.526782e+01       2.488574e+00           1.428212e+01        2.350167e+01    1.883877e+01           2.290825e+00
-----------------------------------------------------------------------------------------------------------
At least one null value method null : 
credit.policy               0
purpose                     0
int.rate                    0
installment                 0
log.annual.inc              0
dti                         0
fico                        0
days.with.cr.line           0
revol.bal                   0
revol.util                  0
inq.last.6mths              0
delinq.2yrs                 0
pub.rec                     0
not.fully.paid              0
credit.policy_zscore        0
int.rate_zscore             0
installment_zscore          0
log.annual.inc_zscore       0
dti_zscore                  0
fico_zscore                 0
days.with.cr.line_zscore    0
revol.bal_zscore            0
revol.util_zscore           0
inq.last.6mths_zscore       0
delinq.2yrs_zscore          0
pub.rec_zscore              0
not.fully.paid_zscore       0
dtype: int64
-----------------------------------------------------------------------------------------------------------
At least one null value method na : 
credit.policy               0
purpose                     0
int.rate                    0
installment                 0
log.annual.inc              0
dti                         0
fico                        0
days.with.cr.line           0
revol.bal                   0
revol.util                  0
inq.last.6mths              0
delinq.2yrs                 0
pub.rec                     0
not.fully.paid              0
credit.policy_zscore        0
int.rate_zscore             0
installment_zscore          0
log.annual.inc_zscore       0
dti_zscore                  0
fico_zscore                 0
days.with.cr.line_zscore    0
revol.bal_zscore            0
revol.util_zscore           0
inq.last.6mths_zscore       0
delinq.2yrs_zscore          0
pub.rec_zscore              0
not.fully.paid_zscore       0
dtype: int64
-----------------------------------------------------------------------------------------------------------
Column which has null is: 
Series([], dtype: int64)
-----------------------------------------------------------------------------------------------------------
Duplicate data: 
0       False
1       False
2       False
3       False
4       False
        ...  
9573    False
9574    False
9575    False
9576    False
9577    False
Length: 9578, dtype: bool
-----------------------------------------------------------------------------------------------------------
Duplicate rows: 
Empty DataFrame
Columns: [credit.policy, purpose, int.rate, installment, log.annual.inc, dti, fico, days.with.cr.line, revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec, not.fully.paid, credit.policy_zscore, int.rate_zscore, installment_zscore, log.annual.inc_zscore, dti_zscore, fico_zscore, days.with.cr.line_zscore, revol.bal_zscore, revol.util_zscore, inq.last.6mths_zscore, delinq.2yrs_zscore, pub.rec_zscore, not.fully.paid_zscore]
Index: []
-----------------------------------------------------------------------------------------------------------
   credit.policy             purpose  int.rate  installment  log.annual.inc    dti  fico  days.with.cr.line  revol.bal  revol.util  inq.last.6mths  delinq.2yrs  pub.rec  not.fully.paid  credit.policy_zscore  int.rate_zscore  installment_zscore  log.annual.inc_zscore  dti_zscore  fico_zscore  days.with.cr.line_zscore  revol.bal_zscore  revol.util_zscore  inq.last.6mths_zscore  delinq.2yrs_zscore  pub.rec_zscore  not.fully.paid_zscore
0              1  debt_consolidation    0.1189       829.10       11.350407  19.48   737        5639.958333      28854        52.1               0            0        0               0              0.492222        -0.139318            2.463099               0.680388    0.998505     0.688825                  0.432230          0.353732           0.182704              -0.716989           -0.299730       -0.237003              -0.436524
1              1         credit_card    0.1071       228.22       11.082143  14.29   707        2760.000000      33623        76.7               0            0        0               0              0.492222        -0.578868           -0.438854               0.244031    0.244540    -0.101303                 -0.721230          0.495018           1.030602              -0.716989           -0.299730       -0.237003              -0.436524
2              1  debt_consolidation    0.1357       366.86       10.373491  11.63   682        4710.000000       3511        25.6               1            0        0               0              0.492222         0.486484            0.230708              -0.908659   -0.141885    -0.759742                  0.059770         -0.397073          -0.730683              -0.262470           -0.299730       -0.237003              -0.436524
3              1  debt_consolidation    0.1008       162.34       11.350407   8.10   712        2699.958333      33667        73.2               1            0        0               0              0.492222        -0.813544           -0.757022               0.680388   -0.654697     0.030385                 -0.745277          0.496321           0.909966              -0.262470           -0.299730       -0.237003              -0.436524
4              1         credit_card    0.1426       102.92       11.299732  14.97   667        4066.000000       4740        39.5               0            1        0               0              0.492222         0.743509           -1.043992               0.597961    0.343326    -1.154806                 -0.198161         -0.360663          -0.251586              -0.716989            1.531147       -0.237003              -0.436524
[[ 0.49222226 -0.13931753  2.46309947 ... -0.2651173  -0.21864717
  -0.26285458]
 [ 0.49222226 -0.57886837 -0.43885443 ... -0.2651173  -0.21864717
  -0.26285458]
 [ 0.49222226  0.48648368  0.23070836 ... -0.2651173  -0.21864717
  -0.26285458]
 ...
 [-2.03160257 -0.57886837 -1.06867038 ... -0.2651173  -0.21864717
  -0.26285458]
 [-2.03160257  1.39166043  0.1569135  ...  3.77191529 -0.21864717
  -0.26285458]
 [-2.03160257  0.61685894  2.58060136 ... -0.2651173  -0.21864717
  -0.26285458]]
Class Weights: {0: 0.5954305253341623, 1: 3.1197068403908794}
Columns to drop (correlation > 0.85): []
/opt/homebrew/Cellar/python@3.12/3.12.9/venv/lib/python3.12/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 1s 817us/step - accuracy: 0.5768 - loss: 0.6887 - val_accuracy: 0.5256 - val_loss: 0.6747
Epoch 2/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 542us/step - accuracy: 0.5654 - loss: 0.6487 - val_accuracy: 0.6409 - val_loss: 0.6301
Epoch 3/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 503us/step - accuracy: 0.5696 - loss: 0.6413 - val_accuracy: 0.5308 - val_loss: 0.7242
Epoch 4/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 510us/step - accuracy: 0.5405 - loss: 0.6587 - val_accuracy: 0.7041 - val_loss: 0.6329
Epoch 5/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 506us/step - accuracy: 0.6673 - loss: 0.6524 - val_accuracy: 0.4823 - val_loss: 0.6767
Epoch 6/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 518us/step - accuracy: 0.6489 - loss: 0.6505 - val_accuracy: 0.7432 - val_loss: 0.5687
Epoch 7/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 503us/step - accuracy: 0.6659 - loss: 0.6460 - val_accuracy: 0.5819 - val_loss: 0.6639
Epoch 8/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 493us/step - accuracy: 0.6212 - loss: 0.6433 - val_accuracy: 0.6754 - val_loss: 0.6604
Epoch 9/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 506us/step - accuracy: 0.5671 - loss: 0.6679 - val_accuracy: 0.7630 - val_loss: 0.5989
Epoch 10/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 520us/step - accuracy: 0.6958 - loss: 0.6706 - val_accuracy: 0.7046 - val_loss: 0.6376
Epoch 11/25
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 503us/step - accuracy: 0.6763 - loss: 0.6482 - val_accuracy: 0.7255 - val_loss: 0.6349
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 420us/step
Test AUC Score: 0.7065

Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.79      0.84      1611
           1       0.31      0.50      0.38       305

    accuracy                           0.74      1916
   macro avg       0.60      0.64      0.61      1916
weighted avg       0.80      0.74      0.77      1916

Test AUC Score: 0.7065
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Model saved successfully to loan_default_keras_model_0_7073.h5!
please wait for 15 min to complete the model optimization
Epoch 1/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 555us/step - accuracy: 0.6045 - loss: 0.6468 - val_accuracy: 0.5177 - val_loss: 0.6770
Epoch 2/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.6472 - loss: 0.6494 - val_accuracy: 0.6117 - val_loss: 0.6640
Epoch 3/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 501us/step - accuracy: 0.6468 - loss: 0.6567 - val_accuracy: 0.6086 - val_loss: 0.7070
Epoch 4/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 512us/step - accuracy: 0.6709 - loss: 0.6430 - val_accuracy: 0.6524 - val_loss: 0.6482
Epoch 5/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 504us/step - accuracy: 0.6674 - loss: 0.6339 - val_accuracy: 0.7109 - val_loss: 0.6090
Epoch 6/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 506us/step - accuracy: 0.7069 - loss: 0.6402 - val_accuracy: 0.7187 - val_loss: 0.6041
Epoch 7/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 506us/step - accuracy: 0.6763 - loss: 0.6532 - val_accuracy: 0.7004 - val_loss: 0.6206
Epoch 8/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 491us/step - accuracy: 0.6002 - loss: 0.6322 - val_accuracy: 0.4541 - val_loss: 0.6560
Epoch 9/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 490us/step - accuracy: 0.6095 - loss: 0.6461 - val_accuracy: 0.5094 - val_loss: 0.6666
Epoch 10/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 494us/step - accuracy: 0.6044 - loss: 0.6373 - val_accuracy: 0.5449 - val_loss: 0.6881
Epoch 11/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 491us/step - accuracy: 0.5610 - loss: 0.6280 - val_accuracy: 0.4723 - val_loss: 0.6625
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 254us/step
Test AUC Score: 0.7066
Training stopped at epoch: 11
Best epoch based on val_loss: 5

Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.75      0.82      1611
           1       0.30      0.56      0.39       305

    accuracy                           0.72      1916
   macro avg       0.60      0.66      0.60      1916
weighted avg       0.80      0.72      0.75      1916

Test AUC Score: 0.7066
Model saved successfully to loan_default_keras_model_0_7073.h5!
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Epoch 1/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 545us/step - accuracy: 0.6420 - loss: 0.6553 - val_accuracy: 0.6686 - val_loss: 0.6333
Epoch 2/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 503us/step - accuracy: 0.6841 - loss: 0.6279 - val_accuracy: 0.6962 - val_loss: 0.6385
Epoch 3/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 511us/step - accuracy: 0.7017 - loss: 0.6275 - val_accuracy: 0.6571 - val_loss: 0.6476
Epoch 4/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - accuracy: 0.6631 - loss: 0.6419 - val_accuracy: 0.6905 - val_loss: 0.6290
Epoch 5/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 513us/step - accuracy: 0.6686 - loss: 0.6530 - val_accuracy: 0.7067 - val_loss: 0.6911
Epoch 6/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 498us/step - accuracy: 0.7044 - loss: 0.6373 - val_accuracy: 0.6790 - val_loss: 0.6348
Epoch 7/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.6467 - loss: 0.6418 - val_accuracy: 0.7427 - val_loss: 0.6157
Epoch 8/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 487us/step - accuracy: 0.7104 - loss: 0.6413 - val_accuracy: 0.7396 - val_loss: 0.6203
Epoch 9/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 507us/step - accuracy: 0.7025 - loss: 0.6445 - val_accuracy: 0.7056 - val_loss: 0.6649
Epoch 10/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 497us/step - accuracy: 0.6874 - loss: 0.6434 - val_accuracy: 0.5277 - val_loss: 0.6591
Epoch 11/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 497us/step - accuracy: 0.5918 - loss: 0.6195 - val_accuracy: 0.5297 - val_loss: 0.6280
Epoch 12/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.4991 - loss: 0.6484 - val_accuracy: 0.7563 - val_loss: 0.6175
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 279us/step
Test AUC Score: 0.6956
Epoch 1/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 547us/step - accuracy: 0.7189 - loss: 0.6337 - val_accuracy: 0.7114 - val_loss: 0.6715
Epoch 2/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 498us/step - accuracy: 0.7106 - loss: 0.6467 - val_accuracy: 0.7552 - val_loss: 0.5850
Epoch 3/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.7323 - loss: 0.6230 - val_accuracy: 0.7291 - val_loss: 0.6164
Epoch 4/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 497us/step - accuracy: 0.7189 - loss: 0.6274 - val_accuracy: 0.7009 - val_loss: 0.6545
Epoch 5/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 506us/step - accuracy: 0.6862 - loss: 0.6413 - val_accuracy: 0.7020 - val_loss: 0.6068
Epoch 6/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 507us/step - accuracy: 0.6945 - loss: 0.6355 - val_accuracy: 0.6905 - val_loss: 0.6417
Epoch 7/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 508us/step - accuracy: 0.7035 - loss: 0.6200 - val_accuracy: 0.4901 - val_loss: 0.6616
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 254us/step
Test AUC Score: 0.6960
Epoch 1/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 560us/step - accuracy: 0.7197 - loss: 0.6333 - val_accuracy: 0.6743 - val_loss: 0.6412
Epoch 2/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 505us/step - accuracy: 0.6974 - loss: 0.6278 - val_accuracy: 0.6816 - val_loss: 0.6341
Epoch 3/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 497us/step - accuracy: 0.6808 - loss: 0.6385 - val_accuracy: 0.6806 - val_loss: 0.6361
Epoch 4/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 511us/step - accuracy: 0.6707 - loss: 0.6383 - val_accuracy: 0.7432 - val_loss: 0.6142
Epoch 5/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 501us/step - accuracy: 0.7330 - loss: 0.6317 - val_accuracy: 0.7338 - val_loss: 0.6229
Epoch 6/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 508us/step - accuracy: 0.6513 - loss: 0.6331 - val_accuracy: 0.7771 - val_loss: 0.5772
Epoch 7/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 501us/step - accuracy: 0.7503 - loss: 0.6205 - val_accuracy: 0.6848 - val_loss: 0.6618
Epoch 8/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 498us/step - accuracy: 0.7269 - loss: 0.6164 - val_accuracy: 0.4760 - val_loss: 0.6702
Epoch 9/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 496us/step - accuracy: 0.6219 - loss: 0.6280 - val_accuracy: 0.7270 - val_loss: 0.6356
Epoch 10/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.5816 - loss: 0.6323 - val_accuracy: 0.7124 - val_loss: 0.6554
Epoch 11/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 488us/step - accuracy: 0.7225 - loss: 0.6136 - val_accuracy: 0.7443 - val_loss: 0.6597
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 254us/step
Test AUC Score: 0.6866
Epoch 1/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 551us/step - accuracy: 0.7482 - loss: 0.6171 - val_accuracy: 0.7009 - val_loss: 0.6911
Epoch 2/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.7165 - loss: 0.6308 - val_accuracy: 0.7056 - val_loss: 0.6222
Epoch 3/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 502us/step - accuracy: 0.6631 - loss: 0.6338 - val_accuracy: 0.6989 - val_loss: 0.6669
Epoch 4/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.6966 - loss: 0.6418 - val_accuracy: 0.7229 - val_loss: 0.6364
Epoch 5/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.7206 - loss: 0.6240 - val_accuracy: 0.7317 - val_loss: 0.6295
Epoch 6/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 497us/step - accuracy: 0.7133 - loss: 0.6385 - val_accuracy: 0.7385 - val_loss: 0.6583
Epoch 7/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 492us/step - accuracy: 0.7218 - loss: 0.6339 - val_accuracy: 0.7166 - val_loss: 0.6512
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 251us/step
Test AUC Score: 0.6880
Epoch 1/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 547us/step - accuracy: 0.6783 - loss: 0.6348 - val_accuracy: 0.7291 - val_loss: 0.6447
Epoch 2/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 508us/step - accuracy: 0.7247 - loss: 0.6416 - val_accuracy: 0.7025 - val_loss: 0.5978
Epoch 3/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 509us/step - accuracy: 0.7100 - loss: 0.6383 - val_accuracy: 0.7547 - val_loss: 0.5746
Epoch 4/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 497us/step - accuracy: 0.7016 - loss: 0.6291 - val_accuracy: 0.4896 - val_loss: 0.6568
Epoch 5/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 501us/step - accuracy: 0.6570 - loss: 0.6224 - val_accuracy: 0.4666 - val_loss: 0.7031
Epoch 6/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 492us/step - accuracy: 0.6558 - loss: 0.6240 - val_accuracy: 0.7521 - val_loss: 0.6123
Epoch 7/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 491us/step - accuracy: 0.6386 - loss: 0.6299 - val_accuracy: 0.5021 - val_loss: 0.6871
Epoch 8/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 493us/step - accuracy: 0.6529 - loss: 0.6276 - val_accuracy: 0.7803 - val_loss: 0.5521
Epoch 9/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - accuracy: 0.7682 - loss: 0.6124 - val_accuracy: 0.6858 - val_loss: 0.6504
Epoch 10/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 505us/step - accuracy: 0.5871 - loss: 0.6228 - val_accuracy: 0.7255 - val_loss: 0.6390
Epoch 11/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.7328 - loss: 0.6261 - val_accuracy: 0.7062 - val_loss: 0.6607
Epoch 12/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - accuracy: 0.7243 - loss: 0.6163 - val_accuracy: 0.4577 - val_loss: 0.6831
Epoch 13/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - accuracy: 0.6555 - loss: 0.6109 - val_accuracy: 0.4807 - val_loss: 0.6635
Epoch 14/20
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 498us/step - accuracy: 0.6160 - loss: 0.6221 - val_accuracy: 0.7234 - val_loss: 0.6429
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 255us/step
Test AUC Score: 0.6769
Epoch 1/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 535us/step - accuracy: 0.7396 - loss: 0.6256 - val_accuracy: 0.7135 - val_loss: 0.6453
Epoch 2/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 494us/step - accuracy: 0.7092 - loss: 0.6246 - val_accuracy: 0.5115 - val_loss: 0.6407
Epoch 3/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 490us/step - accuracy: 0.6295 - loss: 0.6114 - val_accuracy: 0.7182 - val_loss: 0.6330
Epoch 4/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.6872 - loss: 0.6196 - val_accuracy: 0.7135 - val_loss: 0.6481
Epoch 5/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 503us/step - accuracy: 0.6394 - loss: 0.6199 - val_accuracy: 0.7082 - val_loss: 0.6411
Epoch 6/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.6972 - loss: 0.6089 - val_accuracy: 0.7234 - val_loss: 0.6529
Epoch 7/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 498us/step - accuracy: 0.7456 - loss: 0.6166 - val_accuracy: 0.7129 - val_loss: 0.6553
Epoch 8/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - accuracy: 0.6632 - loss: 0.6284 - val_accuracy: 0.6754 - val_loss: 0.6751
Epoch 9/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 502us/step - accuracy: 0.6970 - loss: 0.6236 - val_accuracy: 0.7088 - val_loss: 0.6216
Epoch 10/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 489us/step - accuracy: 0.7184 - loss: 0.6158 - val_accuracy: 0.6915 - val_loss: 0.6891
Epoch 11/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.7027 - loss: 0.6406 - val_accuracy: 0.7521 - val_loss: 0.5903
Epoch 12/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 496us/step - accuracy: 0.6914 - loss: 0.6224 - val_accuracy: 0.5016 - val_loss: 0.6264
Epoch 13/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.6143 - loss: 0.6165 - val_accuracy: 0.7432 - val_loss: 0.5902
Epoch 14/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 494us/step - accuracy: 0.7035 - loss: 0.6270 - val_accuracy: 0.7093 - val_loss: 0.6535
Epoch 15/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 489us/step - accuracy: 0.7308 - loss: 0.6155 - val_accuracy: 0.7213 - val_loss: 0.6507
Epoch 16/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 487us/step - accuracy: 0.6961 - loss: 0.6207 - val_accuracy: 0.7333 - val_loss: 0.6535
Epoch 17/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 482us/step - accuracy: 0.7464 - loss: 0.6162 - val_accuracy: 0.7323 - val_loss: 0.6280
Epoch 18/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 491us/step - accuracy: 0.7315 - loss: 0.6312 - val_accuracy: 0.6947 - val_loss: 0.6669
Epoch 19/21
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 510us/step - accuracy: 0.5986 - loss: 0.6258 - val_accuracy: 0.5094 - val_loss: 0.6309
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 282us/step
Test AUC Score: 0.6764
Epoch 1/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 554us/step - accuracy: 0.5460 - loss: 0.6223 - val_accuracy: 0.7302 - val_loss: 0.6488
Epoch 2/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 506us/step - accuracy: 0.6819 - loss: 0.6151 - val_accuracy: 0.7088 - val_loss: 0.6648
Epoch 3/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 509us/step - accuracy: 0.5916 - loss: 0.6345 - val_accuracy: 0.7542 - val_loss: 0.6118
Epoch 4/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 494us/step - accuracy: 0.5602 - loss: 0.6393 - val_accuracy: 0.7411 - val_loss: 0.6278
Epoch 5/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 503us/step - accuracy: 0.6711 - loss: 0.5948 - val_accuracy: 0.7343 - val_loss: 0.6415
Epoch 6/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 488us/step - accuracy: 0.5687 - loss: 0.6202 - val_accuracy: 0.7088 - val_loss: 0.6665
Epoch 7/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.7135 - loss: 0.6191 - val_accuracy: 0.7234 - val_loss: 0.6670
Epoch 8/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.5831 - loss: 0.6198 - val_accuracy: 0.7568 - val_loss: 0.6511
Epoch 9/22
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 512us/step - accuracy: 0.5843 - loss: 0.6159 - val_accuracy: 0.7625 - val_loss: 0.6310
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 261us/step
Test AUC Score: 0.6755
Epoch 1/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 549us/step - accuracy: 0.7100 - loss: 0.6274 - val_accuracy: 0.7484 - val_loss: 0.6102
Epoch 2/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 516us/step - accuracy: 0.6869 - loss: 0.6152 - val_accuracy: 0.4290 - val_loss: 0.6664
Epoch 3/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 491us/step - accuracy: 0.5651 - loss: 0.6099 - val_accuracy: 0.7484 - val_loss: 0.6179
Epoch 4/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 490us/step - accuracy: 0.7531 - loss: 0.6178 - val_accuracy: 0.7495 - val_loss: 0.6265
Epoch 5/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 496us/step - accuracy: 0.6418 - loss: 0.6449 - val_accuracy: 0.7479 - val_loss: 0.5964
Epoch 6/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 510us/step - accuracy: 0.6442 - loss: 0.6167 - val_accuracy: 0.7547 - val_loss: 0.6147
Epoch 7/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 504us/step - accuracy: 0.7118 - loss: 0.6138 - val_accuracy: 0.7610 - val_loss: 0.5859
Epoch 8/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 494us/step - accuracy: 0.6961 - loss: 0.6162 - val_accuracy: 0.7401 - val_loss: 0.6473
Epoch 9/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.7466 - loss: 0.6092 - val_accuracy: 0.4598 - val_loss: 0.6735
Epoch 10/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.6136 - loss: 0.6039 - val_accuracy: 0.7323 - val_loss: 0.6441
Epoch 11/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 508us/step - accuracy: 0.7278 - loss: 0.6130 - val_accuracy: 0.7343 - val_loss: 0.6514
Epoch 12/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.6289 - loss: 0.6358 - val_accuracy: 0.7683 - val_loss: 0.5826
Epoch 13/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 502us/step - accuracy: 0.7546 - loss: 0.6149 - val_accuracy: 0.7103 - val_loss: 0.6944
Epoch 14/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 517us/step - accuracy: 0.7180 - loss: 0.5979 - val_accuracy: 0.4770 - val_loss: 0.6418
Epoch 15/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 505us/step - accuracy: 0.6336 - loss: 0.6139 - val_accuracy: 0.7333 - val_loss: 0.6402
Epoch 16/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - accuracy: 0.6906 - loss: 0.6110 - val_accuracy: 0.4843 - val_loss: 0.6322
Epoch 17/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 502us/step - accuracy: 0.5470 - loss: 0.6115 - val_accuracy: 0.7886 - val_loss: 0.5927
Epoch 18/23
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 501us/step - accuracy: 0.5728 - loss: 0.6238 - val_accuracy: 0.7651 - val_loss: 0.6284
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 247us/step
Test AUC Score: 0.6675
Epoch 1/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 540us/step - accuracy: 0.7059 - loss: 0.6255 - val_accuracy: 0.7255 - val_loss: 0.6263
Epoch 2/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 493us/step - accuracy: 0.5194 - loss: 0.6135 - val_accuracy: 0.4812 - val_loss: 0.6450
Epoch 3/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 492us/step - accuracy: 0.5473 - loss: 0.6124 - val_accuracy: 0.7646 - val_loss: 0.6296
Epoch 4/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 505us/step - accuracy: 0.7361 - loss: 0.6108 - val_accuracy: 0.4734 - val_loss: 0.6510
Epoch 5/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 496us/step - accuracy: 0.5463 - loss: 0.6211 - val_accuracy: 0.7516 - val_loss: 0.6270
Epoch 6/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.7317 - loss: 0.6031 - val_accuracy: 0.7088 - val_loss: 0.6239
Epoch 7/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 504us/step - accuracy: 0.7308 - loss: 0.5983 - val_accuracy: 0.4562 - val_loss: 0.6647
Epoch 8/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 494us/step - accuracy: 0.6713 - loss: 0.6037 - val_accuracy: 0.4692 - val_loss: 0.6354
Epoch 9/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 488us/step - accuracy: 0.6939 - loss: 0.5939 - val_accuracy: 0.4682 - val_loss: 0.6198
Epoch 10/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - accuracy: 0.6266 - loss: 0.5956 - val_accuracy: 0.7302 - val_loss: 0.6628
Epoch 11/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 507us/step - accuracy: 0.7245 - loss: 0.6195 - val_accuracy: 0.4520 - val_loss: 0.6279
Epoch 12/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 506us/step - accuracy: 0.6129 - loss: 0.6049 - val_accuracy: 0.4311 - val_loss: 0.6620
Epoch 13/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 503us/step - accuracy: 0.6704 - loss: 0.6193 - val_accuracy: 0.7657 - val_loss: 0.6048
Epoch 14/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 499us/step - accuracy: 0.7199 - loss: 0.6209 - val_accuracy: 0.7223 - val_loss: 0.6388
Epoch 15/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 505us/step - accuracy: 0.7519 - loss: 0.6069 - val_accuracy: 0.7500 - val_loss: 0.6154
Epoch 16/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 504us/step - accuracy: 0.5233 - loss: 0.5997 - val_accuracy: 0.4436 - val_loss: 0.6916
Epoch 17/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 501us/step - accuracy: 0.7591 - loss: 0.6034 - val_accuracy: 0.4034 - val_loss: 0.6990
Epoch 18/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 504us/step - accuracy: 0.7012 - loss: 0.6015 - val_accuracy: 0.7584 - val_loss: 0.6312
Epoch 19/24
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 508us/step - accuracy: 0.7708 - loss: 0.6003 - val_accuracy: 0.4567 - val_loss: 0.6253
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 255us/step
Test AUC Score: 0.6569

peakAucScore:
0.7065624650201993

maxPatience:
6
-----------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
Model and Scalar loaded successfully!
-----------------------------------------------------------------------------------------------------------
Shape : 
(14, 14)
-----------------------------------------------------------------------------------------------------------
Data type : 
credit.policy          int64
purpose               object
int.rate             float64
installment          float64
log.annual.inc       float64
dti                  float64
fico                   int64
days.with.cr.line    float64
revol.bal              int64
revol.util           float64
inq.last.6mths         int64
delinq.2yrs            int64
pub.rec                int64
not.fully.paid         int64
dtype: object
-----------------------------------------------------------------------------------------------------------
Row labels : 
RangeIndex(start=0, stop=14, step=1)
-----------------------------------------------------------------------------------------------------------
Column names : 
Index(['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'], dtype='object')
-----------------------------------------------------------------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14 entries, 0 to 13
Data columns (total 14 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   credit.policy      14 non-null     int64  
 1   purpose            14 non-null     object 
 2   int.rate           14 non-null     float64
 3   installment        14 non-null     float64
 4   log.annual.inc     14 non-null     float64
 5   dti                14 non-null     float64
 6   fico               14 non-null     int64  
 7   days.with.cr.line  14 non-null     float64
 8   revol.bal          14 non-null     int64  
 9   revol.util         14 non-null     float64
 10  inq.last.6mths     14 non-null     int64  
 11  delinq.2yrs        14 non-null     int64  
 12  pub.rec            14 non-null     int64  
 13  not.fully.paid     14 non-null     int64  
dtypes: float64(6), int64(7), object(1)
memory usage: 1.7+ KB
Data info : 
None
-----------------------------------------------------------------------------------------------------------
Describe data : 
       credit.policy   int.rate  installment  log.annual.inc        dti        fico  days.with.cr.line     revol.bal  revol.util  inq.last.6mths  delinq.2yrs    pub.rec  not.fully.paid
count           14.0  14.000000    14.000000       14.000000  14.000000   14.000000          14.000000     14.000000   14.000000       14.000000    14.000000  14.000000       14.000000
mean             1.0   0.117107   247.351429       11.103247  12.030714  717.642857        5041.005952  28607.071429   55.475000        1.142857     0.214286   0.142857        0.214286
std              0.0   0.027713   208.337111        1.064862   4.588092   27.048694        2071.722095  22265.719911   24.504213        1.561909     0.425815   0.363137        0.425815
min              1.0   0.064300    86.750000        8.798127   4.200000  680.000000        3199.958333   4021.000000    4.950000        0.000000     0.000000   0.000000        0.000000
25%              1.0   0.095200   134.887500       10.458723   8.775000  696.250000        3365.010417  13450.250000   36.550000        0.000000     0.000000   0.000000        0.000000
50%              1.0   0.122400   178.175000       11.190937  12.005000  720.000000        4733.020834  15185.000000   61.050000        1.000000     0.000000   0.000000        0.000000
75%              1.0   0.131300   269.670000       11.393275  15.000000  736.500000        5558.000000  42430.750000   74.975000        1.000000     0.000000   0.000000        0.000000
max              1.0   0.159600   879.100000       13.184421  19.680000  770.000000       11048.958330  79909.000000   86.800000        5.000000     1.000000   1.000000        1.000000
-----------------------------------------------------------------------------------------------------------
At least one null value method null : 
credit.policy        0
purpose              0
int.rate             0
installment          0
log.annual.inc       0
dti                  0
fico                 0
days.with.cr.line    0
revol.bal            0
revol.util           0
inq.last.6mths       0
delinq.2yrs          0
pub.rec              0
not.fully.paid       0
dtype: int64
-----------------------------------------------------------------------------------------------------------
At least one null value method na : 
credit.policy        0
purpose              0
int.rate             0
installment          0
log.annual.inc       0
dti                  0
fico                 0
days.with.cr.line    0
revol.bal            0
revol.util           0
inq.last.6mths       0
delinq.2yrs          0
pub.rec              0
not.fully.paid       0
dtype: int64
-----------------------------------------------------------------------------------------------------------
Column which has null is: 
Series([], dtype: int64)
-----------------------------------------------------------------------------------------------------------
Duplicate data: 
0     False
1     False
2     False
3     False
4     False
5     False
6     False
7     False
8     False
9     False
10    False
11    False
12    False
13    False
dtype: bool
-----------------------------------------------------------------------------------------------------------
Duplicate rows: 
Empty DataFrame
Columns: [credit.policy, purpose, int.rate, installment, log.annual.inc, dti, fico, days.with.cr.line, revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec, not.fully.paid]
Index: []
-----------------------------------------------------------------------------------------------------------
   credit.policy             purpose  int.rate  installment  log.annual.inc    dti  fico  days.with.cr.line  revol.bal  revol.util  inq.last.6mths  delinq.2yrs  pub.rec  not.fully.paid
0              1  debt_consolidation    0.1289       879.10       11.350407  19.68   750        6139.958333      38854        62.1               1            0        1               0
1              1         credit_card    0.1171       278.22       11.082143  14.49   720        3260.000000      43623        86.7               0            0        0               1
2              1  debt_consolidation    0.1457       416.86       10.373491  11.83   695        5210.000000      13511        35.6               4            1        0               0
3              1    home_improvement    0.1108       212.34       11.350407   8.30   725        3199.958333      43667        83.2               1            0        0               0
4              1         credit_card    0.1526       152.92       11.299732  15.17   680        4566.000000      14740        49.5               0            0        1               0
-----------------------------------------------------------------------------------------------------------
Shape : 
(14, 20)
-----------------------------------------------------------------------------------------------------------
Data type : 
credit.policy                   int64
int.rate                      float64
installment                   float64
log.annual.inc                float64
dti                           float64
fico                            int64
days.with.cr.line             float64
revol.bal                       int64
revol.util                    float64
inq.last.6mths                  int64
delinq.2yrs                     int64
pub.rec                         int64
not.fully.paid                  int64
purpose_all_other               int64
purpose_credit_card             int64
purpose_debt_consolidation      int64
purpose_educational             int64
purpose_home_improvement        int64
purpose_major_purchase          int64
purpose_small_business          int64
dtype: object
-----------------------------------------------------------------------------------------------------------
Row labels : 
RangeIndex(start=0, stop=14, step=1)
-----------------------------------------------------------------------------------------------------------
Column names : 
Index(['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid', 'purpose_all_other', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_major_purchase', 'purpose_small_business'], dtype='object')
-----------------------------------------------------------------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14 entries, 0 to 13
Data columns (total 20 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   credit.policy               14 non-null     int64  
 1   int.rate                    14 non-null     float64
 2   installment                 14 non-null     float64
 3   log.annual.inc              14 non-null     float64
 4   dti                         14 non-null     float64
 5   fico                        14 non-null     int64  
 6   days.with.cr.line           14 non-null     float64
 7   revol.bal                   14 non-null     int64  
 8   revol.util                  14 non-null     float64
 9   inq.last.6mths              14 non-null     int64  
 10  delinq.2yrs                 14 non-null     int64  
 11  pub.rec                     14 non-null     int64  
 12  not.fully.paid              14 non-null     int64  
 13  purpose_all_other           14 non-null     int64  
 14  purpose_credit_card         14 non-null     int64  
 15  purpose_debt_consolidation  14 non-null     int64  
 16  purpose_educational         14 non-null     int64  
 17  purpose_home_improvement    14 non-null     int64  
 18  purpose_major_purchase      14 non-null     int64  
 19  purpose_small_business      14 non-null     int64  
dtypes: float64(6), int64(14)
memory usage: 2.3 KB
Data info : 
None
-----------------------------------------------------------------------------------------------------------
Describe data : 
       credit.policy   int.rate  installment  log.annual.inc        dti        fico  days.with.cr.line     revol.bal  revol.util  inq.last.6mths  delinq.2yrs    pub.rec  not.fully.paid  purpose_all_other  purpose_credit_card  purpose_debt_consolidation  purpose_educational  purpose_home_improvement  purpose_major_purchase  purpose_small_business
count           14.0  14.000000    14.000000       14.000000  14.000000   14.000000          14.000000     14.000000   14.000000       14.000000    14.000000  14.000000       14.000000          14.000000            14.000000                   14.000000            14.000000                 14.000000               14.000000               14.000000
mean             1.0   0.117107   247.351429       11.103247  12.030714  717.642857        5041.005952  28607.071429   55.475000        1.142857     0.214286   0.142857        0.214286           0.142857             0.285714                    0.214286             0.071429                  0.142857                0.071429                0.071429
std              0.0   0.027713   208.337111        1.064862   4.588092   27.048694        2071.722095  22265.719911   24.504213        1.561909     0.425815   0.363137        0.425815           0.363137             0.468807                    0.425815             0.267261                  0.363137                0.267261                0.267261
min              1.0   0.064300    86.750000        8.798127   4.200000  680.000000        3199.958333   4021.000000    4.950000        0.000000     0.000000   0.000000        0.000000           0.000000             0.000000                    0.000000             0.000000                  0.000000                0.000000                0.000000
25%              1.0   0.095200   134.887500       10.458723   8.775000  696.250000        3365.010417  13450.250000   36.550000        0.000000     0.000000   0.000000        0.000000           0.000000             0.000000                    0.000000             0.000000                  0.000000                0.000000                0.000000
50%              1.0   0.122400   178.175000       11.190937  12.005000  720.000000        4733.020834  15185.000000   61.050000        1.000000     0.000000   0.000000        0.000000           0.000000             0.000000                    0.000000             0.000000                  0.000000                0.000000                0.000000
75%              1.0   0.131300   269.670000       11.393275  15.000000  736.500000        5558.000000  42430.750000   74.975000        1.000000     0.000000   0.000000        0.000000           0.000000             0.750000                    0.000000             0.000000                  0.000000                0.000000                0.000000
max              1.0   0.159600   879.100000       13.184421  19.680000  770.000000       11048.958330  79909.000000   86.800000        5.000000     1.000000   1.000000        1.000000           1.000000             1.000000                    1.000000             1.000000                  1.000000                1.000000                1.000000
-----------------------------------------------------------------------------------------------------------
At least one null value method null : 
credit.policy                 0
int.rate                      0
installment                   0
log.annual.inc                0
dti                           0
fico                          0
days.with.cr.line             0
revol.bal                     0
revol.util                    0
inq.last.6mths                0
delinq.2yrs                   0
pub.rec                       0
not.fully.paid                0
purpose_all_other             0
purpose_credit_card           0
purpose_debt_consolidation    0
purpose_educational           0
purpose_home_improvement      0
purpose_major_purchase        0
purpose_small_business        0
dtype: int64
-----------------------------------------------------------------------------------------------------------
At least one null value method na : 
credit.policy                 0
int.rate                      0
installment                   0
log.annual.inc                0
dti                           0
fico                          0
days.with.cr.line             0
revol.bal                     0
revol.util                    0
inq.last.6mths                0
delinq.2yrs                   0
pub.rec                       0
not.fully.paid                0
purpose_all_other             0
purpose_credit_card           0
purpose_debt_consolidation    0
purpose_educational           0
purpose_home_improvement      0
purpose_major_purchase        0
purpose_small_business        0
dtype: int64
-----------------------------------------------------------------------------------------------------------
Column which has null is: 
Series([], dtype: int64)
-----------------------------------------------------------------------------------------------------------
Duplicate data: 
0     False
1     False
2     False
3     False
4     False
5     False
6     False
7     False
8     False
9     False
10    False
11    False
12    False
13    False
dtype: bool
-----------------------------------------------------------------------------------------------------------
Duplicate rows: 
Empty DataFrame
Columns: [credit.policy, int.rate, installment, log.annual.inc, dti, fico, days.with.cr.line, revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec, not.fully.paid, purpose_all_other, purpose_credit_card, purpose_debt_consolidation, purpose_educational, purpose_home_improvement, purpose_major_purchase, purpose_small_business]
Index: []
-----------------------------------------------------------------------------------------------------------
   credit.policy  int.rate  installment  log.annual.inc    dti  fico  days.with.cr.line  revol.bal  revol.util  inq.last.6mths  delinq.2yrs  pub.rec  not.fully.paid  purpose_all_other  purpose_credit_card  purpose_debt_consolidation  purpose_educational  purpose_home_improvement  purpose_major_purchase  purpose_small_business
0              1    0.1289       879.10       11.350407  19.68   750        6139.958333      38854        62.1               1            0        1               0                  0                    0                           1                    0                         0                       0                       0
1              1    0.1171       278.22       11.082143  14.49   720        3260.000000      43623        86.7               0            0        0               1                  0                    1                           0                    0                         0                       0                       0
2              1    0.1457       416.86       10.373491  11.83   695        5210.000000      13511        35.6               4            1        0               0                  0                    0                           1                    0                         0                       0                       0
3              1    0.1108       212.34       11.350407   8.30   725        3199.958333      43667        83.2               1            0        0               0                  0                    0                           0                    0                         1                       0                       0
4              1    0.1526       152.92       11.299732  15.17   680        4566.000000      14740        49.5               0            0        1               0                  0                    1                           0                    0                         0                       0                       0
/Users/vijaypeddimsetti/PycharmProjects/public/analytical_processors/wrangling/prepare.py:37: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  z_scores = data.select_dtypes(include=['number']).apply(lambda x: stats.zscore(x))
-----------------------------------------------------------------------------------------------------------
Shape : 
(14, 27)
-----------------------------------------------------------------------------------------------------------
Data type : 
credit.policy                 int64
purpose                      object
int.rate                    float64
installment                 float64
log.annual.inc              float64
dti                         float64
fico                          int64
days.with.cr.line           float64
revol.bal                     int64
revol.util                  float64
inq.last.6mths                int64
delinq.2yrs                   int64
pub.rec                       int64
not.fully.paid                int64
credit.policy_zscore        float64
int.rate_zscore             float64
installment_zscore          float64
log.annual.inc_zscore       float64
dti_zscore                  float64
fico_zscore                 float64
days.with.cr.line_zscore    float64
revol.bal_zscore            float64
revol.util_zscore           float64
inq.last.6mths_zscore       float64
delinq.2yrs_zscore          float64
pub.rec_zscore              float64
not.fully.paid_zscore       float64
dtype: object
-----------------------------------------------------------------------------------------------------------
Row labels : 
RangeIndex(start=0, stop=14, step=1)
-----------------------------------------------------------------------------------------------------------
Column names : 
Index(['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid', 'credit.policy_zscore', 'int.rate_zscore', 'installment_zscore', 'log.annual.inc_zscore', 'dti_zscore', 'fico_zscore', 'days.with.cr.line_zscore', 'revol.bal_zscore', 'revol.util_zscore', 'inq.last.6mths_zscore', 'delinq.2yrs_zscore', 'pub.rec_zscore', 'not.fully.paid_zscore'], dtype='object')
-----------------------------------------------------------------------------------------------------------
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 14 entries, 0 to 13
Data columns (total 27 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   credit.policy             14 non-null     int64  
 1   purpose                   14 non-null     object 
 2   int.rate                  14 non-null     float64
 3   installment               14 non-null     float64
 4   log.annual.inc            14 non-null     float64
 5   dti                       14 non-null     float64
 6   fico                      14 non-null     int64  
 7   days.with.cr.line         14 non-null     float64
 8   revol.bal                 14 non-null     int64  
 9   revol.util                14 non-null     float64
 10  inq.last.6mths            14 non-null     int64  
 11  delinq.2yrs               14 non-null     int64  
 12  pub.rec                   14 non-null     int64  
 13  not.fully.paid            14 non-null     int64  
 14  credit.policy_zscore      0 non-null      float64
 15  int.rate_zscore           14 non-null     float64
 16  installment_zscore        14 non-null     float64
 17  log.annual.inc_zscore     14 non-null     float64
 18  dti_zscore                14 non-null     float64
 19  fico_zscore               14 non-null     float64
 20  days.with.cr.line_zscore  14 non-null     float64
 21  revol.bal_zscore          14 non-null     float64
 22  revol.util_zscore         14 non-null     float64
 23  inq.last.6mths_zscore     14 non-null     float64
 24  delinq.2yrs_zscore        14 non-null     float64
 25  pub.rec_zscore            14 non-null     float64
 26  not.fully.paid_zscore     14 non-null     float64
dtypes: float64(19), int64(7), object(1)
memory usage: 3.1+ KB
Data info : 
None
-----------------------------------------------------------------------------------------------------------
Describe data : 
       credit.policy   int.rate  installment  log.annual.inc        dti        fico  days.with.cr.line     revol.bal  revol.util  inq.last.6mths  delinq.2yrs    pub.rec  not.fully.paid  credit.policy_zscore  int.rate_zscore  installment_zscore  log.annual.inc_zscore    dti_zscore   fico_zscore  days.with.cr.line_zscore  revol.bal_zscore  revol.util_zscore  inq.last.6mths_zscore  delinq.2yrs_zscore  pub.rec_zscore  not.fully.paid_zscore
count           14.0  14.000000    14.000000       14.000000  14.000000   14.000000          14.000000     14.000000   14.000000       14.000000    14.000000  14.000000       14.000000                   0.0     1.400000e+01        1.400000e+01           1.400000e+01  1.400000e+01  1.400000e+01              1.400000e+01      1.400000e+01       1.400000e+01           1.400000e+01           14.000000    1.400000e+01              14.000000
mean             1.0   0.117107   247.351429       11.103247  12.030714  717.642857        5041.005952  28607.071429   55.475000        1.142857     0.214286   0.142857        0.214286                   NaN     5.392512e-16       -1.982541e-16           6.026925e-16 -1.090398e-17  1.229175e-15              5.630417e-16      5.551115e-17      -3.806479e-16           6.344132e-17            0.000000    1.586033e-17               0.000000
std              0.0   0.027713   208.337111        1.064862   4.588092   27.048694        2071.722095  22265.719911   24.504213        1.561909     0.425815   0.363137        0.425815                   NaN     1.037749e+00        1.037749e+00           1.037749e+00  1.037749e+00  1.037749e+00              1.037749e+00      1.037749e+00       1.037749e+00           1.037749e+00            1.037749    1.037749e+00               1.037749
min              1.0   0.064300    86.750000        8.798127   4.200000  680.000000        3199.958333   4021.000000    4.950000        0.000000     0.000000   0.000000        0.000000                   NaN    -1.977456e+00       -7.999726e-01          -2.246428e+00 -1.771175e+00 -1.444204e+00             -9.222016e-01     -1.145895e+00      -2.139725e+00          -7.593264e-01           -0.522233   -4.082483e-01              -0.522233
25%              1.0   0.095200   134.887500       10.458723   8.775000  696.250000        3365.010417  13450.250000   36.550000        0.000000     0.000000   0.000000        0.000000                   NaN    -8.203513e-01       -5.601946e-01          -6.281137e-01 -7.363876e-01 -8.207574e-01             -8.395251e-01     -7.064212e-01      -8.014704e-01          -7.593264e-01           -0.522233   -4.082483e-01              -0.522233
50%              1.0   0.122400   178.175000       11.190937  12.005000  720.000000        4733.020834  15185.000000   61.050000        1.000000     0.000000   0.000000        0.000000                   NaN     1.982003e-01       -3.445751e-01           8.545774e-02 -5.816137e-03  9.043404e-02             -1.542732e-01     -6.255689e-01       2.361003e-01          -9.491580e-02           -0.522233   -4.082483e-01              -0.522233
75%              1.0   0.131300   269.670000       11.393275  15.000000  736.500000        5558.000000  42430.750000   74.975000        1.000000     0.000000   0.000000        0.000000                   NaN     5.314764e-01        1.111711e-01           2.826439e-01  6.716023e-01  7.234723e-01              2.589682e-01      6.442868e-01       8.258215e-01          -9.491580e-02           -0.522233   -4.082483e-01              -0.522233
max              1.0   0.159600   879.100000       13.184421  19.680000  770.000000       11048.958330  79909.000000   86.800000        5.000000     1.000000   1.000000        1.000000                   NaN     1.591219e+00        3.146806e+00           2.028185e+00  1.730139e+00  2.008732e+00              3.009451e+00      2.391053e+00       1.326608e+00           2.562727e+00            1.914854    2.449490e+00               1.914854
-----------------------------------------------------------------------------------------------------------
At least one null value method null : 
credit.policy                0
purpose                      0
int.rate                     0
installment                  0
log.annual.inc               0
dti                          0
fico                         0
days.with.cr.line            0
revol.bal                    0
revol.util                   0
inq.last.6mths               0
delinq.2yrs                  0
pub.rec                      0
not.fully.paid               0
credit.policy_zscore        14
int.rate_zscore              0
installment_zscore           0
log.annual.inc_zscore        0
dti_zscore                   0
fico_zscore                  0
days.with.cr.line_zscore     0
revol.bal_zscore             0
revol.util_zscore            0
inq.last.6mths_zscore        0
delinq.2yrs_zscore           0
pub.rec_zscore               0
not.fully.paid_zscore        0
dtype: int64
-----------------------------------------------------------------------------------------------------------
At least one null value method na : 
credit.policy                0
purpose                      0
int.rate                     0
installment                  0
log.annual.inc               0
dti                          0
fico                         0
days.with.cr.line            0
revol.bal                    0
revol.util                   0
inq.last.6mths               0
delinq.2yrs                  0
pub.rec                      0
not.fully.paid               0
credit.policy_zscore        14
int.rate_zscore              0
installment_zscore           0
log.annual.inc_zscore        0
dti_zscore                   0
fico_zscore                  0
days.with.cr.line_zscore     0
revol.bal_zscore             0
revol.util_zscore            0
inq.last.6mths_zscore        0
delinq.2yrs_zscore           0
pub.rec_zscore               0
not.fully.paid_zscore        0
dtype: int64
-----------------------------------------------------------------------------------------------------------
Column which has null is: 
Series([], dtype: int64)
-----------------------------------------------------------------------------------------------------------
Duplicate data: 
0     False
1     False
2     False
3     False
4     False
5     False
6     False
7     False
8     False
9     False
10    False
11    False
12    False
13    False
dtype: bool
-----------------------------------------------------------------------------------------------------------
Duplicate rows: 
Empty DataFrame
Columns: [credit.policy, purpose, int.rate, installment, log.annual.inc, dti, fico, days.with.cr.line, revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec, not.fully.paid, credit.policy_zscore, int.rate_zscore, installment_zscore, log.annual.inc_zscore, dti_zscore, fico_zscore, days.with.cr.line_zscore, revol.bal_zscore, revol.util_zscore, inq.last.6mths_zscore, delinq.2yrs_zscore, pub.rec_zscore, not.fully.paid_zscore]
Index: []
-----------------------------------------------------------------------------------------------------------
   credit.policy             purpose  int.rate  installment  log.annual.inc    dti  fico  days.with.cr.line  revol.bal  revol.util  inq.last.6mths  delinq.2yrs  pub.rec  not.fully.paid  credit.policy_zscore  int.rate_zscore  installment_zscore  log.annual.inc_zscore  dti_zscore  fico_zscore  days.with.cr.line_zscore  revol.bal_zscore  revol.util_zscore  inq.last.6mths_zscore  delinq.2yrs_zscore  pub.rec_zscore  not.fully.paid_zscore
0              1  debt_consolidation    0.1289       879.10       11.350407  19.68   750        6139.958333      38854        62.1               1            0        1               0                   NaN         0.441604            3.146806               0.240867    1.730139     1.241413                  0.550478          0.477583           0.280568              -0.094916           -0.522233        2.449490              -0.522233
1              1         credit_card    0.1171       278.22       11.082143  14.49   720        3260.000000      43623        86.7               0            0        0               1                   NaN        -0.000267            0.153760              -0.020567    0.556249     0.090434                 -0.892126          0.699855           1.322373              -0.759326           -0.522233       -0.408248               1.914854
2              1  debt_consolidation    0.1457       416.86       10.373491  11.83   695        5210.000000      13511        35.6               4            1        0               0                   NaN         1.070710            0.844340              -0.711175   -0.045398    -0.868715                  0.084651         -0.703590          -0.841703               1.898316            1.914854       -0.408248              -0.522233
3              1    home_improvement    0.1108       212.34       11.350407   8.30   725        3199.958333      43667        83.2               1            0        0               0                   NaN        -0.236182           -0.174396               0.240867   -0.843825     0.282264                 -0.922202          0.701905           1.174149              -0.094916           -0.522233       -0.408248              -0.522233
4              1         credit_card    0.1526       152.92       11.299732  15.17   680        4566.000000      14740        49.5               0            0        1               0                   NaN         1.329092           -0.470373               0.191483    0.710053    -1.444204                 -0.237936         -0.646309          -0.253040              -0.759326           -0.522233        2.449490              -0.522233
-----------------------------------------------------------------------------------------------------------
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 21ms/step
    default_probability_percent  predicted_class
0                     49.441772                0
1                     30.992874                0
2                     52.501118                1
3                     48.976734                0
4                     55.034153                1
5                      4.349671                0
6                     48.976734                0
7                     48.976734                0
8                     44.634773                0
9                     41.996666                0
10                    56.652420                1
11                     0.901935                0
12                    48.976734                0
13                     1.287172                0
-----------------------------------------------------------------------------------------------------------

Classification Report data_new:
              precision    recall  f1-score   support

           0       0.82      0.82      0.82        11
           1       0.33      0.33      0.33         3

    accuracy                           0.71        14
   macro avg       0.58      0.58      0.58        14
weighted avg       0.71      0.71      0.71        14

Test AUC Score data_new: 0.5758
-----------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------
Imbalance Ratio: 5.24
/opt/homebrew/Cellar/python@3.12/3.12.9/venv/lib/python3.12/site-packages/xgboost/training.py:183: UserWarning: [13:33:20] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:738: 
Parameters: { "use_label_encoder" } are not used.

  bst.update(dtrain, iteration=i, fobj=obj)
[0]	validation_0-auc:0.63817
[100]	validation_0-auc:0.68099
[200]	validation_0-auc:0.66487
[300]	validation_0-auc:0.64568
[400]	validation_0-auc:0.63540
[499]	validation_0-auc:0.62525

Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.74      0.80      1611
           1       0.24      0.42      0.30       305

    accuracy                           0.69      1916
   macro avg       0.55      0.58      0.55      1916
weighted avg       0.77      0.69      0.72      1916

Test AUC Score: 0.6253
-----------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------

Process finished with exit code 0
```

# EDA

## Loan Data Null Analysis
![mvvisualization.visualize_ld.png](20250517095032/mvvisualization.visualize_ld.png)
![nvvisualization.visualize_ld.png](20250517095032/nvvisualization.visualize_ld.png)
![mvvisualization.visualize_lde.png](20250517095032/mvvisualization.visualize_lde.png)
![nvvisualization.visualize_lde.png](20250517095032/nvvisualization.visualize_lde.png)

## Loan Data Box Plots
![bancredit.policy_visualization.visualize_lde.png](20250517095032/bancredit.policy_visualization.visualize_lde.png)
![banpurpose_visualization.visualize_lde.png](20250517095032/banpurpose_visualization.visualize_lde.png)
![banint.rate_visualization.visualize_lde.png](20250517095032/banint.rate_visualization.visualize_lde.png)
![baninstallment_visualization.visualize_lde.png](20250517095032/baninstallment_visualization.visualize_lde.png)
![banlog.annual.inc_visualization.visualize_lde.png](20250517095032/banlog.annual.inc_visualization.visualize_lde.png)
![bandti_visualization.visualize_lde.png](20250517095032/bandti_visualization.visualize_lde.png)
![banfico_visualization.visualize_lde.png](20250517095032/banfico_visualization.visualize_lde.png)
![bandays.with.cr.line_visualization.visualize_lde.png](20250517095032/bandays.with.cr.line_visualization.visualize_lde.png)
![banrevol.bal_visualization.visualize_lde.png](20250517095032/banrevol.bal_visualization.visualize_lde.png)
![banrevol.util_visualization.visualize_lde.png](20250517095032/banrevol.util_visualization.visualize_lde.png)
![baninq.last.6mths_visualization.visualize_lde.png](20250517095032/baninq.last.6mths_visualization.visualize_lde.png)
![bandelinq.2yrs_visualization.visualize_lde.png](20250517095032/bandelinq.2yrs_visualization.visualize_lde.png)
![banpub.rec_visualization.visualize_lde.png](20250517095032/banpub.rec_visualization.visualize_lde.png)
![bannot.fully.paid_visualization.visualize_lde.png](20250517095032/bannot.fully.paid_visualization.visualize_lde.png)

## Loan Data Pair Plot
![ppvisualization.visualize_lde.png](20250517095032/ppvisualization.visualize_lde.png)

## Loan Data Z Scores
![zsbpvisualization.visualize_lde.png](20250517095032/zsbpvisualization.visualize_lde.png)

## Loan Data Keras Model Accuracy
### Run 1
![keras_accuracy___main___.png](20250517095032/keras_accuracy___main___.png)
### Run 2
![keras_accuracy___main___.png](20250517095032/keras_accuracy___main___2.png)

## Loan Data Keras Model Loss
### Run 1
![keras_loss___main___.png](20250517095032/keras_loss___main___.png)
### Run 2
![keras_loss___main___.png](20250517095032/keras_loss___main___2.png)

## Loan Data New Null Analysis
![mvvisualization.visualize_ldn.png](20250517095032/mvvisualization.visualize_ldn.png)
![nvvisualization.visualize_ldn.png](20250517095032/nvvisualization.visualize_ldn.png)
![mvvisualization.visualize_ldne.png](20250517095032/mvvisualization.visualize_ldne.png)
![nvvisualization.visualize_ldne.png](20250517095032/nvvisualization.visualize_ldne.png)

## Loan Data New Box Plots
![uancredit.policy_visualization.visualize_.png](20250517095032/uancredit.policy_visualization.visualize_.png)
![uanpurpose_visualization.visualize_.png](20250517095032/uanpurpose_visualization.visualize_.png)
![uanint.rate_visualization.visualize_.png](20250517095032/uanint.rate_visualization.visualize_.png)
![uaninstallment_visualization.visualize_.png](20250517095032/uaninstallment_visualization.visualize_.png)
![uanlog.annual.inc_visualization.visualize_.png](20250517095032/uanlog.annual.inc_visualization.visualize_.png)
![uandti_visualization.visualize_.png](20250517095032/uandti_visualization.visualize_.png)
![uanfico_visualization.visualize_.png](20250517095032/uanfico_visualization.visualize_.png)
![uandays.with.cr.line_visualization.visualize_.png](20250517095032/uandays.with.cr.line_visualization.visualize_.png)
![uanrevol.bal_visualization.visualize_.png](20250517095032/uanrevol.bal_visualization.visualize_.png)
![uanrevol.util_visualization.visualize_.png](20250517095032/uanrevol.util_visualization.visualize_.png)
![uaninq.last.6mths_visualization.visualize_.png](20250517095032/uaninq.last.6mths_visualization.visualize_.png)
![uandelinq.2yrs_visualization.visualize_.png](20250517095032/uandelinq.2yrs_visualization.visualize_.png)
![uanpub.rec_visualization.visualize_.png](20250517095032/uanpub.rec_visualization.visualize_.png)
![uannot.fully.paid_visualization.visualize_.png](20250517095032/uannot.fully.paid_visualization.visualize_.png)
![bancredit.policy_visualization.visualize_ldne.png](20250517095032/bancredit.policy_visualization.visualize_ldne.png)
![banpurpose_visualization.visualize_ldne.png](20250517095032/banpurpose_visualization.visualize_ldne.png)
![banint.rate_visualization.visualize_ldne.png](20250517095032/banint.rate_visualization.visualize_ldne.png)
![baninstallment_visualization.visualize_ldne.png](20250517095032/baninstallment_visualization.visualize_ldne.png)
![banlog.annual.inc_visualization.visualize_ldne.png](20250517095032/banlog.annual.inc_visualization.visualize_ldne.png)
![bandti_visualization.visualize_ldne.png](20250517095032/bandti_visualization.visualize_ldne.png)
![banfico_visualization.visualize_ldne.png](20250517095032/banfico_visualization.visualize_ldne.png)
![bandays.with.cr.line_visualization.visualize_ldne.png](20250517095032/bandays.with.cr.line_visualization.visualize_ldne.png)
![banrevol.bal_visualization.visualize_ldne.png](20250517095032/banrevol.bal_visualization.visualize_ldne.png)
![banrevol.util_visualization.visualize_ldne.png](20250517095032/banrevol.util_visualization.visualize_ldne.png)
![baninq.last.6mths_visualization.visualize_ldne.png](20250517095032/baninq.last.6mths_visualization.visualize_ldne.png)
![bandelinq.2yrs_visualization.visualize_ldne.png](20250517095032/bandelinq.2yrs_visualization.visualize_ldne.png)
![banpub.rec_visualization.visualize_ldne.png](20250517095032/banpub.rec_visualization.visualize_ldne.png)
![bannot.fully.paid_visualization.visualize_ldne.png](20250517095032/bannot.fully.paid_visualization.visualize_ldne.png)

## Loan Data Correlation Matrix
![cmvisualization.visualize_ld.png](20250517095032/cmvisualization.visualize_ld.png)
![cmvisualization.visualize_lde.png](20250517095032/cmvisualization.visualize_lde.png)
![cmvisualization.visualize_lddhc.png](20250517095032/cmvisualization.visualize_lddhc.png)

## Loan Data New Correlation Matrix
![cmvisualization.visualize_ldne.png](20250517095032/cmvisualization.visualize_ldne.png)

## Loan Data New Pair Plot
![ppvisualization.visualize_ldne.png](20250517095032/ppvisualization.visualize_ldne.png)

## Loan Data New Z Scores
![zsbpvisualization.visualize_ldne.png](20250517095032/zsbpvisualization.visualize_ldne.png)

## XG Boost Model Top 10 Features
![xgboost_tif___main___.png](20250517095032/xgboost_tif___main___.png)



