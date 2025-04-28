# Output
```commandline
/opt/homebrew/Cellar/python@3.12/3.12.9/venv/bin/python public/analytical_processors/lending_club.py
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
Class Weights: {0: 0.5954305253341623, 1: 3.1197068403908794}
/opt/homebrew/Cellar/python@3.12/3.12.9/venv/lib/python3.12/site-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Epoch 1/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 1s 749us/step - accuracy: 0.5113 - loss: 0.7273 - val_accuracy: 0.6493 - val_loss: 0.6529
Epoch 2/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 505us/step - accuracy: 0.6046 - loss: 0.6718 - val_accuracy: 0.6649 - val_loss: 0.6345
Epoch 3/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 468us/step - accuracy: 0.6274 - loss: 0.6542 - val_accuracy: 0.6795 - val_loss: 0.6338
Epoch 4/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 507us/step - accuracy: 0.6541 - loss: 0.6341 - val_accuracy: 0.5934 - val_loss: 0.6626
Epoch 5/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 474us/step - accuracy: 0.5933 - loss: 0.6369 - val_accuracy: 0.6456 - val_loss: 0.6267
Epoch 6/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 505us/step - accuracy: 0.5993 - loss: 0.6507 - val_accuracy: 0.6613 - val_loss: 0.6328
Epoch 7/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 465us/step - accuracy: 0.6365 - loss: 0.6366 - val_accuracy: 0.6910 - val_loss: 0.6157
Epoch 8/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 498us/step - accuracy: 0.6302 - loss: 0.6438 - val_accuracy: 0.6367 - val_loss: 0.6332
Epoch 9/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 457us/step - accuracy: 0.6209 - loss: 0.6384 - val_accuracy: 0.6394 - val_loss: 0.6316
Epoch 10/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 489us/step - accuracy: 0.6376 - loss: 0.6303 - val_accuracy: 0.6566 - val_loss: 0.6307
Epoch 11/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 460us/step - accuracy: 0.6363 - loss: 0.6213 - val_accuracy: 0.6451 - val_loss: 0.6367
Epoch 12/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 498us/step - accuracy: 0.6256 - loss: 0.6242 - val_accuracy: 0.6503 - val_loss: 0.6287
Epoch 13/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 489us/step - accuracy: 0.6155 - loss: 0.6337 - val_accuracy: 0.6696 - val_loss: 0.6243
Epoch 14/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 501us/step - accuracy: 0.6380 - loss: 0.6223 - val_accuracy: 0.6775 - val_loss: 0.6193
Epoch 15/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 480us/step - accuracy: 0.6516 - loss: 0.6234 - val_accuracy: 0.6420 - val_loss: 0.6414
Epoch 16/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 476us/step - accuracy: 0.6258 - loss: 0.6346 - val_accuracy: 0.6696 - val_loss: 0.6352
Epoch 17/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 495us/step - accuracy: 0.6227 - loss: 0.6316 - val_accuracy: 0.6696 - val_loss: 0.6242
Epoch 18/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 469us/step - accuracy: 0.6384 - loss: 0.6268 - val_accuracy: 0.6775 - val_loss: 0.6080
Epoch 19/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 500us/step - accuracy: 0.6380 - loss: 0.6220 - val_accuracy: 0.6660 - val_loss: 0.6191
Epoch 20/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 485us/step - accuracy: 0.6309 - loss: 0.6203 - val_accuracy: 0.6816 - val_loss: 0.6086
Epoch 21/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 502us/step - accuracy: 0.6407 - loss: 0.6251 - val_accuracy: 0.6905 - val_loss: 0.6074
Epoch 22/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 473us/step - accuracy: 0.6444 - loss: 0.6195 - val_accuracy: 0.6842 - val_loss: 0.6130
Epoch 23/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 504us/step - accuracy: 0.6440 - loss: 0.6150 - val_accuracy: 0.6822 - val_loss: 0.6131
Epoch 24/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 471us/step - accuracy: 0.6591 - loss: 0.6043 - val_accuracy: 0.6498 - val_loss: 0.6384
Epoch 25/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 489us/step - accuracy: 0.6349 - loss: 0.6161 - val_accuracy: 0.6884 - val_loss: 0.6185
Epoch 26/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 467us/step - accuracy: 0.6449 - loss: 0.6327 - val_accuracy: 0.6806 - val_loss: 0.6182
Epoch 27/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 491us/step - accuracy: 0.6366 - loss: 0.6245 - val_accuracy: 0.6947 - val_loss: 0.6044
Epoch 28/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 490us/step - accuracy: 0.6476 - loss: 0.6254 - val_accuracy: 0.6942 - val_loss: 0.6044
Epoch 29/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 492us/step - accuracy: 0.6491 - loss: 0.6161 - val_accuracy: 0.6409 - val_loss: 0.6342
Epoch 30/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 474us/step - accuracy: 0.6335 - loss: 0.6298 - val_accuracy: 0.6566 - val_loss: 0.6288
Epoch 31/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 493us/step - accuracy: 0.6446 - loss: 0.6138 - val_accuracy: 0.6722 - val_loss: 0.6133
Epoch 32/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 462us/step - accuracy: 0.6586 - loss: 0.6158 - val_accuracy: 0.6848 - val_loss: 0.6177
Epoch 33/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 478us/step - accuracy: 0.6669 - loss: 0.5977 - val_accuracy: 0.6555 - val_loss: 0.6249
Epoch 34/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 460us/step - accuracy: 0.6338 - loss: 0.6105 - val_accuracy: 0.6581 - val_loss: 0.6369
Epoch 35/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 480us/step - accuracy: 0.6678 - loss: 0.5975 - val_accuracy: 0.6227 - val_loss: 0.6525
Epoch 36/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 480us/step - accuracy: 0.6359 - loss: 0.6128 - val_accuracy: 0.6879 - val_loss: 0.6012
Epoch 37/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 486us/step - accuracy: 0.6385 - loss: 0.6169 - val_accuracy: 0.6806 - val_loss: 0.6157
Epoch 38/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 477us/step - accuracy: 0.6658 - loss: 0.6114 - val_accuracy: 0.6665 - val_loss: 0.6248
Epoch 39/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 502us/step - accuracy: 0.6551 - loss: 0.6042 - val_accuracy: 0.6785 - val_loss: 0.6258
Epoch 40/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 474us/step - accuracy: 0.6615 - loss: 0.6128 - val_accuracy: 0.6968 - val_loss: 0.5981
Epoch 41/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 493us/step - accuracy: 0.6572 - loss: 0.6006 - val_accuracy: 0.6733 - val_loss: 0.6214
Epoch 42/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 464us/step - accuracy: 0.6430 - loss: 0.6196 - val_accuracy: 0.6785 - val_loss: 0.6111
Epoch 43/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 561us/step - accuracy: 0.6521 - loss: 0.6232 - val_accuracy: 0.7067 - val_loss: 0.5930
Epoch 44/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 475us/step - accuracy: 0.6567 - loss: 0.6223 - val_accuracy: 0.6905 - val_loss: 0.6138
Epoch 45/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 493us/step - accuracy: 0.6625 - loss: 0.6036 - val_accuracy: 0.6926 - val_loss: 0.6091
Epoch 46/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 472us/step - accuracy: 0.6593 - loss: 0.6119 - val_accuracy: 0.7020 - val_loss: 0.6087
Epoch 47/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 497us/step - accuracy: 0.6702 - loss: 0.6112 - val_accuracy: 0.7213 - val_loss: 0.5910
Epoch 48/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 470us/step - accuracy: 0.6713 - loss: 0.6096 - val_accuracy: 0.7004 - val_loss: 0.6020
Epoch 49/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 497us/step - accuracy: 0.6755 - loss: 0.5905 - val_accuracy: 0.6592 - val_loss: 0.6248
Epoch 50/50
240/240 ━━━━━━━━━━━━━━━━━━━━ 0s 469us/step - accuracy: 0.6552 - loss: 0.6037 - val_accuracy: 0.6691 - val_loss: 0.6214
60/60 ━━━━━━━━━━━━━━━━━━━━ 0s 387us/step

Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.68      0.78      1611
           1       0.27      0.62      0.37       305

    accuracy                           0.67      1916
   macro avg       0.59      0.65      0.57      1916
weighted avg       0.80      0.67      0.71      1916

Test AUC Score: 0.6950
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
Model saved successfully to loan_default_keras_model.h5!
-----------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------
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
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
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
public/analytical_processors/wrangling/prepare.py:35: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
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
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
    default_probability_percent  predicted_class
0                     33.253212                0
1                     17.123117                0
2                     47.052078                0
3                     51.133556                1
4                     56.945766                1
5                      4.683031                0
6                     53.945057                1
7                     44.567673                0
8                     47.383392                0
9                     37.217052                0
10                    54.526722                1
11                     1.147902                0
12                    53.407467                1
13                     4.285240                0
-----------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------
Imbalance Ratio: 5.24
/opt/homebrew/Cellar/python@3.12/3.12.9/venv/lib/python3.12/site-packages/xgboost/training.py:183: UserWarning: [13:09:37] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:738: 
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