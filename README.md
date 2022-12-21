# Job Salary Prediction
# 1. Problem Definition
Predict the salary for the job profile posted in Indeed using many features such as
* CompanyID
* jobType
* degree
* major
* industry
* yearsExperience
* milesFromMetropolis

Mean squared Erros(MSE) is used to calculate the prediction accuracy. Our task
is to reduce the MSE as much as possible

# 2. Exploratory Data Analysis
## 2.1 Dataset details
* train_features.csv : Each row represents metadata for an individual job posting.The “jobId” column represents a unique identifier for the job posting. The remaining columns describe features of the job posting.
* train_salaries.csv : Each row associates a “jobId” with a “salary”.
* test_features.csv : Similar to train_features.csv , each row represents
metadata for an individual job posting

## 2.2 Dataset study
* 1 million records in the training set
* Numeric variables: yearsExperience, milesFromMetropolis, salary
* Categorical variables: jobId, companyId, jobType, degree, major, industry
* jobId is all unique, not included as a feature
* companyId has 63 unique values, can not easily visualize
* The rest of the categorical variables, jobType, degree, major and industry have a small amount of unique values and can visualize
* The minimum of salary is 0, need to check
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.2.png" width="500">
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.21.png" width="900">

## 2.3 check where the salary is equal to 0
* The rows where salary is equal to 0 looks normal, it does not look like unpaid position
* There are only 5 such rows
* We can remove these rows.
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.3.png" width="900">

## 2.4 Salary plot
* Salary is following a normal distribution
* There are some outliers present above the upper bound

## 2.5 plot each features in relation to salary
### jobType
* Job types are distributed in equal amount in the training set.
* Salaries are sorted in the order(high to low): 'CEO', 'CTO', 'CFO', 'VICE_PRESIDENT', 'MANAGER', 'SENIOR', 'JUNIOR', 'JANITOR'.
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.5_jobtype.png" width="900">

### degree
* There were more high school and none degrees than other categories
* The salaries in high school and none degrees are lower than other categories
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.5_degree.png" width="900">

### major
* More than 50% of the case have none major.
* Cases that have a major are pretty evenly distributed across different majors.
* None major has a lower salary than any major.
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.5_major.png" width="900">

### industry
* There were fairly equal amount of industry types in the training set
* salaries are the highest in 'FINANCE', 'OIL'
* salaries are the lowest in 'EDUCATION', 'SERVICE'
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.5_industry.png" width="900">

### yearsExperience
* Years of experience is fairly evenly distributed across the range of 0 to 24 years.
* There is a positive linear correlation between salary and years of experience
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.5YOE.png" width="900">

### milesFromMetropolis
* Miles from metropolis is fairly evenly distributed across the range of 0 to 100 miles.
* There is a negative linear correlation between salary and miles from metropolis
<img src="https://github.com/tikna123/job-salary-prediction/main/images/2.5_miles.png" width="900">

## 3. Machine learning Models
* I have tried several machine learning models. Below is the model 
comparison on test RMSE.
* Catboost with non-encoded features performs best among all the models.
* LightGBM is fastest when it comes to training a model
<img src="https://github.com/tikna123/job-salary-prediction/main/images/3_model_rmse.png" width="900">
<img src="https://github.com/tikna123/job-salary-prediction/main/images/3_model_time.png" width="900">

# 4. Feature Importance
* 'yearsOfExperience' and 'jobType' are the 2 most important features.
<img src="https://github.com/tikna123/job-salary-prediction/main/images/feat_imp.png" width="900">

# 5. Select the base model and run on test data
* select the best model(i.e. catboost )run the inference on test_salaries.csv