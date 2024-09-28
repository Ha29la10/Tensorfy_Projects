# **Project 2: Semi-Supervised Learning for Heart Disease Prediction**

## **Project Description**

This project focuses on developing an advanced machine learning model that leverages semi-supervised learning techniques to predict heart disease using clinical data. The primary goal is to improve predictive accuracy by utilizing both labeled and unlabeled data, which is particularly useful in real-world scenarios where labeled data is limited.

## **Overview**

Heart disease prediction is a critical task in healthcare, where accurate diagnosis can significantly impact patient outcomes. However, acquiring labeled clinical data is often costly and time-consuming. This project addresses this challenge by employing semi-supervised learning techniques that use both labeled and unlabeled data, enhancing the model's ability to make accurate predictions even when labeled data is scarce.

## **Objectives**

- **Implement Semi-Supervised Learning**: Utilize techniques like Label Spreading, Label Propagation to develop the best possible predictive model.
- **Maximize Data Utility**: Effectively use both labeled and unlabeled data to improve the model’s performance, simulating real-world scenarios where only a fraction of data points are labeled.

## **Core Features**

1. **Data Preprocessing**:
   - Clean and preprocess clinical data to handle missing values, outliers, and inconsistent entries.
   - Standardize numerical variables to ensure optimal model performance and reduce the influence of scale differences.

2. **Label Management**:
   - Implement innovative approaches to mask labels, ensuring a balanced representation of labeled and unlabeled data across different classes.
   - Simulate real-world scenarios by masking a significant portion of the data to reflect cases where not all data points are labeled.

3. **Model Training**:
   - Employ semi-supervised learning algorithms such as Label Spreading and Label Propagation to utilize both labeled and unlabeled data during training.
   - Incorporate SVM for classification tasks, enhancing the model's ability to generalize from limited labeled data.

4. **Evaluation**:
   - Evaluate the model using standard metrics such as accuracy, precision, recall, and F1-score.
   - Visualize the results with confusion matrices and other graphical tools to assess classification performance.
   - Leverage both labeled and unlabeled data during evaluation to ensure a comprehensive understanding of the model’s effectiveness.

## **Observations**

- **Age and Heart Rate**: There is a slightly negative correlation between age and heart rate, indicating that as age increases, heart rate tends to slightly decrease.
- **Max Heart Rate**: The majority of individuals in the sample have maximum heart rate levels between 140 and 180 beats per minute, with a smaller number having higher or lower heart rate levels.
- **Cholesterol Levels**: Most individuals have cholesterol levels between 200 and 300 mg/dL. There are fewer individuals with cholesterol levels outside this range.
- **Resting Blood Pressure**: The majority of the sample population has resting blood pressure levels between 120 and 140 mmHg, with a smaller subset having blood pressure levels outside this range.

These observations help highlight the typical ranges of critical health indicators in the sample, providing valuable context for understanding the data and guiding model development.

## **Project Workflow**

1. **Data Collection and Preprocessing**:
   - Acquire clinical data relevant to heart disease prediction.
   - Clean and preprocess the data, including missing value imputation, outlier detection, and standardization of features.

2. **Label Masking and Management**:
   - Randomly mask a significant portion of the dataset to create a mix of labeled and unlabeled data.
   - Ensure balanced representation by carefully managing the distribution of labels across the dataset.

3. **Model Development**:
   - Implement Label Spreading and Label Propagation algorithms to learn from both labeled and unlabeled data.
    

4. **Model Evaluation**:
   - Assess the model’s performance using labeled test data, focusing on key metrics such as accuracy .
   - Visualize performance through confusion matrices to identify strengths and weaknesses in classification.
 

 

## **Results and Insights**

- The semi-supervised learning approach significantly improved model performance, demonstrating the value of unlabeled data in enhancing predictive accuracy.
- Label Spreading and Label Propagation effectively utilized the unlabeled data to generalize well on the heart disease classification task.
 

 

---

**Disclaimer**: This project is intended for educational purposes and aims to explore the application of semi-supervised learning techniques in healthcare data analysis.
