# **🤖 Prodigy InfoTech Internship – Task 3**
# **Customer Purchase Prediction using Decision Tree**

---

## **📌 Task Objective**

Build a Decision Tree Classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data using the Bank Marketing dataset.

---

## **📁 Dataset**
- The dataset used in this task is the Bank Marketing Dataset from the UCI Machine Learning Repository.
- It contains information about customers contacted during marketing campaigns of a Portuguese bank.
- To download the dataset: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- (Unzip the file and use bank-full.csv in your notebook or Python script)

### Key Columns:
- **age – Customer age**
- **job – Type of job**
- **marital – Marital status**
- **education – Education level**
- **default – Has credit in default**
- **balance – Account balance**
- **housing – Housing loan**
- **loan – Personal loan**
- **contact – Contact communication type**
- **duration – Last contact duration**
- **campaign – Number of contacts**
- **pdays – Days since last contact**
- **previous – Number of contacts before campaign**
- **poutcome – Outcome of previous campaign**
- **y – Target variable (Yes/No purchase)**

---

## **🛠 Tools & Libraries**
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## **📊 Exploratory Data Analysis**

### 1. Target Distribution
Bar chart showing number of customers who purchased (Yes) and did not purchase (No).

---

### 2. Purchase by Job
Shows which professions are more likely to subscribe.

---

### 3. Purchase by Marital Status
Shows relationship between marital status and purchasing behavior.

---

## **⚙️ Data Preprocessing**
- Checked missing values
- Converted categorical variables using One-Hot Encoding
- Split dataset into features (X) and target (y)
- Split into training and testing sets (80/20)

---

## **🤖 Model Building**
- Used DecisionTreeClassifier
- Trained on the training dataset
- Predicted results on test dataset

---

## **📈 Model Evaluation**
- Accuracy Score
- Classification Report
- Confusion Matrix
- Decision Tree Visualization
- Feature Importance Bar Chart

---

## **📂 Project Structure**
Prodigy_DS_Task3/
│
├── bank-full.csv
├── task3_decision_tree.ipynb
└── task3_decision_tree.py

---

## **▶ How to Run**

### **1. Install dependencies**
pip install pandas numpy matplotlib seaborn scikit-learn

### **2. Run the script or notebook**
- python task3_decision_tree.py
  OR
- task3_decision_tree.ipynb

---

## **📈 Conclusion**
- This project demonstrates how machine learning models like Decision Trees can predict customer behavior based on real-world data.
- It highlights the importance of EDA, preprocessing, and model evaluation in building reliable ML systems.

---

## **✨ Author**

**Sanjana S M**

**Prodigy Infotech**

**Data Science Intern**
