# Heart Disease Analysis

This project aims to analyze a heart disease dataset using various clustering techniques and evaluate the clustering performance. The primary goal is to group patients based on their medical history and identify risk factors associated with heart disease.

## Dataset

The dataset used in this analysis is the Heart Disease UCI dataset, which contains the following features:

- **age**: Age of the patient
- **sex**: Gender of the patient (1 = male, 0 = female)
- **chest_pain_type**: Type of chest pain experienced by the patient
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol in mg/dl
- **fast_blood_sugar**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalch**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
- **target**: Diagnosis of heart disease (1 = presence of heart disease, 0 = absence of heart disease)

## Preprocessing

1. **Handling Missing Values**: Rows with missing values were dropped.
2. **Scaling Numerical Features**: Numerical features were scaled using `RobustScaler` to handle outliers.
3. **Encoding Categorical Variables**: Categorical variables were encoded using `OrdinalEncoder`.

## Clustering Techniques

The following clustering techniques were applied to the preprocessed dataset:

1. **K-means Clustering**
2. **Hierarchical Clustering**
3. **DBSCAN Clustering**
4. **Gaussian Mixture Model (GMM)**

### Visualization

To visualize the clustering results, Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE) were used to reduce the data dimensionality to 2 components.

### Evaluation

The clustering performance was evaluated using the following metrics:

1. **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters.
2. **Davies-Bouldin Index**: Measures the average similarity ratio of each cluster with the cluster that is most similar to it.

### Results

The clustering algorithms were evaluated, and their performance metrics were compared to choose the best algorithm. 

The best algorithm was determined based on the highest Silhouette Score and the lowest Davies-Bouldin Index.

## Conclusion

This analysis provided insights into the grouping of patients based on their medical history and helped identify potential risk factors associated with heart disease. The comparison of different clustering techniques highlighted the strengths and weaknesses of each method, aiding in the selection of the most suitable clustering algorithm for this dataset.

## Dependencies

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly

## How to Run

1. Install the required libraries.
2. Preprocess the dataset by scaling and encoding.
3. Apply the clustering techniques.
4. Evaluate the clustering performance using Silhouette Score and Davies-Bouldin Index.
5. Visualize the clusters using PCA and t-SNE.

## Acknowledgments

This analysis is based on the Heart Disease UCI dataset, available from the UCI Machine Learning Repository.

