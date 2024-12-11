
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('C:/Users/marya/OneDrive/Desktop/Assignment/Assignment Applied DS/UCI_Heart_Disease_Dataset_Combined.csv')
df


df.info()


df.describe()


sns.countplot(x='ChestPainType', data=df, hue='HeartDisease')
plt.title('Count of Chest Pain Types by Heart Disease Status')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()


sns.violinplot(y='RestingBP', x='HeartDisease', data=df, orient='v')
plt.title('Distribution of Resting Blood Pressure by Heart Disease Status')
plt.xlabel('Heart Disease Status (0: No, 1: Yes)')
plt.ylabel('Resting Blood Pressure (RestingBP)')
plt.show()


sns.scatterplot(data=df, y='Cholesterol', x='RestingBP', hue='HeartDisease')
# Adding Title and Labels
plt.title('Relationship Between Resting Blood Pressure and Cholesterol by Heart Disease Status')
plt.xlabel('Resting Blood Pressure (RestingBP)')
plt.ylabel('Cholesterol Level')
plt.legend(title='Heart Disease')
plt.show()


sns.lineplot(data=df, hue='FastingBS', x='MaxHR', y='Cholesterol')
plt.title('Cholesterol Levels vs Max Heart Rate by Fasting Blood Sugar Status')
plt.xlabel('Maximum Heart Rate (MaxHR)')
plt.ylabel('Cholesterol Level')
plt.legend(title='Fasting Blood Sugar (FastingBS)',
           labels=['0: Normal', '1: High'])
plt.show()


def k_means_fitting_no_plot(data, k):
    """
    Fits a K-means model to the provided dataset and returns the inertia.

    Arguments:
        data (DataFrame): The dataset to fit the K-means model to. It should include numerical columns
                           for clustering, and optionally, a 'HeartDisease' column to be excluded from clustering.
        k (int): The number of clusters to form.

    Returns:
        tuple: A tuple containing the following:
            model (KMeans): The trained KMeans model.
            float: The inertia (sum of squared distances of samples to their closest cluster center),
                which indicates the quality of the clustering. Lower values indicate better clustering.
    """
    # Fitting KMeans model without visualization
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')

    # Dropping 'HeartDisease' column
    model.fit(data.drop(columns=['HeartDisease']))
    return model, model.inertia_


def k_means_clustering_no_plot(model, data):
    """
    Performs K-means clustering on the provided dataset and returns the cluster labels and model.

    Arguments:
        model (KMeans): he trained KMeans model.
        data (DataFrame): The dataset to apply K-means clustering to. It should include numerical columns
                           for clustering, and optionally, a 'HeartDisease' column to be excluded from clustering.

    Returns:
        cluster_labels (numpy.ndarray): An array of cluster labels for each data point.
    """

    # Dropping 'HeartDisease' column
    cluster_labels = model.predict(data.drop(columns=['HeartDisease']))
    return cluster_labels


model, interia = k_means_fitting_no_plot(df, 3)
df['Cluster'] = k_means_clustering_no_plot(model, df)
# Visualize clusters
sns.scatterplot(data=df, x='Cholesterol', y='MaxHR',
                hue='Cluster', palette='coolwarm', s=100)
plt.title('K-Means Clustering: Cholesterol vs MaxHR')
plt.xlabel('Cholesterol')
plt.ylabel('MaxHR')
plt.legend(title='Cluster')
plt.show()


wcss_list = []


for i in range(1, 11):
    _, temp = k_means_fitting_no_plot(df, i)
    wcss_list.append(temp)
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('Within Cluster Sum of Squares (WCSS)')
plt.show()
