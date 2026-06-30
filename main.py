import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,Ridge,Lasso,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score,classification_report,confusion_matrix,mean_squared_error,r2_score

df = pd.read_csv('data.csv')

df = df.drop(['customerID'], axis=1)

# Fix TotalCharges (was stored as text)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check how many NaNs this created
print(df['TotalCharges'].isnull().sum())
df['TotalCharges']=df['TotalCharges'].fillna(0)

# Encode binary columns FIRST
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# NOW one-hot encode remaining multi-category text columns
df = pd.get_dummies(df, drop_first=True)
#changing true false to 1 nd 0
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

#print(df.info())
#print(df.shape)
#print(df.head())

# split data into x nd y

X=df.drop(['Churn'],axis=1)
y=df['Churn']

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#PCA

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)
#print("oringnal shape",X_scaled.shape)
#print("new shape",X_pca.shape)
#print("xplained variance :",pca.explained_variance_ratio_)


#kmeans
k_means_log=KMeans(n_clusters=2,n_init=10,random_state=42)
k_means_log.fit(X_pca)

labels=k_means_log.labels_
centroids=k_means_log.cluster_centers_

#graph 
#plt.scatter(X_pca[:, 0],X_pca[:, 1],c=y,cmap='viridis')
#plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
##plt.legend()
#plt.xlabel("other featues")
#plt.ylabel("churn")
#plt.title('tel churn — Reduced to 2D via PCA')

#plt.show()

X_train,X_tesxt,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_scaled_train=scaler.fit_transform(X_train)
X_scaled_test=scaler.transform(X_tesxt)


log_logistic=LogisticRegression(random_state=42,max_iter=100)
log_logistic.fit(X_scaled_train,y_train)
pred_log=log_logistic.predict(X_scaled_test)



log_rf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
log_rf.fit(X_scaled_train,y_train)
pred_rf=log_rf.predict(X_scaled_test)
#print(pred_rf)

log_lg_proba=log_logistic.predict_proba(X_scaled_test)[:,1]
log_rf_proba=log_rf.predict_proba(X_scaled_test)[:,1]

print("classification report of logisctic regressor")
print(classification_report(y_test,pred_log))
print("confusion matrix of logisticregressin :")
print(confusion_matrix(y_test,pred_log))

#roc auc

log_fpr,log_tpr,_=roc_curve(y_test,log_lg_proba)
log_auc=roc_auc_score(y_test,log_lg_proba)

# for rf
print("classification report of random forrest")
print(classification_report(y_test,pred_rf))
print("confusion matrix of random forrest :")
print(confusion_matrix(y_test,pred_rf))

rf_fpr,rf_tpr,_=roc_curve(y_test,log_rf_proba)
rf_auc=roc_auc_score(y_test,log_rf_proba)

# 7. Plot both curves together
plt.plot(log_fpr, log_tpr, label=f'Logistic Regression (AUC = {log_auc:.3f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

print("Logistic Regression AUC:", log_auc)
print("Random Forest AUC:", rf_auc)

importances = log_rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)
print(importance_df.head(10))

df['cluster'] = labels
print(df.groupby('cluster')['Churn'].mean())
print(df.groupby('cluster')[['tenure', 'MonthlyCharges']].mean())

lasso_mod=Lasso(alpha=1.0)
# Separate regression setup -- predicting MonthlyCharges, not Churn
X_reg = df.drop(['Churn', 'MonthlyCharges','TotalCharges', 'cluster'], axis=1)
y_reg = df['MonthlyCharges']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

lr = LinearRegression()
lr.fit(X_reg_train_scaled, y_reg_train)

ridge_mod = Ridge(alpha=1.0)
ridge_mod.fit(X_reg_train_scaled, y_reg_train)

lasso_mod = Lasso(alpha=1.0)
lasso_mod.fit(X_reg_train_scaled, y_reg_train)

for name, model in [('Linear', lr), ('Ridge', ridge_mod), ('Lasso', lasso_mod)]:
    pred = model.predict(X_reg_test_scaled)
    print(f"{name} R2: {r2_score(y_reg_test, pred):.3f}, MSE: {mean_squared_error(y_reg_test, pred):.3f}")
