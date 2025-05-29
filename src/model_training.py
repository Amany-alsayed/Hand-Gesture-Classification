from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import logging
import joblib

def train_gredient_Boost(X_train,y_train,model_name):
    logging.info("Training gredient boosting model...")
    gbc = GradientBoostingClassifier(random_state=42)
    param_grid_gbc = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    }
    model_gb= GridSearchCV(gbc, param_grid_gbc,scoring='accuracy')
    model_gb.fit(X_train,y_train)
    joblib.dump(model_gb, f"pkl_files/{model_name}.pkl")
    return model_gb


def train_random_forest(X_train,y_train,model_name):
    logging.info("Training Random Forest model...")
    rfc=RandomForestClassifier(random_state=42)
    param_grid_rfc = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
    }
    model_rf= GridSearchCV(rfc, param_grid_rfc,scoring='accuracy')
    model_rf.fit(X_train,y_train)
    joblib.dump(model_rf, f"pkl_files/{model_name}.pkl")
    return model_rf


def train_SVM(X_train,y_train,model_name):
    logging.info("Training SVM model...")
    param_grid={
    'C': [0.1,1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
    }
    grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,   
    scoring='accuracy',
    verbose=2,
    )  
    model_svc=grid_search.fit(X_train,y_train)
    joblib.dump(model_svc, f"pkl_files/{model_name}.pkl")
    return model_svc