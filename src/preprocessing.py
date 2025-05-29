import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split

def recenter_landmarks(landmarks):
  landmarks=landmarks.copy()
  wrist_x, wrist_y = landmarks.iloc[:,0], landmarks.iloc[:,1]
  # Subtract wrist coordinates from all landmarks
  for i in range(21):  # 21 landmarks
      landmarks.iloc[:,i * 3] -= wrist_x
      landmarks.iloc[:,i * 3 + 1] -= wrist_y

  return landmarks


def normalize_landmarks(landmarks):
    mid_finger_x, mid_finger_y = landmarks.iloc[:,9 * 3], landmarks.iloc[:,9 * 3 + 1]
    norm=np.sqrt(mid_finger_x**2,mid_finger_y**2)
    # Prevent division by zero
    norm[norm==0]=1
    # Normalize landmarks
    for i in range(21):  # 21 landmarks
        landmarks.iloc[:,i * 3] /= norm
        landmarks.iloc[:,i * 3 + 1] /= norm
    return landmarks

def load_and_preprocess_data(file_path):
    logging.info("Loading dataset...")
    data = pd.read_csv(file_path)
    X=data.iloc[:,:63]
    Y=data.iloc[:,-1]
    X_train,Xvaltest,y_train,y_valtest=train_test_split(X,Y,test_size=0.4,random_state=42,stratify=Y)
    X_val,X_test,y_val,y_test=train_test_split(Xvaltest,y_valtest,test_size=0.5,random_state=42,stratify=y_valtest)
    
    X_train=recenter_landmarks(X_train)
    X_val=recenter_landmarks(X_val)
    X_test=recenter_landmarks(X_test)

    X_train=normalize_landmarks(X_train)
    X_val=normalize_landmarks(X_val)
    X_test=normalize_landmarks(X_test)

    encoder=LabelEncoder()
    y_train=encoder.fit_transform(y_train)
    y_val=encoder.transform(y_val)
    y_test=encoder.transform(y_test)

    scaling=RobustScaler()
    X_train=scaling.fit_transform(X_train)
    X_val=scaling.transform(X_val)
    X_test=scaling.transform(X_test)
     
    joblib.dump(scaling, "pkl_files/scaling.pkl")
    
    return X_train,X_val,X_test,y_train,y_val,y_test
