from src.preprocessing import load_and_preprocess_data
from src.model_training import train_random_forest, train_SVM, train_gredient_Boost
from src.mlflowLogging import log_model_with_mlflow, setup_mlflow_experiment
from pathlib import Path
from colorama import Fore, Style
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - {Fore.BLUE}%(levelname)s{Style.RESET_ALL} - %(message)s"
    )

def main():
    experiment_id = setup_mlflow_experiment("Hand_Gesture_Classification")
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "dataset/hand_landmarks_data.csv"
    output_dir = BASE_DIR / "plots"
    
    X_train,X_val,X_test,y_train,y_val,y_test = load_and_preprocess_data(data_path)
    
    rf_model = train_random_forest(X_train, y_train,"RandomForestClassifier")
    log_model_with_mlflow(rf_model, X_val, y_val, "RandomForestClassifier", experiment_id, output_dir)
    
    gb_model = train_gredient_Boost(X_train, y_train,"GredientBoostingClassifier")
    log_model_with_mlflow(gb_model, X_val, y_val, "GredientBoostingClassifier", experiment_id, output_dir)
    
    sv_model = train_SVM(X_train, y_train,"SVMClassifier")
    log_model_with_mlflow(sv_model, X_val, y_val, "SVMClassifier", experiment_id, output_dir)
    
     

if __name__ == "__main__":
    main()