import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
from .evaluation import evaluation
import logging

def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "http://127.0.0.1:5000") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id


def log_model_with_mlflow(model, X_val, y_val, model_name, exp_id, output_dir):
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        logging.info(f"Logging {model_name} to MLflow...")
        mlflow.set_tag("model", model_name)

        pred = model.predict(X_val)
        accuracy,pres,recall,f1_sc= evaluation(y_val, pred,output_dir)
         

        mlflow.log_params(model.best_params_)
        mlflow.log_metrics({
            "Mean CV score": model.best_score_,
            "Accuracy": accuracy,
            "f1-score": f1_sc,
            "recall": recall,
            "precision":pres,
        })

        mlflow.log_artifact(str(output_dir / "conf_matrix.png"))
        mlflow.log_artifact(f"pkl_files/{model_name}.pkl")

        pd_dataset = mlflow.data.from_pandas(X_val, name="validation Dataset")
        mlflow.log_input(pd_dataset, context="validation")

        signature = mlflow.models.infer_signature(X_val, y_val)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_val.iloc[[0]])