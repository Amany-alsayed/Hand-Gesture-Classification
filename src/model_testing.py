import logging
import mlflow
from sklearn.metrics import accuracy_score, classification_report

def testing(experiment_id,X_test,y_test):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.val_Accuracy DESC"],
        max_results=1
    )

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    logging.info(f"Best model: {best_run.data.tags['mlflow.runName']}")
    logging.info(f"Validation Accuracy: {best_run.data.metrics['val_accuracy']}")
    best_model = mlflow.sklearn.load_model(model_uri)
    y_pred_test = best_model.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred_test)
    logging.info(f"Test Accuracy: {accuracy}")
    logging.info(classification_report(y_test, y_pred_test))
    return accuracy

