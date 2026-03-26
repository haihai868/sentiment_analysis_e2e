from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from onnx_export import export_sklearn_to_onnx
from model_registry import register_model, transition_stage


# run ngrok http 5000 in local
TRACKING_URI = "https://b31880069539.ngrok-free.app"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("sentiment analysis")

def run_experiment(
    df,
    vectorizer_type="tfidf",
    ngram_range=(1, 1),
    vectorizer_max_features=5000,

    model_type="logistic",
    model_params=None,

    model_path = "model",

    test_size=0.2,
    random_state=42
):
    """
    Run training experiment with given parameters
    """

    default_params = {
        "logistic": {
            "C": 1.0,              # regularization strength
            "max_iter": 1000,
            "solver": "lbfgs"
        },
        "nb": {
            "alpha": 1.0           # smoothing
        },
        "svm": {
            "C": 1.0               # margin control
        }
    }


    if model_params is None:
        model_params = default_params[model_type]
    else:
        model_params = {**default_params[model_type], **model_params}

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['category'],
        test_size=test_size,
        random_state=random_state
    )

    if vectorizer_type == "bow":
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=vectorizer_max_features
        )
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=vectorizer_max_features
        )
    else:
        raise ValueError("invalid vectorizer_type")

    if model_type == "logistic":
        model = LogisticRegression(**model_params)
    elif model_type == "nb":
        model = MultinomialNB(**model_params)
    elif model_type == "svm":
        model = LinearSVC(**model_params)
    else:
        raise ValueError("invalid model_type")

    with mlflow.start_run() as run:
        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("model", model)
        ])

        # X_train = vectorizer.fit_transform(X_train)
        # X_test = vectorizer.transform(X_test)


        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", str(ngram_range))
        mlflow.log_param("max_features", vectorizer_max_features)

        mlflow.log_param("model_type", model_type)
        for k, v in model_params.items():
            mlflow.log_param(k, v)

        # metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # model
        onnx_model = export_sklearn_to_onnx(
            model=pipeline,
            feature_names=vectorizer.get_feature_names_out()
        )
        mlflow.sklearn.log_model(pipeline, model_path)
        mlflow.onnx.log_model(onnx_model, f"onnx_{model_path}" )


    
    registered_version = register_model(
        run_id=run.info.run_id,
        model_path=model_path,
        description="Twitter Sentiment Analysis Model"
    )

    onnx_registered_version = register_model(
        run_id=run.info.run_id,
        model_path=f"onnx_{model_path}",
        description="ONNX version of Twitter Sentiment Analysis Model"
    )

    if f1 > 0.9 and acc > 0.9:
        transition_stage(version=registered_version, stage="Staging")
        transition_stage(version=onnx_registered_version, stage="Staging")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "model_Version": registered_version,
        "run_id": run.info.run_id
    }

# run_experiment(
#     df,
#     vectorizer_type="tfidf",
#     ngram_range=(1, 1),
#     vectorizer_max_features=5000,

#     model_type="logistic",
#     model_params=None,

#     model_path = "model",

#     test_size=0.2,
#     random_state=42
# )