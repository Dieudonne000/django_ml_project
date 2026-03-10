from __future__ import annotations

import pandas as pd
from django.shortcuts import render

from predictor.data_exploration import (
    data_exploration,
    dataset_exploration,
    rwanda_clients_map,
)


def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_clients_map": rwanda_clients_map(df),
    }
    return render(request, "predictor/index.html", context)


def regression_analysis(request):
    # Imported lazily so the server can start even before models exist.
    from model_generators.regression.train_regression import (
        get_regression_model,
        evaluate_regression_model,
        predict_regression_price,
    )

    regression_bundle = get_regression_model()
    context = {"evaluations": evaluate_regression_model()}
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = predict_regression_price(
            regression_bundle,
            year=year,
            kilometers_driven=km,
            seating_capacity=seats,
            estimated_income=income,
        )
        context["price"] = float(prediction)
    return render(request, "predictor/regression_analysis.html", context)


def classification_analysis(request):
    from model_generators.classification.train_classifier import (
        get_classification_model,
        evaluate_classification_model,
    )

    classification_model = get_classification_model()
    context = {"evaluations": evaluate_classification_model()}
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = classification_model.predict([[year, km, seats, income]])[0]
        context["prediction"] = prediction
    return render(request, "predictor/classification_analysis.html", context)


def clustering_analysis(request):
    from model_generators.clustering.train_cluster import (
        get_clustering_bundle,
        evaluate_clustering_model,
        predict_cluster_id,
    )
    from model_generators.regression.train_regression import (
        get_regression_model,
        predict_regression_price,
    )

    regression_bundle = get_regression_model()
    evaluations = evaluate_clustering_model()
    cluster_bundle = get_clustering_bundle()
    cluster_mapping = cluster_bundle["mapping"]

    context = {"evaluations": evaluations}
    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])

            predicted_price = predict_regression_price(
                regression_bundle,
                year=year,
                kilometers_driven=km,
                seating_capacity=seats,
                estimated_income=income,
            )
            cluster_id = predict_cluster_id(
                cluster_bundle,
                estimated_income=income,
                selling_price=float(predicted_price),
                seating_capacity=seats,
            )
            context.update(
                {
                    "prediction": cluster_mapping.get(cluster_id, "Unknown"),
                    "price": float(predicted_price),
                }
            )
        except Exception as e:
            context["error"] = str(e)
    return render(request, "predictor/clustering_analysis.html", context)
