import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix


def _get_performance_metrics(predictions_df):
    cnf_matrix = confusion_matrix(predictions_df['actual'], predictions_df['predicted'])

    class_false_positive = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    class_false_negative = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    class_true_positive = np.diag(cnf_matrix)
    class_true_negative = cnf_matrix.sum() - (class_false_positive + class_false_negative + class_true_positive)

    class_recall = class_true_positive / (
            class_true_positive + class_false_negative)  # Sensitivity, hit rate, recall, or true positive rate
    class_specificity = class_true_negative / (
            class_true_negative + class_false_positive)  # Specificity or true negative rate
    class_precision = class_true_positive / (
            class_true_positive + class_false_positive)  # Precision or positive predictive value
    class_negative_predictive_value = class_true_negative / (
            class_true_negative + class_false_negative)  # Negative predictive value
    class_fallout = class_false_positive / (
            class_false_positive + class_true_negative)  # Fall out or false positive rate
    class_false_negative_rate = class_false_negative / (
            class_true_positive + class_false_negative)  # False negative rate
    class_false_discovery_rate = class_false_positive / (
            class_true_positive + class_false_positive)  # False discovery rate
    class_accuracy = (class_true_positive + class_true_negative) / (
            class_true_positive + class_false_positive + class_false_negative + class_true_negative)  # Overall accuracy

    recall = np.nanmean(class_recall)
    specificity = np.nanmean(class_specificity)
    precision = np.nanmean(class_precision)
    npv = np.nanmean(class_negative_predictive_value)
    fallout = np.nanmean(class_fallout)
    fnr = np.nanmean(class_false_negative_rate)
    fdr = np.nanmean(class_false_discovery_rate)
    accuracy = np.nanmean(class_accuracy)

    return recall, specificity, precision, npv, fallout, fnr, fdr, accuracy


def compute_performance_metric_model(predictions_df: pd.DataFrame, label=None) -> pd.DataFrame:
    if label != None:
        predictions_df = predictions_df[predictions_df['actual'] == label]

    metrics_df = pd.DataFrame(columns=[
        'imputation',
        'recall',
        'specificity',
        'precision',
        'fallout',
        'accuracy'
    ])

    imputations = predictions_df.imputation.unique()
    for imputation in imputations:
        imputation_df = predictions_df[predictions_df['imputation'] == imputation]

        recall, specificity, precision, npv, fallout, fnr, fdr, accuracy = _get_performance_metrics(imputation_df)
        metrics_df.loc[len(metrics_df.index)] = [
            imputation,
            recall,
            specificity,
            precision,
            fallout,
            accuracy
        ]

    melted_metrics = pd.melt(
        metrics_df,
        id_vars=['imputation'],
        var_name='metric',
        value_name='value'
    )

    melted_metrics = melted_metrics.sort_values(by=['metric', 'imputation'])

    return melted_metrics


def _get_imputation_name(directory: str) -> str:
    imputation = os.path.basename(
        os.path.dirname(
            os.path.dirname(
                os.path.normpath(
                    directory
                )
            )
        )
    )

    return imputation


def _column_is_substring(row) -> bool:
    return row.actual in row.predicted


def load_analysis_data(filename: str) -> pd.DataFrame:
    predictions_df = pd.read_csv(filename)
    predictions_df = predictions_df.sort_values(by=['actual', 'file', 'directory'])

    predictions_df['actual'] = predictions_df['actual'].str.title()
    predictions_df['predicted'] = predictions_df['predicted'].str.title()

    predictions_df['imputation'] = predictions_df.apply(
        lambda x: _get_imputation_name(x['directory']),
        axis=1
    )

    predictions_df['imputation'] = predictions_df['imputation'].str.replace('TorchVision-', '')
    predictions_df['imputation'] = np.where(predictions_df['imputation'] == 'imagenet-mini', 'None',
                                            predictions_df['imputation'])

    predictions_df['predicted'] = predictions_df['predicted'].replace(' ', '_', regex=True)
    predictions_df['predicted'] = predictions_df['predicted'].replace(',_', ' ', regex=True)

    predictions_df['correctly_classified'] = predictions_df.apply(lambda x: _column_is_substring(x), axis=1)

    return predictions_df
