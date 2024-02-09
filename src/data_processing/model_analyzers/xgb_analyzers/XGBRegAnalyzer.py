# Standard library imports
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Literal, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
import yaml
from enum import Enum

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_processing.model_analyzers.xgb_analyzers.XGBRegrResults import XGBRegrResults
from src.data_processing.preprocessing.pandas_preprocessors import xgb_reg_signal_params_only_pd_preprocessor, normalized_preprocessor,  normalize_by_baseline
from src.data_processing.pipelines.ClassifierPipe import ClassifierPipe


class XGBRegAnalyzer:
    def __init__(self, results: XGBRegrResults, metrics: List[Callable] = [mean_squared_error, mean_absolute_error, r2_score]):
        self.results: XGBRegrResults = results
        self._pipeline: ClassifierPipe = None
        self._xgb_model: xgb.XGBRegressor = None
        self.metrics: Dict = metrics
        self._metrics_results = None
        self.__datasets = None
        self.__datasets_w_predictions = None
        self._feature_names = None
        self._feature_importance_df = None
        self._shap_explainer = None
        self._shap_explanation = None

    @property
    def best_xgb_model(self) -> xgb.XGBRegressor:

        if not self._xgb_model:
            self._xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric=['rmse', 'mae'],
                **self.results.best_params)

        return self._xgb_model

    def create_pipeline(self, cls_to_drop: List[str] = None, random_seed=None, shuffle=True) -> None:

        df_processor = partial(xgb_reg_signal_params_only_pd_preprocessor,
                               query=self.results.experiment_query,
                               cls_to_drop=cls_to_drop)
        if not self._pipeline:
            self._pipeline = (ClassifierPipe(self.results.data_path)
                              .read_raw_data()
                              .pandas_pipe(df_processor)
                              .split_by_ratio(target='ratio_avoid', random_seed=random_seed, shuffle=shuffle)
                              .transform_data()
                              )
        if not self._feature_names:
            self._feature_names = self.pipeline.processor.named_transformers_[
                "num"].get_feature_names_out().tolist()
        return

    @property
    def pipeline(self) -> ClassifierPipe:
        if not self._pipeline:
            self.create_pipeline()
        return self._pipeline

    @property
    def feature_names(self) -> List[str]:
        if not self._feature_names:
            raise ValueError(
                'Pipeline not created yet, call create_pipeline() first')
        return self._feature_names

    def fit_best_xgb_model(self) -> None:
        self.best_xgb_model.fit(self.pipeline.X_train, self.pipeline.y_train)

    def predict(self, from_dataset: Literal['test', 'dev', 'train'] = 'train') -> np.ndarray:
        # predict from train by default

        predictions = self.best_xgb_model.predict(
            self._datasets[from_dataset])
        return predictions

    @property
    def _datasets(self) -> Dict[str, pd.DataFrame]:
        if not self.__datasets:
            self.__datasets = {
                'train': self.pipeline.X_train,
                'dev': self.pipeline.X_dev,
                'test': self.pipeline.X_test
            }
        return self.__datasets

    @property
    def _datasets_w_predictions(self) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        if not self.__datasets_w_predictions:
            datasets = {
                'train': (self.pipeline.X_train, self.pipeline.y_train, self.predict('train')),
                'dev': (self.pipeline.X_dev, self.pipeline.y_dev, self.predict('dev')),
                'test': (self.pipeline.X_test, self.pipeline.y_test, self.predict('test'))
            }
            self.__datasets_w_predictions = datasets
        return self.__datasets_w_predictions

    def _compute_metrics(self) -> Dict[str, Dict[str, float]]:
        sets = {'train': (self.pipeline.y_train, self.predict('train')),
                'dev': (self.pipeline.y_dev, self.predict('dev')),
                'test': (self.pipeline.y_test, self.predict('test'))}

        metrics_dict = {metric.__name__:  metric for metric in self.metrics}

        results = {}

        for name, fnc in metrics_dict.items():
            results[name] = {name: fnc(*data) for name, data in sets.items()}
        return results

    @property
    def metrics_results(self) -> Dict[str, Dict[str, float]]:
        if not self._metrics_results:
            self._metrics_results = self._compute_metrics()
        return self._metrics_results

    def _df_from_pipeline(self, from_dataset: Literal['test', 'dev', 'train'] = 'train') -> pd.DataFrame:

        df = (
            pd.DataFrame(
                (self._datasets_w_predictions[from_dataset][0]), columns=self._feature_names)
            .assign(true_values=self._datasets_w_predictions[from_dataset][1].values,
                    predictions=self._datasets_w_predictions[from_dataset][2])
        )
        return df

    @property
    def shap_explainer(self) -> shap.TreeExplainer:
        if not self._shap_explainer:
            print('Creating shap explainer')
            self._shap_explainer = shap.TreeExplainer(
                self.best_xgb_model, self.pipeline.X_train)
        return self._shap_explainer

    @property
    def shap_explanation(self) -> np.ndarray:
        if not self._shap_explanation:
            print('Creating shap values')
            self._shap_explanation = self.shap_explainer(self.pipeline.X_train)

            self._shap_explanation.feature_names = self._feature_names
        return self._shap_explanation

    @property
    def feature_importance_df(self) -> pd.DataFrame:
        if self._feature_importance_df is None:
            self._feature_importance_df = (pd.DataFrame({'feature': self._feature_names,
                                                        'importance': self.best_xgb_model.feature_importances_})
                                           .sort_values('importance', ascending=True)
                                           )
        return self._feature_importance_df

    def plot_feature_importance(self) -> pd.DataFrame:
        (self.feature_importance_df
         .plot
         .barh(x='feature',
               y='importance',
               figsize=(5, 5),
               color='darkred',
               edgecolor='black',
               linewidth=0.2,
               title='Feature importance'
               )
         )
        plt.show()

    def plot_metrics(self):

        metrics_df = (pd.DataFrame
                      .from_dict(self.metrics_results)
                      .reset_index()
                      .rename(columns={'index': 'dataset'})
                      .melt(id_vars='dataset', var_name='metrics', value_name='value')
                      )
        # metrics_df.plot.bar(x='dataset', y=metrics_df.drop(
        #     columns='dataset', ).columns.tolist())

        return (sns
                .FacetGrid(metrics_df, col='metrics', sharey=False, hue='dataset', palette=['black', 'grey', 'darkred'])
                .map(sns.barplot, 'dataset', 'value', order=['train', 'dev', 'test'], alpha=0.5)
                .set_xticklabels(rotation=45)
                .set_titles("{col_name}")
                )

    def plot_distribution(self, from_dataset: Literal['test', 'dev', 'train'] = 'test'):

        df = self._df_from_pipeline(from_dataset)

        results = df[['true_values', 'predictions']].melt(
            var_name='dataset', value_name='ratio_avoid')
        sns.displot(data=results, x='ratio_avoid', hue='dataset',
                    alpha=.6,  kind='kde', fill=True, palette=['grey', 'darkred'])
        plt.title('Distribution of Ratio Avoid for Test Set')
        plt.xlabel('Ratio Avoid')
        plt.show()

    def plot_shap_cluster_bar(self):
        clustering = shap.utils.hclust(
            self.pipeline.X_train, y=self.pipeline.y_train)
        shap.plots.bar(self.shap_explanation, clustering=clustering,
                       max_display=len(self.feature_names))

    def plot_shap_decision_plot(self):
        shap.decision_plot(self.shap_explainer.expected_value,
                           self.shap_explanation.values, self.feature_names, max_display=len(self.feature_names))

    def plot_shapley_heatmap(self):
        shap.plots.heatmap(self.shap_explanation, max_display=20,
                           instance_order=self.shap_explanation.sum(1))

    def plot_decision_tree(self, tree_index: int = 0):
        return xgb.plot_tree(self.best_xgb_model, num_trees=tree_index)

    def plot_shapley_beeswarm(self):
        shap.plots.beeswarm(self.shap_explanation,
                            max_display=len(self.feature_names))

    def plot_model_results(self):
        self.plot_metrics()
        self.plot_feature_importance()
        self.plot_distribution()
        plt.show()

    def plot_shap_results(self):
        self.plot_shap_cluster_bar()
        self.plot_shap_decision_plot()
        self.plot_shapley_heatmap()
        self.plot_shapley_beeswarm()
        plt.show()


class XGBRegAnalyzerFactory:
    def __init__(self, path_to_results: str):
        self.results = XGBRegrResults(path_to_results)

    def create_analyzer(self, cls_to_drop: List[str] = None) -> XGBRegAnalyzer:
        analyzer = XGBRegAnalyzer(self.results)
        analyzer.create_pipeline(cls_to_drop)
        analyzer.fit_best_xgb_model()
        return analyzer
# recommit


class XGBNormRegAnalyzer:
    def __init__(self, results: XGBRegrResults, metrics: List[Callable] = [mean_squared_error, mean_absolute_error, r2_score]):
        self.results: XGBRegrResults = results
        self._pipeline: ClassifierPipe = None
        self._xgb_model: xgb.XGBRegressor = None
        self.metrics: Dict = metrics
        self._metrics_results = None
        self.__datasets = None
        self.__datasets_w_predictions = None
        self._feature_names = None
        self._feature_importance_df = None

    @property
    def best_xgb_model(self) -> xgb.XGBRegressor:

        if not self._xgb_model:
            self._xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric=['rmse', 'mae'],
                **self.results.best_params)

        return self._xgb_model

    def create_pipeline(self, cls_to_drop=None, random_seed=None, shuffle=True) -> None:

        df_processor = partial(normalized_preprocessor,
                               normalizer=normalize_by_baseline,
                               query=self.results.experiment_query, 
                               experiment_cols_to_drop=cls_to_drop)
        if not self._pipeline:
            self._pipeline = (ClassifierPipe(self.results.data_path)
                              .read_raw_data()
                              .pandas_pipe(df_processor)
                              .split_by_ratio(target='ratio_avoid', random_seed=random_seed, shuffle=shuffle)
                              .transform_data()
                              )
        if not self._feature_names:
            self._feature_names = self.pipeline.processor.named_transformers_[
                "num"].get_feature_names_out().tolist()
        return

    @property
    def pipeline(self) -> ClassifierPipe:
        if not self._pipeline:
            self.create_pipeline()
        return self._pipeline

    @property
    def feature_names(self) -> List[str]:
        if not self._feature_names:
            raise ValueError(
                'Pipeline not created yet, call create_pipeline() first')
        return self._feature_names

    def fit_best_xgb_model(self) -> None:
        self.best_xgb_model.fit(self.pipeline.X_train, self.pipeline.y_train)

    def predict(self, from_dataset: Literal['test', 'dev', 'train'] = 'train') -> np.ndarray:
        # predict from train by default

        predictions = self.best_xgb_model.predict(
            self._datasets[from_dataset])
        return predictions

    @property
    def _datasets(self) -> Dict[str, pd.DataFrame]:
        if not self.__datasets:
            self.__datasets = {
                'train': self.pipeline.X_train,
                'dev': self.pipeline.X_dev,
                'test': self.pipeline.X_test
            }
        return self.__datasets

    @property
    def _datasets_w_predictions(self) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
        if not self.__datasets_w_predictions:
            datasets = {
                'train': (self.pipeline.X_train, self.pipeline.y_train, self.predict('train')),
                'dev': (self.pipeline.X_dev, self.pipeline.y_dev, self.predict('dev')),
                'test': (self.pipeline.X_test, self.pipeline.y_test, self.predict('test'))
            }
            self.__datasets_w_predictions = datasets
        return self.__datasets_w_predictions

    def _compute_metrics(self) -> Dict[str, Dict[str, float]]:
        sets = {'train': (self.pipeline.y_train, self.predict('train')),
                'dev': (self.pipeline.y_dev, self.predict('dev')),
                'test': (self.pipeline.y_test, self.predict('test'))}

        metrics_dict = {metric.__name__:  metric for metric in self.metrics}

        results = {}

        for name, fnc in metrics_dict.items():
            results[name] = {name: fnc(*data) for name, data in sets.items()}
        return results

    @property
    def metrics_results(self) -> Dict[str, Dict[str, float]]:
        if not self._metrics_results:
            self._metrics_results = self._compute_metrics()
        return self._metrics_results

    def _df_from_pipeline(self, from_dataset: Literal['test', 'dev', 'train'] = 'train') -> pd.DataFrame:
        print(self._feature_names)
        print(self.best_xgb_model.feature_importances_)
        print(len(self._feature_names))
        print(len(self.best_xgb_model.feature_importances_))
        df = (
            pd.DataFrame(
                (self._datasets_w_predictions[from_dataset][0]), columns=self._feature_names)
            .assign(true_values=self._datasets_w_predictions[from_dataset][1].values,
                    predictions=self._datasets_w_predictions[from_dataset][2])
        )
        return df

    @property
    def feature_importance_df(self) -> pd.DataFrame:

        if self._feature_importance_df is None:
            self._feature_importance_df = (pd.DataFrame({'feature': self._feature_names,
                                                        'importance': self.best_xgb_model.feature_importances_})
                                           .sort_values('importance', ascending=True)
                                           )
        return self._feature_importance_df


class XGBNormRegAnalyzerFactory:
    def __init__(self, path_to_results: str):
        self.results = XGBRegrResults(path_to_results)

    def create_analyzer(self, cls_to_drop: List[str] = None) -> XGBNormRegAnalyzer:
        analyzer = XGBNormRegAnalyzer(self.results)
        analyzer.create_pipeline(cls_to_drop)
        analyzer.fit_best_xgb_model()
        # predict from test
        # save predictions and test data
        return analyzer
