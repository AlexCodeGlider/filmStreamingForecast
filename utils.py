import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from random import choice
from itertools import combinations
from sklearn.metrics import mean_absolute_percentage_error

data_path = 'data/'

def get_country_dfs(country, main_df, talent):
    '''
    Get the dataframe of a specific country
    :param country: the country name
    :param main_df: the main dataframe
    :param talent: the talent dataframe
    :return: the dataframes of a specific country
    '''
    # Get the dataframe of a specific country
    local_df = main_df[main_df['country'] == country]

    # Get the dataframe of a specific country
    local_talent = talent.merge(local_df[['jw_entity_id', 'score']], on='jw_entity_id', how='inner')

    # Create a feature called 'talent_total_score' to measure the total score of each talent
    local_talent['talent_total_score'] = local_talent.groupby('person_id')['score'].transform(lambda x: x.sum())

    # Create a feature called 'talent_average_score' to measure the average score of each talent
    local_talent['talent_average_score'] = local_talent.groupby('person_id')['score'].transform(lambda x: x.mean())

    # Create a feature called 'talent_max_score' to measure the maximum score of each talent
    local_talent['talent_max_score'] = local_talent.groupby('person_id')['score'].transform(lambda x: x.max())

    # Create a feature called 'talent_min_score' to measure the minimum score of each talent
    local_talent['talent_min_score'] = local_talent.groupby('person_id')['score'].transform(lambda x: x.min())

    # Create a feature called 'talent_median_score' to measure the median score of each talent
    local_talent['talent_median_score'] = local_talent.groupby('person_id')['score'].transform(lambda x: x.median())

    # Create a feature called 'talent_std_score' to measure the standard deviation of each talent
    local_talent['talent_std_score'] = local_talent.groupby('person_id')['score'].transform(lambda x: x.std())

    # Create a feature called 'talent_count' to measure the number of movies of each talent
    local_talent['talent_count'] = local_talent.groupby('person_id')['score'].transform(lambda x: x.count())

    # Create a feature called 'talent_total_role_score' to measure the total score of each talent in each role
    local_talent['talent_total_role_score'] = local_talent.groupby(['person_id', 'role'])['score'].transform(lambda x: x.sum())

    # Create a feature called 'talent_average_role_score' to measure the average score of each talent in each role
    local_talent['talent_average_role_score'] = local_talent.groupby(['person_id', 'role'])['score'].transform(lambda x: x.mean())

    # Create a feature called 'talent_total_genre_role_score' to measure the total score of each talent in each genre and each role
    local_talent['talent_total_genre_role_score'] = local_talent.groupby(['person_id', 'genre_1', 'role'])['score'].transform(lambda x: x.sum())

    # Create a feature called 'talent_average_genre_role_score' to measure the average score of each talent in each genre and each role
    local_talent['talent_average_genre_role_score'] = local_talent.groupby(['person_id', 'genre_1', 'role'])['score'].transform(lambda x: x.mean())
    return local_df, local_talent

def get_colab_matrix(local_df, local_talent):
    colab_matrix = pd.DataFrame([
        [n, x, y]
        for n, g in local_talent.groupby('jw_entity_id')['person_id']
        for x, y in combinations(g, 2)
    ], columns=['jw_entity_id', 'person_id_1', 'person_id_2'])
    colab_matrix = colab_matrix.merge(local_df[['jw_entity_id', 'score']], how='left', on='jw_entity_id')
    colab_matrix = colab_matrix.groupby(['person_id_1', 'person_id_2'])[['jw_entity_id', 'score']].agg(
        {'jw_entity_id': lambda x: ','.join(x), 'score': ['count', 'sum', 'mean']}).rename(
        columns={'count': 'num_colabs', 'sum': 'colab_total', 'mean': 'colab_avg'}).reset_index()
    return colab_matrix

def graph(colab_matrix):
    G = nx.Graph()

    for p1, p2, n_c, c_t, c_a in zip(colab_matrix['person_id_1'], colab_matrix['person_id_2'],
                                     colab_matrix.score.num_colabs, colab_matrix.score.colab_total,
                                     colab_matrix.score.colab_avg):
        G.add_edge(p1, p2, attr_dict={'num_colabs': n_c, 'colab_total': c_t, 'colab_avg': c_a})

    deg_cent = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns=['deg_cent'])
    ein_cent = pd.DataFrame.from_dict(nx.eigenvector_centrality(G, max_iter=300), orient='index', columns=['ein_cent'])
    cent_measures = deg_cent.join(ein_cent).reset_index().rename(columns={'index': 'person_id'})
    
    return G, cent_measures

def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()

    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()

    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)

    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values

def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)

    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)

        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature
    return train_feature.values

def mean_target_encoding(train, test, target, categorical, alpha=5):
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)

    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)

    # Return new features to add to the model
    return train_feature, test_feature

def encode(local_df, local_pred_main, object_type):
    cols2exclude = ['jw_entity_id',
                    'id',
                    'country',
                    'original_release_year',
                    'rank',
                    'title',
                    'date',
                    'original_title',
                    'object_type',
                    'short_description',
                    'tmdb_popularity',
                    'cinema_release_date',
                    'localized_release_date']
    genre_cols = [col for col in local_df.columns if 'genre' in col]
    X = local_df.drop(cols2exclude + genre_cols[3:], axis=1, errors='ignore')
    X.index = local_df['jw_entity_id']
    X['is_pred'] = False
    if object_type['fp_name'] == 'movie':
        cols2drop = ['comentarios_vivi', 'budget', 'ask', 'sales', 'plot', 'status']
    else:
        cols2drop = []
    X = X.append(local_pred_main.drop(cols2drop, axis=1))
    X['genre_2'] = X['genre_2'].fillna(X['genre_1'])
    X['genre_3'] = X['genre_3'].fillna(X['genre_2'])
    X['is_nflx_original'] = X['is_nflx_original'].map({True: 'Yes', False: 'No', None: 'NA'})
    feats2encode = list(X.select_dtypes([object, bool]).columns)
    feats2encode.remove('is_pred')
    X_train = X[X['is_pred'] == False]
    X_train['score'] = local_df['score'].values
    X_test = X[X['is_pred'] == True]
    for categorical in feats2encode:
        X_train[categorical + '_enc'], X_test[categorical + '_enc'] = mean_target_encoding(train=X_train,
                                                                                     test=X_test,
                                                                                     target='score',
                                                                                     categorical=categorical,
                                                                                     alpha=5)

    X_train.drop(feats2encode, axis=1, inplace=True)
    X_test.drop(feats2encode, axis=1, inplace=True)

    X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_train = local_df['score']
    X_train.drop(['is_pred', 'score'], axis=1, inplace=True)
    X_test.drop(['is_pred', 'score'], axis=1, inplace=True)
    num_rows, num_cols = X_train.shape
    columns = ['_'.join(col) if type(col) == tuple else col for col in X_train.columns]
    X_train.columns = list(range(num_cols))
    X_test.columns = list(range(num_cols))

    return X_full, X_train, X_test, y_train, columns

def get_params(platform, X_data, y):
    param_dist_vals = {'bagging_fraction': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        'colsample_bytree': [0.4, 0.5, 0.6, 0.7],
                        'learning_rate': [0.05, 0.1, 0.2, 0.3],
                        'max_depth': [3, 4, 5, 6],
                        'num_leaves': [20, 27, 31, 46, 61, 81],
                        'reg_lambda': [0, 1, 2]}

    # @title Hyperparameter Tuning
    # prepare indexes for stratified cross validation
    kf = KFold(shuffle=True)
    kf.get_n_splits(X_data, y)

    n_iterations = 100

    rmse_list = []
    mae_list = []
    map_list = []
    param_list = []
    best_iter_list = []

    # loop for random search

    print("Random search start...")
    print("")

    for i in range(0, n_iterations):
        kf_split = kf.split(X_data, y)
        param_dist = {'num_leaves': choice(param_dist_vals['num_leaves']),
                        'bagging_fraction': choice(param_dist_vals['bagging_fraction']),
                        'colsample_bytree': choice(param_dist_vals['colsample_bytree']),
                        'learning_rate': choice(param_dist_vals['learning_rate']),
                        'boosting_type': 'gbdt',
                        'max_depth': choice(param_dist_vals['max_depth']),
                        'reg_lambda': choice(param_dist_vals['reg_lambda'])}

        for train_index, test_index in kf_split:
            X_train = X_data.iloc[train_index]
            y_train = y.iloc[train_index]

            X_val = X_data.iloc[test_index]
            y_val = y.iloc[test_index]

            gbm = lgb.LGBMRegressor(num_leaves=param_dist['num_leaves'],
                                    bagging_fraction=param_dist['bagging_fraction'],
                                    colsample_bytree=param_dist['colsample_bytree'],
                                    learning_rate=param_dist['learning_rate'],
                                    boosting_type=param_dist['boosting_type'],
                                    max_depth=param_dist['max_depth'],
                                    reg_lambda=param_dist['reg_lambda'],
                                    n_estimators=1000,
                                    n_jobs=-1
                                    )

            gbm.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='l2',
                    early_stopping_rounds=5,
                    verbose=0)

            # predicting
            y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration_)
            rmse = mean_squared_error(y_val, y_pred) ** 0.5
            mae = mean_absolute_error(y_val, y_pred)
            map = mean_absolute_percentage_error(y_val, y_pred)
            rmse_list.append(rmse)
            mae_list.append(mae)
            map_list.append(map)
            param_list.append(param_dist)
            best_iter_list.append(gbm.best_iteration_)

        lgbm_results = pd.DataFrame({"rmse": rmse_list,
                                        "mae": mae_list,
                                        "map": map_list,
                                        "parameters": param_list,
                                        "best_iteration": best_iter_list})

        best_params = lgbm_results['parameters'].iloc[lgbm_results['rmse'].idxmin()]
        best_iter = lgbm_results['best_iteration'].iloc[lgbm_results['rmse'].idxmin()]
        best_params_df = pd.DataFrame({0: best_params, 1: best_iter})
        print(lgbm_results.sort_values("rmse", ascending=True, axis=0).head())
    return best_params_df

def predict(platform, X_data, X_pred, y, best_params):
    best_params = best_params.iloc[:, 0]
    gbm = lgb.LGBMRegressor(num_leaves=best_params['num_leaves'],
                            bagging_fraction=best_params['bagging_fraction'],
                            colsample_bytree=best_params['colsample_bytree'],
                            learning_rate=best_params['learning_rate'],
                            boosting_type=best_params['boosting_type'],
                            max_depth=best_params['max_depth'],
                            reg_lambda=best_params['reg_lambda'],
                            n_estimators=1000,
                            n_jobs=-1
                            )

    X_train, X_val, y_train, y_val = train_test_split(X_data, y)

    gbm.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
            early_stopping_rounds=5,
            verbose=0)

    # predicting
    y_pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration_)
    return y_pred, gbm