import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from random import choice
from itertools import combinations, product
from sklearn.metrics import mean_absolute_percentage_error
from TextRank4Keyword import TextRank4Keyword # PageRank based keyword extraction
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

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
    #local_talent['talent_total_genre_role_score'] = local_talent.groupby(['person_id', 'genre_1', 'role'])['score'].transform(lambda x: x.sum())

    # Create a feature called 'talent_average_genre_role_score' to measure the average score of each talent in each genre and each role
    #local_talent['talent_average_genre_role_score'] = local_talent.groupby(['person_id', 'genre_1', 'role'])['score'].transform(lambda x: x.mean())
    return local_df, local_talent

def get_colab_matrix(local_df, local_talent):
    """
    Get the collaboration matrix of a specific country
    :param local_df: the dataframe of a specific country
    :param local_talent: the talent dataframe of a specific country
    :return: the collaboration matrix of a specific country
    """
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
    """
    Get the graph of a specific country
    :param colab_matrix: the collaboration matrix of a specific country
    :return: the graph of a specific country
    """
    G = nx.Graph()

    for p1, p2, n_c, c_t, c_a in zip(colab_matrix['person_id_1'], colab_matrix['person_id_2'],
                                     colab_matrix.score.num_colabs, colab_matrix.score.colab_total,
                                     colab_matrix.score.colab_avg):
        G.add_edge(p1, p2, attr_dict={'num_colabs': n_c, 'colab_total': c_t, 'colab_avg': c_a})

    deg_cent = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns=['deg_cent'])
    ein_cent = pd.DataFrame.from_dict(nx.eigenvector_centrality(G, max_iter=300), orient='index', columns=['ein_cent'])
    prank_cent = pd.DataFrame.from_dict(nx.pagerank(G, alpha=0.85), orient='index', columns=['prank_cent'])
    cent_measures = deg_cent.join(ein_cent).join(prank_cent).reset_index().rename(columns={'index': 'person_id'})

    return G, cent_measures

def extract_keywords(plot_series):
    """
    Extract keywords from the plot of each movie
    :param plot_series: the plot series of each movie
    :return: the keywords dataframe
    """
    tr4w = TextRank4Keyword()
    keywords = pd.DataFrame()

    print('Extracting keywords from the plot of each of {} movies...'.format(len(plot_series)))

    # iterate through each movie's plot
    for i, row in tqdm(plot_series.items()):
        try:
            tr4w.analyze(row, candidate_pos = ['NOUN'], window_size=4, lower=False)
        except TypeError:
            continue
        local_df = pd.DataFrame(tr4w.node_weight.items())
        local_df['jw_entity_id'] = i
        keywords = pd.concat([keywords, local_df], ignore_index=True)

    keywords.rename(columns={0:'keyword', 1:'node_weight'}, inplace=True)

    # Drop stopwords from the keywords dataframe
    keywords = keywords[~keywords['keyword'].isin(stopwords.words('english'))]

    # Drop punctuation from the keywords dataframe
    keywords = keywords[~keywords['keyword'].str.contains(r'[^\w\s]')]

    # Normalize node weights by dividing by the sum of all node weights for each movie
    keywords['node_weight_normalized'] = keywords.groupby('jw_entity_id')['node_weight'].transform(lambda x: x/x.sum())
    
    print('Total number of keywords extracted: {}'.format(len(keywords)))

    return keywords

def score_keywords(plot_series, main_df):
    """
    Score the keywords extracted from the plot of each movie
    :param plot_series: the plot series of each movie
    :param main_df: the main dataframe
    :return: the scored keywords dataframe
    """
    keywords = extract_keywords(plot_series)

    keywords_score = keywords.merge(main_df[['country', 'jw_entity_id', 'score']], on='jw_entity_id', how='inner')

    # Weigh the node weights by the movie's score
    keywords_score['node_weight_scored'] = keywords_score['node_weight_normalized'] * keywords_score['score']

    # Create a dataframe called 'keywords_scored_by_keyword_and_country' with the 'node_weight_scored' column summed by 'keyword' and 'country'
    keywords_scored_by_keyword_and_country = keywords_score.groupby(['keyword', 'country'])['node_weight_scored'].sum().reset_index()

    # Merge the 'keywords_scored' dataframe with the 'keywords_scored_by_keyword_and_country' dataframe on 'keyword' and 'country'
    keywords_scored = keywords_score.merge(keywords_scored_by_keyword_and_country, on=['keyword', 'country'], suffixes=('', '_by_keyword_and_country'))

    return keywords_scored

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

def encode(local_df, pred_set):
  """
    Encode the dataframes
  """
  cols2exclude = [
      'country', 
      'jw_entity_id', 
      'rank', 
      'is_nflx_original', 
      'date',
      'age_certification', 
      'object_type', 
      'original_release_year',
      'original_title', 
      'production_countries', 
      'runtime',
      'short_description', 
      'title', 
      'localized_release_date', 
      'genres',
      'comentarios_vivi', 
      'budget', 
      'ask', 
      'sales', 
      'plot', 
      'status',
      'market'
      ]
  genre_cols = [col for col in local_df.columns if 'genre' in col]
  X_train = local_df.drop(cols2exclude + genre_cols[4:], axis=1, errors='ignore')
  X_train.index = local_df['jw_entity_id']
  X_pred = pred_set.drop(cols2exclude + genre_cols[4:], axis=1, errors='ignore')
  X_pred.index = pred_set['title']
  X_train['genre_2'] = X_train['genre_2'].fillna(X_train['genre_1'])
  X_train['genre_3'] = X_train['genre_3'].fillna(X_train['genre_2'])
  X_pred['genre_2'] = X_pred['genre_2'].fillna(X_pred['genre_1'])
  X_pred['genre_3'] = X_pred['genre_3'].fillna(X_pred['genre_2'])
  feats2encode = ['genre_1', 'genre_2', 'genre_3']
  for categorical in feats2encode:
      X_train[categorical + '_enc'], X_pred[categorical + '_enc'] = mean_target_encoding(train=X_train,
                                                                                      test=X_pred,
                                                                                      target='score',
                                                                                      categorical=categorical,
                                                                                      alpha=5)

  X_train.drop(feats2encode, axis=1, inplace=True)
  X_pred.drop(feats2encode, axis=1, inplace=True)
  y_data = X_train['score']
  X_train.drop('score', axis=1, inplace=True)
  # Concatenate the 'X' and 'X_pred' dataframes
  X = pd.concat([X_train, X_pred], axis=0)

  # Fill null values with 0
  X.fillna(0, inplace=True)

  col_names = X.columns
  X.columns = range(X.shape[1])

  # Separate back into 'X_train' and 'X_pred'
  X_data = X.iloc[:X_train.shape[0], :]
  X_pred = X.iloc[X_train.shape[0]:, :]

  return X_data, y_data, X_pred, col_names

def get_lgbm_params(X_data, y_data):
    """
    Returns a dataframe of parameters for the LGBMRegressor
    """
    param_dist_vals = {'bagging_fraction': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        'colsample_bytree': [0.4, 0.5, 0.6, 0.7],
                        'learning_rate': [0.05, 0.1, 0.2, 0.3],
                        'max_depth': [3, 4, 5, 6],
                        'num_leaves': [20, 27, 34, 46, 61, 81],
                        'reg_lambda': [0, 1, 2]}

    # Filter out incompatible max_depth and num_leaves combinations
    compatible_params = [(md, nl) for md, nl in product(param_dist_vals['max_depth'], param_dist_vals['num_leaves']) if 2 ** md > nl]

    # Hyperparameter Tuning
    # prepare indexes for stratified cross validation
    kf = KFold(shuffle=True)
    kf.get_n_splits(X_data, y_data)

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
        kf_split = kf.split(X_data, y_data)
        max_depth, num_leaves = choice(compatible_params)
        param_dist = {'num_leaves': num_leaves,
                        'bagging_fraction': choice(param_dist_vals['bagging_fraction']),
                        'colsample_bytree': choice(param_dist_vals['colsample_bytree']),
                        'learning_rate': choice(param_dist_vals['learning_rate']),
                        'boosting_type': 'gbdt',
                        'max_depth': max_depth,
                        'reg_lambda': choice(param_dist_vals['reg_lambda'])}

        for train_index, test_index in kf_split:
            X_train = X_data.iloc[train_index]
            y_train = y_data.iloc[train_index]

            X_val = X_data.iloc[test_index]
            y_val = y_data.iloc[test_index]

            gbm = lgb.LGBMRegressor(num_leaves=param_dist['num_leaves'],
                                    subsample=param_dist['bagging_fraction'],
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
                    verbose=False
            )

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

    return pd.DataFrame({
        "rmse": rmse_list,
        "mae": mae_list,
        "map": map_list,
        "parameters": param_list,
        "best_iteration": best_iter_list
        })

def predict(X_data, y_data, X_pred, best_params):
    """
    Predicts the score of each movie
    """
    gbm = lgb.LGBMRegressor(num_leaves=best_params['num_leaves'],
                            subsample=best_params['bagging_fraction'],
                            colsample_bytree=best_params['colsample_bytree'],
                            learning_rate=best_params['learning_rate'],
                            boosting_type=best_params['boosting_type'],
                            max_depth=best_params['max_depth'],
                            reg_lambda=best_params['reg_lambda'],
                            n_estimators=1000,
                            n_jobs=-1
                            )

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data)

    gbm.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l2',
            early_stopping_rounds=5,
            verbose=0)

    # predicting
    y_pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration_)
    return y_pred, gbm