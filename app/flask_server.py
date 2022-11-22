from flask import Flask, request, jsonify

from sklift import datasets
import pandas as pd
import numpy as np
import os

import pickle

from sklift.metrics import uplift_at_k
from sklift.viz import plot_uplift_preds
from sklift.models import SoloModel
from catboost import CatBoostClassifier


FEATURES_FOLDER = '/app_docker/app/features/'
MODEL_FOLDER = '/app_docker/app/model/'

class DataLoadingTrasform:
    """Data preprocessing"""

    limit_age_min = 12
    limit_age_max = 90


    cols = ['dif_between_transactions_of_client',
            'regular_points_received',
            'express_points_received',
            'regular_points_spent',
            'express_points_spent',
            'purchase_sum']

    mean_cols = ['mean_time_between_transactions_client',
                 'mean_regular_points_received_client',
                 'mean_express_points_received_client',
                 'mean_regular_spent_client',
                 'mean_express_spent_client',
                 'mean_purchase_sum_client']

    dropped_cols = ['transaction_id',
                    'transaction_datetime',
                    'store_id',
                    'product_id',
                    'product_quantity',
                    'trn_sum_from_iss',
                    'trn_sum_from_red']

    def __init__(self, dict_with_values):

        self.mode_client_store = dict_with_values['mode_client_store']
        self.med_ages_in_stores = dict_with_values['med_ages_in_stores']
        self.min_client_transaction_datetime = dict_with_values['min_client_transaction_datetime']
        self.mean_target_store = dict_with_values['mean_target_store']
        self.mean_value_along_cols_dict = dict_with_values['mean_value_along_cols_dict']


        # self.mode_client_store = None
        # self.med_ages_in_stores = None
        # self.min_client_transaction_datetime = None
        # self.mean_target_store = None
        # self.mean_value_along_cols_dict = None

    def transform_features(self, df_clients, df_purchases, df_out):
        df_clients_purch = df_clients.merge(df_purchases, how='inner', on='client_id')

        # Transform min and max ages of clients
        df_clients_purch = df_clients_purch.merge(self.mode_client_store, 
                                                  how='inner', 
                                                  on='client_id')
        
        df_clients_purch = df_clients_purch.merge(self.med_ages_in_stores, 
                                                  how='inner', 
                                                  on='store_id')

        filtr_min_age = df_clients_purch['age'] < self.limit_age_min
        filtr_max_age = df_clients_purch['age'] > self.limit_age_max

        df_clients_purch.loc[filtr_min_age | filtr_max_age,
                            'age'] = \
                             df_clients_purch.loc[filtr_min_age | filtr_max_age,
                                                'median_client_ages_in_store']

        # Add mean target points to mode store of client
        df_clients_purch = df_clients_purch.merge(self.mean_target_store,
                                                  how='inner',
                                                  on='mode_client_store')

        #Compute difference between first_redem and first_issue data

        df_clients_purch = \
             df_clients_purch.merge(self.min_client_transaction_datetime, 
                                    how='inner', 
                                    on='client_id')

        df_clients_purch['first_redeem_date'].fillna(df_clients_purch['min_transaction_datetime'],
                                                     inplace=True)
        df_clients_purch['dif_first_issue_and_redeem_date_in_h'] = \
            (df_clients_purch['first_redeem_date'] - \
             df_clients_purch['first_issue_date']).astype('timedelta64[h]')

        filtr = df_clients_purch['dif_first_issue_and_redeem_date_in_h'] < 0
        df_clients_purch.loc[filtr,
                    'dif_first_issue_and_redeem_date_in_h'] = 0
        df_clients_purch.drop(columns={'median_client_ages_in_store',
                                       'first_issue_date',
                                       'first_redeem_date',
                                       'min_transaction_datetime'},
                              inplace=True)
        


        #Merge columns with mean values
        for col, mean_col in zip(self.cols, self.mean_cols):

            df_clients_purch = df_clients_purch.merge(self.mean_value_along_cols_dict[mean_col], 
                                                      how='inner', 
                                                      on='client_id')
            
            if col in df_clients_purch.columns:
                df_clients_purch.drop(columns=col,
                                      inplace=True)

        df_clients_purch.drop(columns=self.dropped_cols,
                              inplace=True)

        #Merge columns with mean values
        df_out = df_out.merge(df_clients_purch,
                              how='inner',
                              on='client_id')
        df_out = df_out.drop_duplicates(subset=['client_id'],
                                        ignore_index = True)

        return df_out

    def reduce_mem_usage(self, df, cols_for_memory_reduction, type_names):
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.        
        """
        # start_mem = df.memory_usage().sum() / 1024**2
        # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        
        for col, type_name in zip(cols_for_memory_reduction, type_names):
            # print(col)
            if type_name == 'object':
                pass
            elif type_name == 'datetime':
                df[col] = pd.to_datetime(df[col])
            else:
                df[col] = df[col].astype(type_name)
        
        # end_mem = df.memory_usage().sum() / 1024**2
        # red_mem = (100 * (start_mem - end_mem) / start_mem)
        # print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        # print(f'Decreased by {red_mem:.1f}%')

        return df

    def _calc_mode_func_series(self, x):
        x = pd.Series.mode(x)[0]
        return x


#Loading data for features and model
with open(FEATURES_FOLDER + 'saved_datapreprocessing_fields_old.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

data_loading_transform = DataLoadingTrasform(loaded_dict)

sm = pickle.load(open(MODEL_FOLDER + 'solo_model_old.sav', 'rb'))


# Flask app
app = Flask(__name__) #__name__ module name

# @app.route('/', methods=['GET'])
@app.route('/')
def index():
    return 'Machine Learning API'

@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}

    request_json = request.get_json()

    if request_json['df_clients']:
        df_clients = request_json['df_clients']
        df_clients = pd.DataFrame.from_dict(df_clients,
                                            orient='columns')
        df_out = df_clients['client_id'].to_frame()
    else:
        return 'No data about clients'
    
    if request_json['df_purch']:
        df_purch = request_json['df_purch']
        df_purch = pd.DataFrame.from_dict(df_purch,
                                        orient='columns')
    else:
        return 'No data about purchases of clients'
    
    df_cols = df_clients.columns
    df_types = ['object',
                'datetime',
                'datetime',
                np.uint8,
                'category']

    df_clients = \
        data_loading_transform.reduce_mem_usage(df_clients,
                                                df_cols,
                                                df_types)

    df_cols = ['transaction_datetime',
            'regular_points_received',
            'express_points_received',
            'regular_points_spent',
            'express_points_spent',
            'purchase_sum',
            'product_quantity',
            'trn_sum_from_iss']
    df_types = ['datetime',
                np.float16,
                np.float16,
                np.float16,
                np.float16,
                np.uint16,
                np.uint8,
                np.uint8,
                np.float16]

    df_purch['regular_points_spent'] = \
        df_purch['regular_points_spent'].apply(np.abs)

    df_purch['express_points_spent'] = \
        df_purch['express_points_spent'].apply(np.abs)

    df_purch = \
    data_loading_transform.reduce_mem_usage(df_purch,
                                            df_cols,
                                            df_types)

    # Transform features
    df_test_transformed = \
        data_loading_transform.transform_features(df_clients,
                                                df_purch,
                                                df_out)

    X_data = df_test_transformed.drop(columns=['client_id'])
    #Compute predictions
    y_preds = sm.predict(X_data)

    y_preds_df = pd.DataFrame(y_preds,
                            index=df_test_transformed['client_id'],
                            columns=['target_preds'])
    y_preds_dict = y_preds_df.to_dict()

    data['success'] = True
    print('Predictions have been done!')
    data['predictions'] = y_preds_dict

    return jsonify(data)


if __name__ == '__main__':
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)