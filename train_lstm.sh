python3 -u train_models.py  ../too_big_for_git/features/filtered_features_balanced_LLD ../too_big_for_git/models/ALC_lstm_default_params_balanced_data_LLD.pt LLD "{'lr': 0.0001, 'layers': [16, 8, 4], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10, 'bidirectional': False, 'bias': True, 'lstm_layers': 2}"

python3 -u train_models.py  ../too_big_for_git/features/filtered_features_balanced_LLD ../too_big_for_git/models/ALC_lstm_combo1_balanced_data_LLD.pt LLD "{'lr': 0.0001, 'layers': [8], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10, 'bidirectional': False, 'bias': True, 'lstm_layers': 2}"

python3 -u train_models.py  ../too_big_for_git/features/filtered_features_balanced_LLD ../too_big_for_git/models/ALC_lstm_combo2_balanced_data_LLD.pt LLD "{'lr': 0.0001, 'layers': [16, 8, 4], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10, 'bidirectional': True, 'bias': True, 'lstm_layers': 2}"

python3 -u train_models.py  ../too_big_for_git/features/filtered_features_balanced_LLD ../too_big_for_git/models/ALC_lstm_combo3_balanced_data_LLD.pt LLD "{'lr': 0.0001, 'layers': [16, 8, 4], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10, 'bidirectional': False, 'bias': False, 'lstm_layers': 2}"

python3 -u train_models.py  ../too_big_for_git/features/filtered_features_balanced_LLD ../too_big_for_git/models/ALC_lstm_combo4_balanced_data_LLD.pt LLD "{'lr': 0.0001, 'layers': [16, 8, 4], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 10, 'num_epochs': 10, 'bidirectional': False, 'bias': True, 'lstm_layers': 2}"

python3 -u train_models.py  ../too_big_for_git/features/filtered_features_balanced_LLD ../too_big_for_git/models/ALC_lstm_combo5_balanced_data_LLD.pt LLD "{'lr': 0.0001, 'layers': [16, 8, 4], 'dropout': True, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10, 'bidirectional': False, 'bias': True, 'lstm_layers': 2}"