python3 -u train_models.py ../too_big_for_git/features/filtered_features_balanced_Functional ../too_big_for_git/models/ALC_default_params_balanced_data_Functional.pt Functional "{'lr': 0.0001, 'layers': [64, 32, 16, 8, 4], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10}"

python3 -u train_models.py ../too_big_for_git/features/filtered_features_balanced_Functional ../too_big_for_git/models/ALC_combo1_balanced_data_Functional.pt Functional "{'lr': 0.0001, 'layers': [32], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10}"

python3 -u train_models.py ../too_big_for_git/features/filtered_features_balanced_Functional ../too_big_for_git/models/ALC_combo2_balanced_data_Functional.pt Functional "{'lr': 0.0001, 'layers': [64, 16, 4], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10}"

python3 -u train_models.py ../too_big_for_git/features/filtered_features_balanced_Functional ../too_big_for_git/models/ALC_combo3_balanced_data_Functional.pt Functional "{'lr': 0.0001, 'layers': [64, 32, 16, 8, 4], 'dropout': False, 'optim': 'adam', 'bn': True, 'batch_size': 10, 'num_epochs': 10}"

python3 -u train_models.py ../too_big_for_git/features/filtered_features_balanced_Functional ../too_big_for_git/models/ALC_combo4_balanced_data_Functional.pt Functional "{'lr': 0.0001, 'layers': [64, 32, 16, 8, 4], 'dropout': True, 'optim': 'adam', 'bn': True, 'batch_size': 5, 'num_epochs': 10}"