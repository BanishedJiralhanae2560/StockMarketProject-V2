# download_data.py — RETIRED
#
# This script was previously used to download a static Apple CSV for training.
# The system now fetches all training data live from the Polygon API in
# train_model.py via collect_bulk_api_data(). This file is no longer needed
# and should not be run.
#
# If you ever need to pre-download data for offline training, extend
# collect_bulk_api_data() in train_model.py to save CSVs to the data/ folder.

print("download_data.py is retired. Run train_model.py directly to fetch and train.")