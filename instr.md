python -m pip install -r requirements.txt
huggingface-cli download --repo-type dataset --local-dir /workspace/ds --max-workers 40 --force-download samitizerxu/openfwi
python split_all_data.py
python gen_ds.py
python check_files.py
python split_dataset.py