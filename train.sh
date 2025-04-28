# huggingface-cli upload-large-folder samitizerxu/openfwi-unpacked --private --num-workers 10 --repo-type dataset --no-bars .

# baseline
# python train.py -ds all -n invnet_all_baseline -m InversionNet -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 128

# cosine
python train.py -ds all -n invnet_moredata_cosine -m InversionNet -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 192 -r /home/ziggy/dev/openfwi/Invnet_models/invnet_moredata_cosine/checkpoint.pth

# skip conns
# python train.py -ds all -n invnet_moredata_cosine -m InversionNetSkip -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 192

# skip conns
# python train.py -ds all -n invnet_moredata_cosine -m InversionNetSkip -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 192

# python gan_train.py -ds all -n gan_net -m InversionNet -g2v 0 --tensorboard -t split_files/train_ds.csv -v split_files/val_ds.csv -b 192 -eb 40 -nb 3

python train.py -ds all -n invnet_3d_cosine_lr_1e-4_b192_gac -m InversionNet3D -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 64 --lr 0.0001 --grad-accum-steps 3
