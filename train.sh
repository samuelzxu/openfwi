# baseline
# python train.py -ds all -n invnet_all_baseline -m InversionNet -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 128

# cosine
# python train.py -ds all -n invnet_moredata_cosine -m InversionNet -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 192

# skip conns
# python train.py -ds all -n invnet_moredata_cosine -m InversionNetSkip -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 192

# skip conns
# python train.py -ds all -n invnet_moredata_cosine -m InversionNetSkip -g2v 0 --tensorboard -t train_ds.csv -v val_ds.csv -b 192

python gan_train.py -ds all -n gan_net -m InversionNet -g2v 0 --tensorboard -t split_files/train_ds.csv -v split_files/val_ds.csv -b 192 -eb 40 -nb 3
