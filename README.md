# ECFEND

Source code and datasets for the paper "Enhancing Fake News Detection by Incorporating Evidence Credibility".

## Requirements

We use Pytorch 1.9.1 and python 3.6. Other requirements are in requirements.txt.

```
pip install -r requirements.txt
```

## Data 

 We utilize two widely used datasets. 

* Snopes: http://resources.mpi-inf.mpg.de/impact/dl_cred_analysis/Snopes.zip
* PolitiFact: http://resources.mpi-inf.mpg.de/impact/dl_cred_analysis/PolitiFact.zip

## Usage

You can run the commands below to train and test our model on Snopes Dataset.

```
python MasterFC/ecfend.py --dataset="Snopes" \
                            --cuda=1 \
                            --fixed_length_left=30 \
                            --fixed_length_right=100 \
                            --log="logs/ecfend" \
                            --loss_type="cross_entropy" \
                            --batch_size=32 \
                            --num_folds=5 \
                            --use_claim_source=0 \
                            --use_article_source=1 \
                            --path="formatted_data/declare/" \
                            --hidden_size=300 \
                            --epochs=100 \
                            --num_att_heads_for_words=5 \
                            --num_att_heads_for_evds=2 \
                            --gnn_window_size=3 \
                            --lr=0.0001 \
                            --gnn_dropout=0.2 \
                            --seed=34480 \
                            --gsl_rate=0.6 \
                            --iter_num=4
```

You can also simply run the bash script.

```
sh run_snopes.sh
```
or
``` 
sh run_politifact.sh (on the PolitiFact dataset)
```
or
``` 
sh run_snopesCG.sh (on the SnopesCG dataset)
```
