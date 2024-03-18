# GFN
Pytorch Implementation of



# Overview


<h2 align="center">
<figure> <img src="topology.jpg" height="300"></figure>
</h2>

Illustration of GFN.

# Dataset
YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).

Run `python src/data_process.py` to pre-process the data.

# Dependencies
Please set up the environment following Requirements in requirements.txt.
```sh
argparse          1.1.0
networkx          1.11
numpy             1.16.4
scikit_learn      0.21rc2
scipy             1.2.1
torch             1.4.0
```

# Reproduce
```sh
python main.py --config ./config/yelpchi.yml
```

# Acknowledgement
Our code references:
- [CAREGNN](https://github.com/YingtongDou/CARE-GNN)

- [PCGNN](https://github.com/PonderLY/PC-GNN)