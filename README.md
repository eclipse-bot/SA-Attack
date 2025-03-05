# SA-Attack
2024 IEEE Intelligent Vehicles Symposium(IV) SA-Attack: Speed-adaptive stealthy adversarial attack on trajectory prediction

## Requirements

* Python 3.8+

Install necessary packages.
```
pip install -r requirements.txt
```
## Prepare datasets
place datasets in directory `/data` following `README.md` in `data/apolloscape`, `data/NGSIM`, and `data/nuScenes`.
translate raw dataset into JSON-format testing data.

## Prepare models
The models should be placed in `/data/${model_name}_${dataset_name}/model/${attack_mode}`.

## Run normal prediction as well as the SA-Attack
```
python SA-Attack.py --help
```
