## Target attack adversarial examples detection

```
project
|   data
|   |   data
|   |   |   {dataset}-{model_name}-{datetime}
|   |   |   |   {attack}
|   |   |   |   |   {segments}
|   target-fashion-mnist
|   |   src
|   |   |
|   readme.md
```

#### First of all, run derived model to build EM classifierss
`CIFAR10`:
```
python src/detect/derived_model.py --attack-method=org --dataset=cifar10 --batch-num=4 --date=20220408 --target=2 --rate=1 --gradcam=True --channels=22 --winsize=256
```
`FMNist`: Robust Model
```
python src/detect/derived_model.py --attack-method=org --dataset=fm --batch-num=4 --date=20220323 --robust=True --target=2 --rate=1 --gradcam=True --channels=22 --winsize=256
```
#### Generate cifar-10 model
```
python src/scripts/run_fmnist.py attack-cifar --model-name=cnn-cifar --attack-name=targetcw --verbose=True --save-example=True
```

#### CIFAR-10 EM classifiers 
```
python src/detect/derived_model.py --attack-method=pgd --dataset=cifar10 --victim-model=vgg --batch-num=5 --date=20210907 --channels=4 --winsize=512 --rate=0.5
```

#### Adversarial training model of FMNist
```
python src/scripts/run_fmnist.py adv-evaluate --model-name=cnn --attack-name=FGM --evaluate-attack-name=targetcw --verbose=True --save-example=True --target=3
```
`attack-name`: the attack used during adversarial training; `evaluate-attack-name`: the attack used in evaluating the robust model; `target`: is the attack is target attack, set a target


#### Adaptive attack
One of the methods of adaptive attack is add some noise to the figure -> the noise will affect the EM traces