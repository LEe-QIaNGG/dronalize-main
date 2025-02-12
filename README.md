conda env create -f F:\workplace\cursorProjects\dronalize-main\build\environment.yml
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

python -m preprocessing\preprocess_highway.py --config 'highD' --path "F:\workplace\cursorProjects\IRL+MAMBA\data"
preprocessing\preprocess_highway.py