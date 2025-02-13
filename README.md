conda env create -f F:\workplace\cursorProjects\dronalize-main\build\environment.yml
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

python -m preprocessing\preprocess_highway.py --config 'highD' --path "F:\workplace\cursorProjects\IRL+MAMBA\data"
preprocessing\preprocess_highway.py
pip install mamba-ssm
pip install .


HeteroDataBatch(
  rec_id=[32],
  agent={
    num_nodes=545,
    ta_index=[32],
    ids=[32],
    type=[545],
    inp_pos=[545, 10, 2],
    inp_vel=[545, 10, 2],
    inp_acc=[545, 10, 2],
    inp_yaw=[545, 10, 1],
    trg_pos=[545, 25, 2],
    trg_vel=[545, 25, 2],
    trg_acc=[545, 25, 2],
    trg_yaw=[545, 25, 1],
    intention=[545],
    input_mask=[545, 10],
    valid_mask=[545, 25],
    sa_mask=[545, 25],
    ma_mask=[545, 25],
    batch=[545],
    ptr=[33],
  },
  map_point={
    num_nodes=5702,
    type=[5702],
    position=[5702, 2],
    batch=[5702],
    ptr=[33],
  },
  (map_point, to, map_point)={
    edge_index=[2, 11152],
    type=[11152, 1],
  }
)


