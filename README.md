conda env create -f F:\workplace\cursorProjects\dronalize-main\build\environment.yml
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

python -m preprocessing\preprocess_highway.py --config 'highD' --path "F:\workplace\cursorProjects\IRL+MAMBA\data"
preprocessing\preprocess_highway.py
pip install mamba-ssm









�������ͨ�� `pip install .` �� GitHub �ֿⰲװ Python ��Ŀ������Դ�밲װ��������԰������²��������

### 1. **��¡ GitHub �ֿ⵽����**
���ȣ�����Ҫ�� GitHub �ֿ��¡����ı��ػ����������ʹ���������

```bash
git clone https://github.com/username/repository.git
```
�뽫 `username/repository` �滻Ϊʵ�ʵ� GitHub �û����Ͳֿ����ơ�

### 2. **����ֿ�Ŀ¼**
������ղſ�¡�Ĳֿ�Ŀ¼��

```bash
cd repository
```

### 3. **ʹ�� `pip install .` ��װ**
ȷ�����ڲֿ�ĸ�Ŀ¼�£������� `setup.py` �ļ���Ŀ¼����Ȼ��ʹ��������������Դ�밲װ��

```bash
pip install .
```

����� `.` ��ʾ��ǰĿ¼��`pip` ���ڸ�Ŀ¼�²��� `setup.py` �ļ������а�װ��`setup.py` �ļ��� Python ��Ŀ�İ�װ�ű�����������Ŀ����������װҪ�����Ϣ��

### 4. **��װ����ģʽ����ѡ��**
�����ϣ���ڰ�װ���ܹ�ֱ���޸�Դ���벢������Ч������ʹ�ÿ���ģʽ���а�װ��

```bash
pip install -e .
```

`-e`���� `--editable`����־��ʾ���ֿⰲװΪ���ɱ༭ģʽ��������ζ�����Դ����κ��޸Ķ�������Ӱ�쵽��װ�İ���������Ҫ���°�װ��

### 5. **ȷ�ϰ�װ**
��װ��ɺ������ʹ����������ȷ�ϰ��Ƿ�װ�ɹ���

```bash
pip show package-name
```

�� `package-name` �滻Ϊ�㰲װ�İ������ơ�������ʾ�ð�����ϸ��Ϣ��������װ·�����汾�ȡ�

### �������������⣺
- **ȱ��������** ��� `setup.py` ���г���������������Щ��û���Զ���װ��������ֶ���װȱʧ��������
  ```bash
  pip install -r requirements.txt
  ```
  ��� `requirements.txt` �ļ����ڵĻ���
  
- **Ȩ�����⣺** �����û��Ȩ�ް�װ��ȫ�ֻ���������ʹ�� `--user` ��־�����û�����װ��
  ```bash
  pip install --user .
  ```

��Щ����Ӧ���ܹ�������� GitHub �ֿ�Դ�밲װ��Ŀ����������������⣬����ʱ�����ң�