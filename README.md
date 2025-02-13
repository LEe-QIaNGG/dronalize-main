conda env create -f F:\workplace\cursorProjects\dronalize-main\build\environment.yml
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

python -m preprocessing\preprocess_highway.py --config 'highD' --path "F:\workplace\cursorProjects\IRL+MAMBA\data"
preprocessing\preprocess_highway.py
pip install mamba-ssm









如果你想通过 `pip install .` 从 GitHub 仓库安装 Python 项目（即从源码安装），你可以按照以下步骤操作：

### 1. **克隆 GitHub 仓库到本地**
首先，你需要将 GitHub 仓库克隆到你的本地机器。你可以使用以下命令：

```bash
git clone https://github.com/username/repository.git
```
请将 `username/repository` 替换为实际的 GitHub 用户名和仓库名称。

### 2. **进入仓库目录**
进入你刚才克隆的仓库目录：

```bash
cd repository
```

### 3. **使用 `pip install .` 安装**
确保你在仓库的根目录下（即包含 `setup.py` 文件的目录）。然后使用以下命令来从源码安装：

```bash
pip install .
```

这里的 `.` 表示当前目录，`pip` 会在该目录下查找 `setup.py` 文件并进行安装。`setup.py` 文件是 Python 项目的安装脚本，包含了项目的依赖、安装要求等信息。

### 4. **安装开发模式（可选）**
如果你希望在安装后能够直接修改源代码并立刻生效，可以使用开发模式进行安装：

```bash
pip install -e .
```

`-e`（或 `--editable`）标志表示将仓库安装为“可编辑模式”，这意味着你对源码的任何修改都会立即影响到安装的包，而不需要重新安装。

### 5. **确认安装**
安装完成后，你可以使用以下命令确认包是否安装成功：

```bash
pip show package-name
```

将 `package-name` 替换为你安装的包的名称。它会显示该包的详细信息，包括安装路径、版本等。

### 可能遇到的问题：
- **缺少依赖：** 如果 `setup.py` 中列出了依赖包，但这些包没有自动安装，你可以手动安装缺失的依赖：
  ```bash
  pip install -r requirements.txt
  ```
  如果 `requirements.txt` 文件存在的话。
  
- **权限问题：** 如果你没有权限安装到全局环境，可以使用 `--user` 标志进行用户级安装：
  ```bash
  pip install --user .
  ```

这些步骤应该能够帮助你从 GitHub 仓库源码安装项目。如果遇到其他问题，请随时告诉我！