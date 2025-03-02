# Research Notes

## 3.2
### git push 失败
你遇到的`The agent has no identities.`消息表明当前SSH代理没有加载任何私钥。这意味着即使你的`~/.ssh/config`文件配置正确，Git操作也无法使用指定的私钥进行身份验证。下面是解决这个问题的具体步骤：

#### 1. 添加私钥到SSH代理

首先，你需要将你的私钥（例如`id_rsa_jamieyi2004`）添加到SSH代理中。执行以下命令：

```bash
ssh-add ~/.ssh/id_rsa_jamieyi2004
```

如果这个命令成功运行但没有任何输出，通常意味着私钥已经被成功添加。你可以再次运行`ssh-add -l`来确认私钥是否已经加载：

```bash
ssh-add -l
```

你应该能看到类似如下的输出，显示已添加的密钥及其指纹：

```plaintext
2048 SHA256:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx /home/user/.ssh/id_rsa_jamieyi2004 (RSA)
```

#### 2. 启动SSH代理

如果你之前从未启动过SSH代理或者它未在后台运行，可能需要手动启动它。可以使用以下命令来启动SSH代理并设置环境变量：

```bash
eval "$(ssh-agent -s)"
```

然后重复上述步骤，尝试添加你的私钥。

#### 3. 检查SSH代理是否被其他程序管理

有时候，某些桌面环境或开发工具会自动管理SSH代理，这种情况下直接使用`ssh-add`可能不会生效。如果你怀疑这种情况，请检查是否有特定于你的环境或IDE的SSH配置，并根据需要调整。

#### 4. 测试SSH连接

一旦确保私钥已被正确添加到SSH代理，尝试测试与GitHub的连接：

```bash
ssh -T git@github.com
```

正常情况下，你应该看到一条欢迎信息，表明你已经成功通过身份验证。

#### 5. 再次尝试Git操作

完成以上步骤后，回到你的Git仓库目录下，再次尝试执行`git push`或其他Git操作。

如果仍然遇到问题，请提供详细的错误信息或调试日志（例如通过`ssh -vT git@github.com`获取的详细输出），以便进一步分析和解决问题。

### 命令
```bash
tmux new -s <session-name>
tmux attach-session -t <session-name>

nvidia-smi

python dataset_tool.py --source=./data/pokemon --dest=./data/pokemon256.zip --resolution=256x256 --transform=center-crop

python train.py --outdir=./training-runs/ --cfg=fastgan --data=./data/pokemon256.zip --gpus=8 --batch=64 --mirror=1 --snap=50 --batch-gpu=8 --kimg=10000

```
