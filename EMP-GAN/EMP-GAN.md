# EMP-GAN
### 指导老师：杨秀隆
### 学生：易嘉鸣
## 1. 模型结构
在`ProjectedGAN`模型的基础上修改。
喂给判别器`patch_num`个增强后的图像，增强用的是`StyleGAN2ada`的代码。   
没有使用`EMP-SSL`的特征提取网络，直接用了`ProjectedGAN`的。  
使用了`EMP-SSL`的投影器`projector`，将`ProjectedGAN`的特征提取网络输出的最抽象的那个特征图展平后作为`projector`的输入，`projector`的输出用于计算`EMP loss`。    
修改后的网络结构如下图(b)，红色的部分是相较于`Projected GAN`增加的结构:
![empgan](image.png)

## 2. 实验结果
### 2.1 复现实验 vs. 论文数据
首先关于PG的复现结果，实际跑的效果和论文的数据有出入，暂时没有去找原因
|Dataset     | kimg | Paper | Exp  | 是否达到论文效果   |
|---         |---   |---    |---   |---               |
|Pokemon     | 0.3M | 36.57 |45.36 |No                |
|Pokemon     | 0.8M | 27.37 |36.15 |No                |
|Art painting| 0.2M | 40.22 |39.38 |Yes               |
|Art painting| 0.6M | 14.99 |20.74 |No                |

### 2.2 EMP-GAN
以下表格是在Pokemon数据集上训练的结果。
总体来说，不好也不坏，在同样的资源（显存和GPU数量）和时间下，EMP-GAN相较ProjectedGAN速度没有较大的提升。

| Model                                                |  0.1M   | 0.2M       | 0.3M       | 0.4M       | 0.6M       | 0.8M         |
| ------                                               | ---     | ---        | ---        | ---        | ---        | ---          |
|PG,               batch=88,VRAM=17721MiB              |75.09/15m|59.33/31m   |46.51/48m   |40.17/1h 4m |41.61/1h 39m| 36.15/2h 12m | 
|EG,patch=4,prob=1,batch=64                            |51.62/24m|48.08/51m   |40.43/1h 17m|41.15/1h 44m|40.23/2h 37m| 41.80/3h 28m | 
|EG,patch=4,prob=2,batch=64                            |72.76/24m|50.20/51m   |51.43/1h 17m|52.59/1h 44m|52.42/2h 35m| 42.14/3h 26m | 
|EG,patch=4,prob=3,batch=64                            |69.08/24m|49.77/51m   |53.05/1h 17m|51.28/1h 44m|44.49/2h 35m| 46.02/3h 26m | 
|EG,patch=8,prob=1,batch=32,VRAM=16825MiB,w=10,len=16  |45.22/47m|37.32/1h 36m|37.85/2h 25m|            |            |              | 
|EG,patch=8,prob=1,batch=32,VRAM=16625MiB,w=10,len=16,C|52.91/50m|43.84/1h 44m|            |            |            |              | 
|EG,patch=8,prob=1,batch=32,VRAM=17325MiB,w=10,len=32  |49.81/54m|41.86/1h 52m|            |            |            |              | 
|EG,patch=8,prob=1,batch=32,VRAM=16625MiB,w=10,len=32,C|53.91/51m|40.84/1h 42m|40.22/2h 33m|37.29/3h 23m|35.15/4h 16m|32.36/6h 58m  | 
|EG,patch=8,prob=1,batch=32,VRAM=17325MiB,w=10,len=128 |145.2/47m|115.8/1h 36m|            |            |            |              | 
|EG,patch=8,prob=1,batch=32,VRAM=17325MiB,w=5,len=128  |57.22/49m|78.02/1h 40m|69.73/2h 31m|            |            |              | 
注释：
- PG：Projected GAN
- EG：EMP GAN
- batch：batch_size
- patch：patch_num，图像增强后的数量
- prob：图像增强的强度
- VRAM：显存
- w：损失函数的权重，loss = w * d_loss + emp_loss
- len：projector的输出的特征的长度 
- C：在projector里加了一个1*1的卷积降维，以降低计算量，效果貌似并不理想
- 表格中的数据是：FID/time


### 2.3 实验小感悟
- 目前唯一能确定的参数是——图像增强的强度，还是调小点好，太大的话模型不能收敛。
- 勤打注释，及时对变量的类型、张量的形状标注，防止搞晕
- 训练有很强的不稳定性，同样的参数，每次结果都不太一样
- 中途装了一些包导致模型跑不起来，以后谨慎安装
- 
## 3. 未来的工作
- 继续调整模型结构，如projector的结构，projector的位置
- 继续调整参数，如patch_num
- 在其他数据集上试试
