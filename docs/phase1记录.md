

# PHASE 1 (waist 6) 



## STEP 01：只测试sub1（epoch=5）



哈哈哈，这不是模型“三类都不会”，而是它学成了一个**过度敏感的跌倒预警器**：

> Near-Fall 和 Fall 几乎都能抓住，但把大量正常 ADL 错报成 Near-Fall。

而且这只是 `sub1` 的60条测试数据，不是完整10折 LOSO 的最终结果。

#### 先读懂混淆矩阵

纵轴是真实类别，横轴是预测类别；对角线才是预测正确的。

| 真实类别          | 正确 | 错误情况                 | Recall |
| ----------------- | ---- | ------------------------ | ------ |
| ADL，共24条       | 5    | 16条→Near-Fall，3条→Fall | 20.83% |
| Near-Fall，共15条 | 14   | 1条→Fall                 | 93.33% |
| Fall，共21条      | 20   | 1条→Near-Fall            | 95.24% |

所以你的判断完全正确：

- ADL只识别对了5条；
- ADL错误率为 (19/24=79.17%)；
- 最严重的是16条ADL被误报成Near-Fall；
- 真正的Fall只漏了1条。

#### Accuracy 是什么？

Accuracy 就是全部样本中预测正确的比例：

[
Accuracy=\frac{5+14+20}{60}=0.65
]

也就是60条试验中39条正确。

它的缺点是：每条样本权重相同，因此样本较多的类别影响更大。这里ADL数量最多，却识别得最差，所以Accuracy被明显拉低。

#### Recall 是什么？

某一类的 Recall 回答的是：

> 所有真实属于这一类的样本，模型成功找回了多少？

例如：

[
Recall_{Fall}=\frac{20}{21}=95.24%
]

这意味着21次真实跌倒中，模型识别出了20次。

但高Recall不等于低误报。例如Near-Fall：

- 15条真实Near-Fall抓住了14条，Recall很高；
- 模型一共预测了31次Near-Fall；
- 其中只有14次是真的，另外17次是误报。

因此Near-Fall虽然Recall为93.33%，但预测可信度并不高。

#### Precision 和 F1

Precision回答的是：

> 模型预测成某一类的样本中，有多少真的属于这一类？

从矩阵可以算出：

| 类别      | Precision | Recall | F1     |
| --------- | --------- | ------ | ------ |
| ADL       | 100.00%   | 20.83% | 34.48% |
| Near-Fall | 45.16%    | 93.33% | 60.87% |
| Fall      | 83.33%    | 95.24% | 88.89% |

ADL的Precision看起来是100%，原因很滑稽：模型总共只预测了5次ADL，而这5次恰好都正确。它不是很会识别ADL，而是**几乎不敢预测ADL**。

F1是Precision与Recall的调和平均：

[
F1=2\frac{Precision\times Recall}{Precision+Recall}
]

只有Precision和Recall都比较好，F1才会高。

#### Macro-F1 是什么？

Macro-F1先分别计算三类F1，然后不考虑样本数量，直接平均：

[
Macro\text{-}F1
=\frac{0.3448+0.6087+0.8889}{3}
=0.6141
]

所以它比Accuracy更能暴露“某一类几乎失效”的问题。虽然Fall表现很好，但不能掩盖ADL的F1只有0.3448。

#### Balanced Accuracy 是什么？

Balanced Accuracy就是三类Recall的平均：

[
\frac{0.2083+0.9333+0.9524}{3}
=0.6980
]

它比普通Accuracy更公平地对待每个类别。

它比Macro-F1高，是因为它只看每类有没有被找回来；Macro-F1还会通过Precision惩罚大量误报，尤其是模型把31条数据预测成Near-Fall、却只有14条正确。

#### 为什么ADL会这么差？

目前只能提出待验证的原因，不能仅凭一张矩阵就下最终结论。最可能有以下几项。

##### 1. 这是5个epoch的最小测试

如果你按照之前的调试命令用了：

```text
--epochs 5
```

那么CNN可能只完成了很早期的学习。查看 `metrics.json` 中的 `epochs_ran`；如果就是5，这个结果不能代表模型收敛后的性能。

##### 2. 类别权重推动模型偏向Near-Fall

脚本默认使用了平衡类别权重。训练集中每名受试者有：

```text
ADL       24
Near-Fall 15
Fall      21
```

对应权重大约是：

```text
ADL        0.833
Near-Fall  1.333
Fall       0.952
```

也就是说，训练时漏掉一个Near-Fall的惩罚，大约是漏掉一个ADL的1.6倍。在训练不充分时，这可能让模型宁可把模糊样本判成Near-Fall。

这不是代码错误，但以后值得把“使用类别权重”和“不使用类别权重”作为对照实验。

##### 3. ADL本身非常杂

ADL不是一种动作，而是八种差异很大的活动：

- 正常行走；
- 上下楼梯；
- 坐下、起立和躺下；
- 捡东西；
- 静止站立。

其中坐下、躺下、弯腰捡东西等动作本身就可能包含明显的加速度和身体姿态变化，与Near-Fall的“失衡后恢复”有局部相似性。

##### 4. 这是未见受试者测试

模型没有见过 `sub1`。不同人的：

- 运动速度；
- 身材；
- 动作幅度；
- IMU佩戴角度；
- 腰部传感器朝向

都可能不同。我们直接输入原始XYZ轴，因此传感器方向差异尤其可能影响主体外泛化。

##### 5. 整段重采样可能丢失信息

当前模板把每个trial整体插值为1024点，因此：

- 原始动作持续时间被归一化；
- 关键失衡事件在序列中的位置不固定；
- `GlobalAveragePooling1D` 又会把整段时间特征平均。

这适合做第一版baseline，但可能模糊“正常转身/坐下”和“短暂失衡”的区别。

#### 现在最合理的下一步

首先用 `predictions.csv` 查明究竟是哪几种ADL被误判：

```python
import pandas as pd

p = pd.read_csv(r"D:\你的结果目录\test_sub1\predictions.csv")

print(pd.crosstab(
    [p["true_label"], p["scenario"]],
    p["predicted_label"],
    margins=True
))
```

如果主要是 `DSS / DSL / POG` 被误判，说明正常的姿态转换确实与Near-Fall相似；如果连 `SQ` 静止站立都大量错判，就更可能是训练不足、佩戴方向差异或标准化问题。

然后：

1. 查看 `history.csv` 的训练和验证损失；
2. 用100 epochs和Early Stopping重新跑一次单折；
3. 暂时不要根据 `sub1` 测试结果反复调参；
4. 管线稳定后运行完整10折LOSO；
5. 完整矩阵应该包含600个trial，而不是现在的60个。

因此，这次65%不能视为最终失败。它至少证明模型已学到Fall/Near-Fall信号，但产生了非常明确的ADL误报问题——接下来真正需要研究的就是：**哪些ADL被误报，以及误报来自训练不足、类别权重、主体差异，还是整段表示方式。**





### 测试sub1的结果分析



看到了。这不是原始 `predictions.csv`，而是它按“真实类别 + 动作类型”汇总后的交叉表：每行三个数字表示该动作的 3 次 trial 分别被模型判成什么。

结论很清楚：模型学到的是“运动/失衡强度”，还没有学会“正常活动 vs 近跌倒”的真正边界。

| 真实 ADL 动作                  | 模型输出                | 含义                                   |
| ------------------------------ | ----------------------- | -------------------------------------- |
| `SQ`                           | 3/3 判为 ADL            | 静止、低运动强度的正常活动能识别       |
| `RSS`                          | 2 ADL、1 Near-Fall      | 还算可以                               |
| `AS`、`DS`、`DSS`、`NW`、`POG` | 全部 3/3 判为 Near-Fall | 只要是明显动态的正常活动，几乎都被误报 |
| `DSL`                          | 全部 3/3 判为 Fall      | 这个正常动作的姿态/下降变化尤其像跌倒  |

而异常类几乎都很好：

- Fall：20/21 正确；只有 1 个 `slip` 被判成 Near-Fall。
- Near-Fall：14/15all。
- 正确；只有 1 个 `trip` 被判成 Fall。

所以这不是随机乱猜。它稳定地形成了这样一个粗糙规则：

> 静止/弱动作 → ADL
> 一般明显运动 → Near-Fall
> 强烈下落或姿态变化 → Fall

这也解释了为什么 ADL Recall 只有 20.8%：模型只敢把 `SQ` 和大部分 `RSS` 判成 ADL。

最可能的原因有三层：

1. 这次是 `--epochs 5` 的最小调试训练，模型很可能还没收敛；
2. 代码默认启用了类别权重。训练中 Near-Fall 的漏检惩罚约为 ADL 的 1.6 倍，早期训练时模型更倾向于“宁可报 Near-Fall”；
3. 整段重采样 + `GlobalAveragePooling1D` 容易把“动作发生在哪里、如何恢复平衡”压扁成整体运动强度。正常走路、上下楼、弯腰捡物本身就可能有较大腰部加速度。

这张图本身很有价值：它已经明确指出 Phase 1 的核心难点不是 Fall，而是**动态 ADL 的误报控制**。

下一步先不要改网络结构。按原定基线，用 100 epochs + Early Stopping 重跑这一折，看看验证损失是否仍在下降：

```powershell
python phase1_cnn_loso_commented.py 
  --data-root "D:\00-FLagship Project\\SFU-IMU Dataset\\IMU Dataset" 
  --output-dir "D:\00-Flagship Project\results\sub1_100epochs"
  --test-subject sub1 
  --sequence-length 1024
  --epochs 100
```

然后把新结果的 `history.csv` 最后十行、以及新的这张动作交叉表发我。之后再做一个“不使用类别权重”的诊断对照，判断 ADL 大量误报究竟主要来自训练不足，还是类别权重与整段表示方式。

 



## STEP 02：只测试sub1（epoch = 100）

这次是真的漂亮。`sub1` 这个未见受试者上，60 条里只错了 2 条：

- 1 条 ADL → Near-Fall
- 1 条 Near-Fall → Fall
- 21/21 条 Fall 全部识别正确

所以：

[
\text{Accuracy}=\frac{58}{60}=96.67%
]

而且三类并没有被某一类“虚高”掩盖：

| 指标              | 结果   | 含义                        |
| ----------------- | ------ | --------------------------- |
| Macro-F1          | 96.29% | 三类综合且兼顾误报后的表现  |
| Balanced Accuracy | 96.39% | 三类 Recall 的平均          |
| ADL Recall        | 95.83% | 24 条正常活动中识别对 23 条 |
| Near-Fall Recall  | 93.33% | 15 条近跌倒中识别对 14 条   |
| Fall Recall       | 100%   | 21 条跌倒全部识别对         |

这也基本确认：之前 5 epoch 的 `65%` 主要是**训练远远不够**，并不是路径、标签、标准化或 LOSO 划分出了问题。

唯一要严格区分的是：终端里的“最终 pooled held-out 指标”目前其实只是 `sub1` 这一折的 pooled 结果，不是完整 10 人 LOSO 的最终成绩。现在不要继续针对 `sub1` 调结构或刷参数，直接保持这套设置跑完整 LOSO：

```powershell
python phase1_cnn_loso_commented.py `
  --data-root "D:\00-Flagship Project\SFU-IMU Dataset" `
  --output-dir "D:\00-Flagship Project\results\phase1_cnn_100epochs" `
  --sequence-length 1024 `
  --epochs 100
```

完整跑完后，`fold_metrics.csv` 应有 10 行；那时的 `summary.json` 和 `loso_confusion_matrix*.png` 才是你可以写进项目报告的 Phase 1 CNN 基线结果。













## STEP 03：1D-CNN 100 epoch LOSO

### 1. 实验结果



#### 1.1 最终指标

最终 pooled held-out 指标：
                accuracy: 0.9483
                macro_f1: 0.9433
       balanced_accuracy: 0.9435
              recall_ADL: 0.9667
        recall_Near_Fall: 0.9067
             recall_Fall: 0.9571

#### 1.2 详细情况

| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 28   | 0    | 2         | 30   |
| ADL        | DSL      | 27   | 3    | 0         | 30   |
| ADL        | DSS      | 29   | 0    | 1         | 30   |
| ADL        | NW       | 29   | 0    | 1         | 30   |
| ADL        | POG      | 30   | 0    | 0         | 30   |
| ADL        | RSS      | 29   | 0    | 1         | 30   |
| ADL        | SQ       | 30   | 0    | 0         | 30   |
| Fall       | CS       | 0    | 27   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 0    | 30   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 0    | 26   | 4         | 30   |
| Fall       | trip     | 0    | 25   | 5         | 30   |
| Near_Fall  | CS       | 1    | 1    | 25        | 27   |
| Near_Fall  | HB       | 3    | 1    | 26        | 30   |
| Near_Fall  | ITCS     | 0    | 0    | 3         | 3    |
| Near_Fall  | ITRS     | 1    | 1    | 28        | 30   |
| Near_Fall  | slip     | 0    | 2    | 28        | 30   |
| Near_Fall  | trip     | 1    | 3    | 26        | 30   |
| All        |          | 238  | 212  | 150       | 600  |







### 2. 实验分析



对，row-normalized（每一行归一化）混淆矩阵的对角线，就是各真实类别的 Recall／召回率。

因为它在问：

> 真实是这一类的样本中，有多少被模型成功找出来？

例如 Near_Fall 这一行：

```text
真实 Near_Fall 共 150 个
预测为 ADL：6  = 4%
预测为 Near_Fall：136 = 90.67%
预测为 Fall：8 = 5.33%
```

所以对角线上的 `0.91` 就是：
$$
Recall_{NearFall}=\frac{136}{150}=0.9067
$$


你的解读是对的：

- ADL 的召回率最高：`232 / 240 = 96.67%`
- Fall：`201 / 210 = 95.71%`
- Near_Fall 最难：`136 / 150 = 90.67%`
  - 约 4% 被判成 ADL；
  - 约 5% 被判成 Fall；
  - 合计约 9.3% 识别错误。

而且这个错误结构是有意义的：Near-Fall 本来就处在“正常活动”和“真正跌倒”之间，因此它被推向两边是相当合理、也值得分析的现象，不是随机乱错。

最后的 10-fold LOSO 是这样总结的：

```text
第1折：sub1 从未参与训练 → 得到 sub1 的60条测试预测
第2折：sub2 从未参与训练 → 得到 sub2 的60条测试预测
...
第10折：sub10 从未参与训练 → 得到 sub10 的60条测试预测
                    ↓
把十折的 held-out predictions 合并
                    ↓
600 条“每条都来自未见受试者”的预测
                    ↓
统一计算最终混淆矩阵、accuracy、F1、recall
```

所以你这张最终矩阵不是某一折的结果，也不是把十折 accuracy 简单平均；而是把十折测试集预测拼成 `600` 条 pooled held-out predictions 后重新统计的。

你的数值完全能对上：
$$
accuracy=\frac{232+136+201}{600}
=\frac{569}{600}=0.9483
$$


`balanced_accuracy` 则是三个类别召回率的平均：


$$
balancedAccuracy=\frac{0.9667+0.9067+0.9571}{3}=0.9435
$$
这也解释了为什么它略低于 accuracy：Near_Fall 相对更难，balanced accuracy 不会让样本更多、识别更好的 ADL 把这个问题“冲淡”。

对 scenario 的观察也基本正确：

- Fall 中主要错误来自 `slip`（4 个判为 Near_Fall）和 `trip`（5 个判为 Near_Fall）；其余 Fall scenario 全对。
- 这说明模型没有把真正 Fall 误判成 ADL；Fall 的边界问题集中在“这次到底算跌倒，还是险跌倒”。
- Near_Fall 的错误分散在几乎所有动作类型：`CS`、`HB`、`ITRS`、`slip`、`trip` 都有少量错分。这意味着 Near-Fall 类内部的动作形态更杂，不只是被某一个特定 scenario 拖累。

这组结果已经相当不错：在严格按受试者隔离的 LOSO 下，三类 pooled macro-F1 `0.9433`，并且 Fall 的召回率 `0.9571`。下一步很适合写成一段正式实验结论，再加一个 “scenario-level error analysis” 小节。





### 3. 使用的CNN模型



```python
def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 1D CNN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 第一层用较大的 kernel=7 捕捉局部运动变化，再把时间长度减半。
    x = tf.keras.layers.Conv1D(32, 7, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # 第二层增加特征通道，继续提取更高层的局部时间模式。
    x = tf.keras.layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # 第三层后用全局平均池化，避免直接 Flatten 带来过多参数。
    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dropout 用于缓解这个小数据集上的过拟合。
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_1dcnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
```





之前说的 **45,507 个参数是正确的**。它是把三层卷积、三层 BatchNorm 和两层 Dense 的参数全部加起来：
$$
1344+128+10240+256+24576+512+8256+195
=\boxed{45507}
$$
下面逐层计算。默认输入是 `(1024, 6)`，类别数是 3。

#### 1）第一层 Conv1D

```python
Conv1D(32, kernel_size=7, use_bias=False)
```

每个卷积核：

- 时间宽度：7
- 输入通道：6
- 因此一个卷积核有 (7\times6=42) 个权重
- 一共有32个卷积核

所以：
$$
7\times6\times32=\boxed{1344}
$$
因为 `use_bias=False`，没有额外的32个偏置参数。

------

#### 2）第一层 BatchNormalization

卷积输出有32个通道。每个通道保存4个参数：

- (\gamma)：缩放参数，可训练
- (\beta)：平移参数，可训练
- moving mean：移动平均值，不可训练
- moving variance：移动方差，不可训练

所以：


$$
32\times4=\boxed{128}
$$
其中：

- 可训练参数：(32*2=64)
- 不可训练参数：(32*2=64)

------

#### 3）第二层 Conv1D

```python
Conv1D(64, kernel_size=5, use_bias=False)
```

这一层接收上一层产生的32个特征通道：

$$
5\times32\times64=\boxed{10240}
$$
含义是：

- 一个卷积核覆盖5个时间位置
- 每个位置有32个输入特征
- 因此每个卷积核有 (5\times32=160) 个权重
- 一共有64个卷积核

------

#### 4）第二层 BatchNormalization

这一层有64个输出通道：

$$
64\times4=\boxed{256}
$$
其中128个可训练，128个不可训练。

------

#### 5）第三层 Conv1D

```python
Conv1D(128, kernel_size=3, use_bias=False)
```

输入通道数是64，输出通道数是128：


$$
3\times64\times128=\boxed{24576}
$$



这是整个模型参数最多的一层。

------

#### 6）第三层 BatchNormalization

这一层有128个输出通道：

$$
128\times4=\boxed{512}
$$
其中256个可训练，256个不可训练。

------

#### 7）GlobalAveragePooling1D

```python
GlobalAveragePooling1D()
```

它只是把每个特征通道沿时间轴取平均：

```text
(256, 128) → (128,)
```

没有需要学习的权重：
$$
\boxed{0}
$$
ReLU、MaxPooling和Dropout也都没有参数。

------

#### 8）Dense(64)

```python
Dense(64, activation="relu")
```

输入有128个数字，输出有64个神经元。

每个输出神经元需要：

- 128个连接权重
- 1个偏置

所以：

$$
128\times64+64
=8192+64
=\boxed{8256}
$$



------

#### 9）最终 Dense(3)

```python
Dense(3, activation="softmax")
```

输入有64个数字，输出3个类别概率：



$$
64\times3+3
=192+3
=\boxed{195}
$$


------

#### 10）汇总

| 层                         | 参数量     |
| -------------------------- | ---------- |
| Conv1D：6 → 32，kernel=7   | 1,344      |
| BatchNorm(32)              | 128        |
| Conv1D：32 → 64，kernel=5  | 10,240     |
| BatchNorm(64)              | 256        |
| Conv1D：64 → 128，kernel=3 | 24,576     |
| BatchNorm(128)             | 512        |
| Dense：128 → 64            | 8,256      |
| Dense：64 → 3              | 195        |
| 总参数                     | **45,507** |

其中：

- **可训练参数：45,059**
- **不可训练参数：448**
- **总参数：45,507**

不可训练的448个参数就是三个 BatchNorm 保存的均值和方差：

$$
32\times2+64\times2+128\times2=448
$$
一个容易困惑的关键点是：输入有1024个时间点，但卷积层的参数量里没有1024。因为同一个卷积核会沿时间轴反复滑动，所有时间位置共享同一套权重。1024会影响计算量和中间数据大小，但不增加卷积核参数数量。











## STEP 04 LSTM 2layers 100 epoch LOSO



### 1. 实验结果



#### 1. 1 最终指标

最终 pooled held-out 指标：
                accuracy: 0.7250
                macro_f1: 0.7120
       balanced_accuracy: 0.7201
              recall_ADL: 0.6042
        recall_Near_Fall: 0.6133
             recall_Fall: 0.9429

#### 1. 2 详细情况

| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 22   | 0    | 8         | 30   |
| ADL        | DS       | 23   | 0    | 7         | 30   |
| ADL        | DSL      | 0    | 30   | 0         | 30   |
| ADL        | DSS      | 22   | 1    | 7         | 30   |
| ADL        | NW       | 17   | 0    | 13        | 30   |
| ADL        | POG      | 20   | 0    | 10        | 30   |
| ADL        | RSS      | 22   | 0    | 8         | 30   |
| ADL        | SQ       | 19   | 0    | 11        | 30   |
| Fall       | CS       | 1    | 26   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 0    | 30   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 4    | 24   | 2         | 30   |
| Fall       | trip     | 1    | 25   | 4         | 30   |
| Near_Fall  | CS       | 14   | 0    | 13        | 27   |
| Near_Fall  | HB       | 13   | 0    | 17        | 30   |
| Near_Fall  | ITCS     | 0    | 0    | 3         | 3    |
| Near_Fall  | ITRS     | 10   | 2    | 18        | 30   |
| Near_Fall  | slip     | 7    | 0    | 23        | 30   |
| Near_Fall  | trip     | 11   | 1    | 18        | 30   |
| All        |          | 206  | 232  | 162       | 600  |





### 2. 实验分析

你的解读基本准确，而且这个结果很有“可分析性”。

整体上，LSTM 不是随机地差，而是形成了很明确的错误模式：

| 真类别    | 召回率 | 主要错到哪里                    |
| --------- | ------ | ------------------------------- |
| ADL       | 60.4%  | 26.7% → Near Fall，12.9% → Fall |
| Near Fall | 61.3%  | 36.7% → ADL                     |
| Fall      | 94.3%  | 只有各 2.9% → ADL / Near Fall   |

所以核心不是 “Fall 和 Near Fall 混淆严重”。恰恰相反：

- **Fall 与 Near Fall 分得相当干净**：真实 Near Fall 只有 3/150 被报成 Fall。
- 真正的难点是 **ADL ↔ Near Fall**：
  - 64 个 ADL 被报成 Near Fall；
  - 55 个 Near Fall 被报成 ADL。
- 模型整体略偏向输出“风险动作”：真实非 ADL 共 360 个，但预测为非 ADL 的有 394 个。不过这不等于它在无差别地乱报 Fall；它对 Fall 的误报主要集中在一个非常异常的 ADL 子动作上。

最值得单独写的是 **DSL**：

> 30 个 DSL 全部被 LSTM 预测为 Fall。

这不是普通的“该动作较难”，而是模型对 DSL 建立了近乎确定的错误规则。去掉 DSL 后，其余 ADL 的正确数是 145/210，召回率会从 **60.4% 上升到约 69.0%**。所以 ADL 总体差，很大一部分是 DSL 这个单点灾难拉低的。

而 `NW / POG / SQ` 更常被判成 Near Fall，也很符合直觉：它们可能包含明显的姿态变化、转向、下蹲或动态加速度，和“失衡但未真正跌倒”的运动特征相邻。

这也支持一个很清楚的模型层面解释：

- 你的两层 LSTM 最终把整段 **1024 点**动作压缩成最后一个隐藏状态；
- 它比较容易抓住“整段动作是否剧烈、是否有明显姿态变化”；
- 但对 ADL 和 Near Fall 之间细微的局部时序差别，保留得不够好；
- 1D-CNN 用局部卷积核扫描冲击、震荡、短时加速度峰值，更适合这个数据集里的区分方式。

因此报告里可以很稳妥地写：

> Under subject-independent LOSO evaluation, the LSTM achieved 72.5% accuracy and 0.712 macro-F1, but underperformed the 1D-CNN baseline. The main error mode was mutual confusion between ADL and Near Fall, whereas Fall was detected with high sensitivity (94.3%). In particular, the DSL ADL scenario was consistently misclassified as Fall, suggesting that the LSTM’s global sequence representation did not sufficiently preserve the local temporal cues needed to distinguish certain dynamic daily activities from fall-like motion.

下一步最值得做的不是立刻继续堆 LSTM 层数，而是直接比较：**CNN 对 DSL 的识别率是多少**。如果 CNN 能正确识别大多数 DSL，这就是 CNN 局部特征优势的最有力证据；如果 CNN 也大量错 DSL，再回头检查该动作的原始波形、标签和传感器方向。



那这个对照非常有力。

- **LSTM：DSL 30/30 被判为 Fall**
- **1D-CNN：DSL 只有 3/30 被判为 Fall**，即 DSL 的正确识别率约 **90%**

这说明 DSL 本身不是标签错了、也不是腰部 IMU 天生无法区分；问题主要是 LSTM 的表征方式没有学好它和真实 Fall 的差异。

更稳妥的结论是：

> CNN 能保留并识别 DSL 中区别于真实跌倒的局部动态模式，因此不会仅因动作整体剧烈就判为 Fall。相比之下，LSTM 将长序列逐步压缩为最终状态后，更容易把 DSL 的整体运动强度误当作跌倒特征。

所以你的实验已经有一个很漂亮的现象可写：**CNN 不仅总体指标更高，而且显著减少了对动态 ADL（尤其 DSL）的灾难性误报。**

注意表述成“CNN 在本数据集和本设置下更合适”，不要泛化成“CNN 永远比 LSTM 好”。





### 3. 使用的LSTM模型

```python
def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 LSTM baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 第一层 
    x = tf.keras.layers.LSTM(64,   
                             return_sequences=True, 
                             #activity_regularizer= tf.keras.regularizers.l1(1e-4)
                             )(inputs)
    x = tf.keras.layers.Dropout(rate = 0.2)(x)

    # 第二层 
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(rate = 0.2)(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    

    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_lstm1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
     
    return model
```











## STEP 05 LSTM 3layers 100 epoch LOSO



### 1. 实验结果

#### 1.1 最终指标

最终 pooled held-out 指标：
                accuracy: 0.7600
                macro_f1: 0.7461
       balanced_accuracy: 0.7497
              recall_ADL: 0.6833
        recall_Near_Fall: 0.6133
             recall_Fall: 0.9524



#### 2.1 详细情况

| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 24   | 0    | 6         | 30   |
| ADL        | DS       | 23   | 0    | 7         | 30   |
| ADL        | DSL      | 0    | 30   | 0         | 30   |
| ADL        | DSS      | 24   | 2    | 4         | 30   |
| ADL        | NW       | 18   | 0    | 12        | 30   |
| ADL        | POG      | 20   | 0    | 10        | 30   |
| ADL        | RSS      | 28   | 0    | 2         | 30   |
| ADL        | SQ       | 27   | 0    | 3         | 30   |
| Fall       | CS       | 1    | 26   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 0    | 30   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 2    | 25   | 3         | 30   |
| Fall       | trip     | 1    | 26   | 3         | 30   |
| Near_Fall  | CS       | 13   | 0    | 14        | 27   |
| Near_Fall  | HB       | 11   | 0    | 19        | 30   |
| Near_Fall  | ITCS     | 0    | 0    | 3         | 3    |
| Near_Fall  | ITRS     | 10   | 0    | 20        | 30   |
| Near_Fall  | slip     | 9    | 0    | 21        | 30   |
| Near_Fall  | trip     | 14   | 1    | 15        | 30   |
| All        |          | 225  | 233  | 142       | 600  |



### 2. 实验分析

是的，`LSTM(16)` 里的 **16 不是步长**，也不是“读 16 个时间点”。

它叫 `units=16`：表示这一层 LSTM 在每个时间点维护一个 **16 维的隐藏状态／记忆向量**。可以把它理解为：

```python
x = LSTM(16)(x)
```

> 让这一层用 16 个“记忆特征”来概括前面传来的整段动作信息。

它和 CNN 里的 `Conv1D(32, kernel_size=7)` 中的 `32` 有一点类似——都是这一层输出特征的数量；但 LSTM 的 16 个特征是随时间不断更新的“记忆状态”，不是卷积核或步长。

如果你的结构是类似：

```python
LSTM(64, return_sequences=True)
LSTM(32, return_sequences=True)
LSTM(16, return_sequences=False)
```

那么前两层仍输出长度为 1024 的时间序列；最后的 `LSTM(16)` 才把整个动作压缩成一个 16 维向量，交给 Softmax 分类器。这个结构本身没有问题。

你的结论也基本成立，而且第三层带来的改善是明确但有限的：

| 指标             | 两层 LSTM | 三层 LSTM | 变化     |
| ---------------- | --------- | --------- | -------- |
| Accuracy         | 0.7250    | 0.7600    | +3.5%    |
| Macro-F1         | 0.7120    | 0.7461    | +0.034   |
| ADL recall       | 0.6042    | 0.6833    | 明显改善 |
| Near Fall recall | 0.6133    | 0.6133    | 几乎不变 |
| Fall recall      | 0.9429    | 0.9524    | 已经很高 |

所以更准确的说法不是“加一层完全没用”，而是：

> 第三层提升了模型对部分 ADL 的识别，但没有解决 LSTM 最核心的 ADL–Near Fall 边界问题，也没有改变 DSL 被系统性误判为 Fall 的模式。

从这张最终混淆矩阵看：

- ADL：`164/240 = 68.3%` 正确；仍有 `44` 被判 Near Fall、`32` 被判 Fall。
- Near Fall：`92/150 = 61.3%` 正确；`57` 被判 ADL，接近 38%。
- Fall：`200/210 = 95.2%` 正确；与另外两类几乎不混淆。

而且 DSL 又是 `30/30 → Fall`。这很关键：加深 LSTM 后，这个错误规则没有被打破，说明不是简单的“容量不足”。

你最后的猜测可以稍微修正得更严谨：

> LSTM 很可能已经学到“动作整体是否包含明显、剧烈的姿态转变”这一类全局线索，因此能稳定检出 Fall；但它将 1024 点序列逐步压缩为隐藏状态，可能未能充分保留区分 DSL、动态 ADL 与 Near Fall 所需的局部冲击、短时震荡和动作阶段细节。相较之下，1D-CNN 的局部卷积和池化机制更直接地针对这些局部波形模式，因此在本数据集、腰部六通道和 LOSO 设置下表现更优。

一个小补充：LSTM 并非天然“看不到局部模式”，它每一步当然也输入局部信号；问题在于它最后依赖连续记忆传递、再压缩成一个摘要。你的 CNN 则能在**任何时间位置**检测一个短时峰值／震荡模式，再通过池化保留下来——这正好更适合这个任务。

所以现在不必再继续堆第四层 LSTM 了。你已经有很完整的 baseline 对照：**三层 LSTM 是有效模型，但明显不如 CNN，且错误模式稳定可解释。**







### 3. 使用的LSTM模型

```python
def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 LSTM baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 第一层 
    x = tf.keras.layers.LSTM(64,   
                             return_sequences=True, 
                             #activity_regularizer= tf.keras.regularizers.l1(1e-4)
                             )(inputs)
    x = tf.keras.layers.Dropout(rate = 0.2)(x)

    # 第二层 
    x = tf.keras.layers.LSTM(32,
                            return_sequences=True,)(x)
    x = tf.keras.layers.Dropout(rate = 0.2)(x)

    # 第三层
    x = tf.keras.layers.LSTM(16)(x)
    x = tf.keras.layers.Dropout(rate = 0.2)(x)

    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    

    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_lstm1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
     
    return model
```







## STEP 06 TCN 2 ResBlock 100 epoch LOSO 



### 1. 实验结果



#### 1.1 最终指标

最终 pooled held-out 指标：               

 accuracy: 0.8050                

macro_f1: 0.8012       

balanced_accuracy: 0.8009              

recall_ADL: 0.7875        

recall_Near_Fall: 0.7533             

recall_Fall: 0.8619



#### 1.2 详细情况

| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 26   | 0    | 4         | 30   |
| ADL        | DS       | 23   | 0    | 7         | 30   |
| ADL        | DSL      | 9    | 21   | 0         | 30   |
| ADL        | DSS      | 26   | 0    | 4         | 30   |
| ADL        | NW       | 23   | 0    | 7         | 30   |
| ADL        | POG      | 27   | 0    | 3         | 30   |
| ADL        | RSS      | 27   | 0    | 3         | 30   |
| ADL        | SQ       | 28   | 0    | 2         | 30   |
| Fall       | CS       | 3    | 24   | 0         | 27   |
| Fall       | HB       | 2    | 28   | 0         | 30   |
| Fall       | ITCS     | 2    | 1    | 0         | 3    |
| Fall       | ITDS     | 6    | 24   | 0         | 30   |
| Fall       | ITRS     | 1    | 29   | 0         | 30   |
| Fall       | LCC      | 3    | 27   | 0         | 30   |
| Fall       | slip     | 5    | 23   | 2         | 30   |
| Fall       | trip     | 0    | 25   | 5         | 30   |
| Near_Fall  | CS       | 6    | 1    | 20        | 27   |
| Near_Fall  | HB       | 6    | 0    | 24        | 30   |
| Near_Fall  | ITCS     | 0    | 0    | 3         | 3    |
| Near_Fall  | ITRS     | 7    | 1    | 22        | 30   |
| Near_Fall  | slip     | 7    | 2    | 21        | 30   |
| Near_Fall  | trip     | 5    | 2    | 23        | 30   |
| All        |          | 242  | 208  | 150       | 600  |





### 2. 实验分析



这版 TCN 的最终结果是“能用，但明显不如 CNN baseline”。

| 模型      | Accuracy | Macro-F1 | Near-Fall recall |
| --------- | -------- | -------- | ---------------- |
| 1D-CNN    | 0.948    | 0.943    | 0.907            |
| 当前 TCN  | 0.805    | 0.801    | 0.753            |
| 两层 LSTM | 0.725    | 0.712    | 0.613            |

所以它比 LSTM 好不少，但相对 CNN 少正确识别了 **86 个 trial**（600 个 held-out trial 中：TCN 对 483 个，CNN 对 569 个）。

截图前 12 个 epoch 严格说还不是“上来就完全过拟合”：

- train loss：`1.001 → 0.488`
- val loss：`0.883 → 0.584`

验证 loss 仍在下降；只是从第 4–7 轮开始，训练准确率明显超过验证准确率，出现了泛化差距。若后续训练准确率到 1.0、而验证 loss 回升，那才是明确的后期过拟合。

不过，**过拟合不是这版 TCN 表现差的唯一、甚至不是最核心的原因。**

当前两残差块的 dilation 是 `1 → 2`，每块两次 `kernel_size=3` 卷积。它的有效感受野只有约 **13 个采样点**。而一条动作有 1024 点——TCN 卷积本身只能看非常局部的瞬时变化。

换成 GAP 后，模型变成：
$$
(1024,16)\rightarrow(16)\rightarrow 3类
$$



它把整段动作压成 16 个“平均特征强度”。这避免了 `Flatten` 记住训练者的具体波形，但也丢掉了重要的时序结构：例如“先失衡、后冲击、再静止”发生的先后顺序和位置。

这能解释混淆矩阵：

- ADL 有 `30+21=51` 条错，很多被认成 Near-Fall 或 Fall；
- Near-Fall 最常被认成 ADL（31 条）；
- Fall 也有 22 条被认成 ADL。

换言之，当前 TCN 没能把一整段动作过程编码好，三类边界变得比 CNN 模糊。

我会把这一版正式记为：

> **TCN-2ResBlock-GAP：参数少、避免 Flatten 的大分类器，但短感受野 + 全局平均汇总导致时序判别不足；在 subject-independent LOSO 上显著弱于 1D-CNN。**

这其实是很好的项目实验结论，不是失败：你的数据更适合 CNN 提取局部冲击、峰值和震荡模式；CNN 对 DSL 的识别也显著更稳。

如果你还想“给 TCN 一次公平机会”，下一版不该只是加一点 dropout，而应构造一个真正覆盖整段序列的 TCN：dilation 用 `1, 2, 4, 8, 16, 32, 64, 128`。两次卷积/块、kernel=3 时，感受野才接近 1021 点，能够真正看到完整动作。否则，这个两块模型更像“局部卷积 + 平均分类器”，而不是充分发挥 TCN 长时序能力的版本。







### 3. 使用的TCN模型



对，你明明改了，**这份 `phase1_tcn_loso_commented(1).py` 确实是 TCN**。上一条我误读成 CNN 文件了，抱歉。

你当前真正的结构是：

```text
输入 (1024, 6)
  ↓
残差 TCN Block 1
  - Conv1D(32, kernel=3, dilation=1) + ReLU
  - Conv1D(32, kernel=3, dilation=1)
  - shortcut 卷积，把 6 通道变为 32 通道
  - 两路相加 + ReLU
  ↓
(1024, 32)
  ↓
残差 TCN Block 2
  - Conv1D(16, kernel=3, dilation=2) + ReLU
  - Conv1D(16, kernel=3, dilation=2)
  - shortcut 卷积，把 32 通道变为 16 通道
  - 两路相加 + ReLU
  ↓
(1024, 16)
  ↓
GlobalAveragePooling1D
  ↓
(16,)
  ↓
Dense(3, softmax)
```

所以它满足 TCN 的核心特征：

- `Conv1D`：沿时间维度处理 IMU 序列；
- `dilation_rate=1 → 2`：使用逐渐扩大的膨胀卷积；
- `ResBlock`：有残差/捷径连接；
- 全程不改变 1024 个时间点的长度。

而且你已经按建议完成了关键改动：

```python
# 原来
# x = tf.keras.layers.Flatten()(x)

# 现在
x = tf.keras.layers.GlobalAveragePooling1D()(x)
```

这意味着现在模型不是把 `(1024, 16)` 的全部 `16,384` 个位置特征交给 Dense 去记忆，而是把每个特征通道在整段动作中的平均响应汇总成 16 个数，再分类。

当前模型参数量也因此非常小，大约只有 **8,259 个参数**，其中最终分类层仅：

[
16 \times 3 + 3 = 51
]

个参数。训练 accuracy 仍可能到 1.0，但它已经不是此前那个 `Flatten + Dense` 造成的大分类器记忆问题了；接下来若仍过拟合，更值得检查的是：训练 trial 数少、不同受试者动作风格差异大，以及 TCN 本身没有 dropout / L2 正则化。





```python
def ResBlock(x,filters,kernel_size,dilation_rate):
    r=tf.keras.layers.Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
    r=tf.keras.layers.Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r) #第二卷积
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    o=tf.keras.layers.Add()([r,shortcut])
    o=tf.keras.layers.Activation('relu')(o)  #激活函数
    return o





def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 第一层
    x=ResBlock(inputs,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)
     
    #x=tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model
```











## STEP 07 TCN 8 ResBlock 100 epoch LOSO 

### 1. 实验结果



#### 1.1最终指标



最终 pooled held-out 指标：
                accuracy: 0.8917
                macro_f1: 0.8799
       balanced_accuracy: 0.8763
              recall_ADL: 0.9375
        recall_Near_Fall: 0.7533
             recall_Fall: 0.9381



#### 1.2 详细情况



| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 28   | 0    | 2         | 30   |
| ADL        | DSL      | 27   | 3    | 0         | 30   |
| ADL        | DSS      | 26   | 0    | 4         | 30   |
| ADL        | NW       | 29   | 0    | 1         | 30   |
| ADL        | POG      | 29   | 0    | 1         | 30   |
| ADL        | RSS      | 26   | 0    | 4         | 30   |
| ADL        | SQ       | 30   | 0    | 0         | 30   |
| Fall       | CS       | 0    | 27   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 3    | 27   | 0         | 30   |
| Fall       | ITRS     | 1    | 29   | 0         | 30   |
| Fall       | LCC      | 1    | 29   | 0         | 30   |
| Fall       | slip     | 0    | 27   | 3         | 30   |
| Fall       | trip     | 0    | 25   | 5         | 30   |
| Near_Fall  | CS       | 3    | 4    | 20        | 27   |
| Near_Fall  | HB       | 2    | 3    | 25        | 30   |
| Near_Fall  | ITCS     | 2    | 0    | 1         | 3    |
| Near_Fall  | ITRS     | 4    | 2    | 24        | 30   |
| Near_Fall  | slip     | 4    | 4    | 22        | 30   |
| Near_Fall  | trip     | 5    | 4    | 21        | 30   |
| All        |          | 250  | 217  | 133       | 600  |





### 2. 实验分析

这组结果非常有价值，基本证明了你刚才的判断：**对这个任务，TCN 的长感受野确实重要。**

| TCN版本    | Accuracy   | Macro-F1   | ADL recall | Near-Fall recall | Fall recall |
| ---------- | ---------- | ---------- | ---------- | ---------------- | ----------- |
| 2 ResBlock | 0.8050     | 0.8012     | 0.7875     | 0.7533           | 0.8619      |
| 8 ResBlock | **0.8917** | **0.8799** | **0.9375** | 0.7533           | **0.9381**  |

从 2-block 到 8-block：

- 正确数从 `483 / 600` 增加到 `535 / 600`，多对了 **52 条**；
- 错误数从 117 降到 65，减少约 **44%**；
- Near-Fall 的召回率却恰好仍是 `113 / 150 = 75.33%`。

所以更准确的结论是：

> 更长的感受野显著改善了 ADL 与 Fall 的整体动作过程建模，但还没有解决 Near-Fall 这种本来就处于两类之间的边界类别。

最漂亮的证据就是 DSL。

- 2-block TCN：30 次 DSL 中，**21 次被误判为 Fall**
- 8-block TCN：30 次 DSL 中，**27 次正确识别为 ADL**，只有 3 次误判为 Fall

这很符合动作逻辑。浅 TCN 只看很局部的片段，看到“身体快速下降、接近躺下”便容易把 DSL 当作 Fall；8-block TCN 能把更完整的 1024 点过程联系起来，因而能区分：

- DSL：有控制地从站立下降到躺姿；
- Fall：失稳/冲击/落地的整体过程。

Near-Fall 则困难得多，因为它本身就是中间态：

- `20 / 150` 被当成 ADL：模型认为这次失衡不够显著，或恢复得太平稳；
- `17 / 150` 被当成 Fall：模型捕捉到了强烈失衡或冲击，但没充分判断出“最后恢复了”。

而且 Near-Fall 的错误分布很广，不是单一动作拖后腿：

- CS：20/27 正确；
- HB：25/30；
- ITRS：24/30；
- slip：22/30；
- trip：21/30。

这说明它不是像原先 DSL 那样的“某个 scenario 被系统性误判”，而是 **Near-Fall 作为类别本身的内部变化更大、与两边都重叠**。

顺带从列看 precision：

- ADL precision：`225 / 250 = 90.0%`
- Near-Fall precision：`113 / 133 = 85.0%`
- Fall precision：`197 / 217 = 90.8%`

Near-Fall 不只是 recall 最低，precision 也最低；它确实是当前三分类任务的主要瓶颈。

这轮可以写成一个很清楚的实验结论：

> Increasing TCN depth with exponentially increasing dilation substantially improved LOSO generalisation, particularly for distinguishing controlled descending movements from true falls. However, Near-Fall remained the most challenging class, with errors distributed toward both ADL and Fall, reflecting its transitional and heterogeneous motion characteristics.





### 3. 使用的TCN模型





```python
def ResBlock(x,filters,kernel_size,dilation_rate):
    r=tf.keras.layers.Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
    r=tf.keras.layers.Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r) #第二卷积
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    o=tf.keras.layers.Add()([r,shortcut])
    o=tf.keras.layers.Activation('relu')(o)  #激活函数
    return o





def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 第一层
    x=ResBlock(inputs,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)

    # 第3-8层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=8)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=16)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=32)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=64)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=128)
     
    #x=tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model
```



### 4. 解释summary.json文件



这个 `summary.json` 是对完整 **10-fold LOSO** 结果的汇总：10 次实验中，每次留 1 位受试者完全不参与训练，最后把 10 位受试者各自的测试预测合并起来统计。

它**不是一个模型在 600 条数据上测试**；而是 10 个分别训练出的模型，各自测试从未见过的一位受试者，合起来正好覆盖 600 条 trial。

#### 4.1 顶部实验信息

- `task`：任务是对每个完整 1024 点 trial 做三分类：ADL / Near Fall / Fall。
- `channels`：输入是腰部 IMU 六通道，加速度 xyz + 角速度 xyz。
- `sequence_length: 1024`：每个动作被处理为 1024 个时间点。
- `folds_completed`：`sub1` 到 `sub10` 均已作为一次独立测试集完成；LOSO 全部跑完。

注意：这个 JSON 没有保存“8 ResBlock TCN”的具体网络结构；结构在运行的 TCN 脚本或保存的模型中，JSON 主要存结果。

#### 4.2`pooled_held_out_predictions`：最重要的一组指标

它把 10 折中所有 held-out 受试者的预测拼在一起，按总共 **600 条从未参与各自模型训练的 trial** 计算。

| 指标              | 数值   | 含义                                                         |
| ----------------- | ------ | ------------------------------------------------------------ |
| accuracy          | 0.8917 | 600 条中总共预测对了 **535 条**                              |
| macro_f1          | 0.8799 | 三个类别的 F1 分数等权平均，不让 ADL / Fall 因样本更多而主导结论 |
| balanced_accuracy | 0.8763 | 三类 recall 的平均值，反映三类是否都识别得均衡               |
| recall_ADL        | 0.9375 | 240 条真实 ADL 中，正确识别 **225 条**                       |
| recall_Near_Fall  | 0.7533 | 150 条真实 Near Fall 中，正确识别 **113 条**                 |
| recall_Fall       | 0.9381 | 210 条真实 Fall 中，正确识别 **197 条**                      |

#### 4.3 你命令行打印的：

```text
最终 pooled held-out 指标：
accuracy: 0.8917
macro_f1: 0.8799
balanced_accuracy: 0.8763
recall_ADL: 0.9375
recall_Near_Fall: 0.7533
recall_Fall: 0.9381
```

就是 JSON 里的 `pooled_held_out_predictions`，只是保留到四位小数显示。

#### 4.4 三个容易混淆的指标

- **Accuracy**：不分类型，直接看总共对了多少。这里是 `535 / 600 = 89.17%`。
- **Recall（召回率）**：只站在某个真实类别的角度看，“真实是这一类的样本，有多少被找回来”。所以混淆矩阵按行归一化时，**对角线就是各类别 recall**。
- **Balanced accuracy**：三类 recall 的算术平均：


$$
\frac{0.9375+0.7533+0.9381}{3}=0.8763
$$





它说明：虽然总 accuracy 是 89.17%，但考虑到 Near Fall 较难之后，三类的平均识别水平是 87.63%。

你的核心瓶颈也因此很清晰：ADL 和 Fall 均约 94% recall，Near Fall 只有 75.33%。

#### 4.5`classification_report`：每个类别更完整的成绩单

例如 Near Fall：

```json
"Near_Fall": {
  "precision": 0.8496,
  "recall": 0.7533,
  "f1-score": 0.7986,
  "support": 150
}
```

- `support`：真实属于该类的样本数。Near Fall 有 150 条。
- `recall`：真实 Near Fall 中预测对的比例，`113 / 150 = 75.33%`。
- `precision`：所有**被模型判成 Near Fall** 的样本中，真正是 Near Fall 的比例。这里模型一共判了 133 条 Near Fall，其中 113 条正确：

$$
\text{precision}_{NearFall}=\frac{113}{133}=84.96%
$$



- `f1-score`：precision 与 recall 的综合，采用调和平均：

$$
F1=2\times\frac{\text{precision}\times\text{recall}}
{\text{precision}+\text{recall}}
$$

Near Fall 的 precision 不算低，但 recall 较低，说明它的主要问题是：**不少真实 Near Fall 被分流到 ADL 或 Fall**；而不是模型把大量 ADL/Fall 胡乱说成 Near Fall。

`macro avg` 是三类各算一次后等权平均，因此和 `macro_f1` 对应：

```text
macro avg F1 = 0.8799
```

`weighted avg` 则按各类样本量加权；ADL 240、Near Fall 150、Fall 210。它更贴近数据集总体表现，但 Near Fall 的困难会被样本更多的两类部分掩盖。因此你的报告里优先写 `macro_f1` 和 `balanced_accuracy` 是合理的。

#### 4.6`mean_across_subject_folds`：每位受试者一票的平均成绩

这里不是合并全部 600 条样本，而是先算每一折的指标，再对 10 位测试受试者取平均。

你的结果：

```text
mean accuracy = 0.8917
mean macro-F1 = 0.8710
mean balanced accuracy = 0.8763
```

之所以 accuracy、三类 recall 和 balanced accuracy 恰好与 pooled 值一致，是因为每位受试者都有相同数量与构成的测试 trial：

- 24 条 ADL
- 15 条 Near Fall
- 21 条 Fall
- 合计 60 条

但 **F1 不是可直接线性相加的指标**，所以：

```text
pooled macro-F1 = 0.8799
mean fold macro-F1 = 0.8710
```

两者不完全相同是正常的，不是计算错误。

#### 4.7`standard_deviation_across_subject_folds`：不同受试者之间的波动

这是 10 折各自指标的样本标准差（代码用 `ddof=1`）。

| 指标              | 标准差     | 解读                               |
| ----------------- | ---------- | ---------------------------------- |
| accuracy          | 0.1052     | 不同受试者上的准确率波动较明显     |
| macro_f1          | 0.1346     | 各受试者三类综合表现有差异         |
| balanced_accuracy | 0.1287     | 类别均衡表现也存在明显个体差异     |
| recall_ADL        | 0.0947     | ADL 相对稳定                       |
| recall_Near_Fall  | **0.3282** | Near Fall 在不同受试者之间极不稳定 |
| recall_Fall       | 0.1030     | Fall 相对稳定                      |

其中最值得写进分析的是：

> Near Fall recall 的跨受试者标准差达到 0.328，远高于 ADL（0.095）和 Fall（0.103）。这说明 Near Fall 的难点不仅是总体召回较低，而且不同人的失衡、恢复和冲击模式差异很大，模型对该类的 subject-independent 泛化不稳定。

所以这组 8-block TCN 的一句话总结可以是：

> 8-residual-block TCN 在完整 LOSO 中达到 89.17% accuracy、0.8799 macro-F1 和 0.8763 balanced accuracy。增大膨胀卷积感受野后，ADL 与 Fall 召回率均接近 94%，但 Near Fall 仍是主要限制因素，且其跨受试者性能波动显著。



对，它们不一样。最关键的区别只有一句：

- **accuracy 看“全部 600 条里对了多少”**
- **precision 看“模型说某一类时，它说得有多准”**

用你这次 8-block TCN 的结果，可以还原成大致这张混淆矩阵：

| 真实类别 \ 预测类别 | 预测 ADL | 预测 Near Fall | 预测 Fall | 合计 |
| ------------------- | -------- | -------------- | --------- | ---- |
| 真实 ADL            | 225      | 12             | 3         | 240  |
| 真实 Near Fall      | 20       | 113            | 17        | 150  |
| 真实 Fall           | 5        | 8              | 197       | 210  |

先看最直观的 **accuracy**。

对角线是预测正确的：


$$
225+113+197=535
$$


总试验数是 600：
$$
\text{accuracy}=\frac{535}{600}=89.17%
$$



所以 accuracy 只是问：

> 不管类别是什么，模型总共猜对了多少？

它不特别关心错的是 Near Fall 还是 Fall。

------

而 **precision（精确率）** 是“站在模型预测结果”的角度问问题。

例如模型一共把多少条判为 Near Fall？
$$
12+113+8=133
$$
其中真正是 Near Fall 的只有 113 条：
$$
\text{precision}_{NearFall}=\frac{113}{133}=84.96%
$$



它的意思是：

> 模型每说“这是一条 Near Fall”，有约 85% 的概率它是对的。

但 Near Fall 的 **recall（召回率）** 是从真实标签的角度看：
$$
\text{recall}_{NearFall}=\frac{113}{150}=75.33%
$$


它的意思是：

> 全部 150 条真实 Near Fall 中，模型成功找回了 113 条；剩下 37 条漏掉了。

所以 Near Fall 当前的情况是：

- precision = 85.0%：模型判成 Near Fall 时，通常还挺靠谱；
- recall = 75.3%：但不少真正的 Near Fall 被它错分到了 ADL 或 Fall。

这两个指标的分母不同：

| 指标      | 分母                   | 关心的问题                           |
| --------- | ---------------------- | ------------------------------------ |
| Precision | 被模型预测成该类的总数 | “我说它是 Near Fall 时，准不准？”    |
| Recall    | 真实属于该类的总数     | “所有真实 Near Fall，我找回了多少？” |

------

至于 **balanced accuracy**，你理解得完全对：在这个多分类任务中，它就是三类 recall 的算术平均。
$$
\frac{0.9375+0.7533+0.9381}{3}=0.8763
$$
即：
$$
\text{balanced accuracy}=87.63%
$$


为什么要额外算它？因为普通 accuracy 会被“容易的类别”拉高。

你这里 ADL 和 Fall 都接近 94% recall，但 Near Fall 只有 75%。单看 accuracy 的 89.17%，容易忽略 Near Fall 是明显短板。balanced accuracy 给每一类同等权重，相当于问：

> 如果我公平地看 ADL、Near Fall、Fall 三门考试，平均考了多少？

因此你这个结果可简单理解为：

- **Accuracy 89.17%**：600 条动作总体对了 535 条。
- **Balanced accuracy 87.63%**：三类平均识别水平约 87.6%，把 Near Fall 的弱项纳入考量。
- **Near-Fall precision 85.0%**：预测成 Near Fall 的样本，大多数确实是 Near Fall。
- **Near-Fall recall 75.3%**：但真实 Near Fall 中约四分之一没有被成功识别出来。





### 5. 对比分析已有的模型

有了这张记录，两层 LSTM 的六项指标现在可以完整补齐。按相同的 **10-fold LOSO、600 条 pooled held-out trial** 口径比较：

| 模型                | Accuracy   | Macro-F1   | Balanced Accuracy | ADL Recall | Near-Fall Recall | Fall Recall |
| ------------------- | ---------- | ---------- | ----------------- | ---------- | ---------------- | ----------- |
| **普通三层 1D-CNN** | **0.9483** | **0.9433** | **0.9435**        | **0.9667** | **0.9067**       | **0.9571**  |
| **8-ResBlock TCN**  | 0.8917     | 0.8799     | 0.8763            | 0.9375     | 0.7533           | 0.9381      |
| **2-ResBlock TCN**  | 0.8050     | 0.8012     | 0.8009            | 0.7875     | 0.7533           | 0.8619      |
| **三层 LSTM**       | 0.7600     | 0.7461     | 0.7497            | 0.6833     | 0.6133           | 0.9524      |
| **两层 LSTM**       | 0.7250     | 0.7120     | 0.7201            | 0.6042     | 0.6133           | 0.9429      |

#### 5.1 总体排名

1. 普通三层 1D-CNN
2. 8-ResBlock TCN
3. 2-ResBlock TCN
4. 三层 LSTM
5. 两层 LSTM

#### 5.2 两层与三层 LSTM 的准确比较

三层 LSTM 相比两层 LSTM：

| 指标              | 两层   | 三层   | 变化        |
| ----------------- | ------ | ------ | ----------- |
| Accuracy          | 0.7250 | 0.7600 | **+0.0350** |
| Macro-F1          | 0.7120 | 0.7461 | **+0.0341** |
| Balanced Accuracy | 0.7201 | 0.7497 | **+0.0296** |
| ADL Recall        | 0.6042 | 0.6833 | **+0.0791** |
| Near-Fall Recall  | 0.6133 | 0.6133 | **无变化**  |
| Fall Recall       | 0.9429 | 0.9524 | **+0.0095** |

所以，增加第三层 LSTM 的改善主要来自 **ADL**：

- ADL recall 从 60.42% 提高到 68.33%；
- Fall 本来就很高，只小幅提高；
- **Near-Fall 完全没有改善，仍然是 61.33%。**

这意味着三层 LSTM 的总体指标变好，并不是因为它更会识别 Near-Fall，而是因为它减少了一部分 ADL 错误。

#### 5.3 五个模型揭示的核心规律

- **1D-CNN 最强且最均衡**：三类 recall 全部超过 90%，尤其 Near-Fall 达到 90.67%。
- **TCN 非常依赖足够长的感受野**：从 2 个增加到 8 个 ResBlock 后，Accuracy 从 80.50% 上升到 89.17%。
- 但 TCN 的这一提升主要发生在 ADL 和 Fall；Near-Fall recall 在两个版本中都恰好是 **75.33%**。
- **LSTM 对 Fall 有明显偏向**：两层和三层 LSTM 的 Fall recall 都超过94%，但 ADL 和 Near-Fall 明显较差。
- 单纯增加 LSTM 层数收益有限：总体提高约3–3.5个百分点，但最困难的 Near-Fall 没有改善。

因此目前最有依据的结论是：

> 普通 1D-CNN 不仅总体成绩最高，而且是唯一同时较好解决 ADL、Near-Fall 和 Fall 三类识别的模型。加深 TCN 确实证明了长感受野的重要性；加深 LSTM 则主要改善了 ADL，对 Near-Fall 没有帮助。









## STEP 08 TCN 6 ResBlock 100 epoch LOSO



### 1. 实验结果

#### 1.1 最终指标

最终 pooled held-out 指标：
                accuracy: 0.8783
                macro_f1: 0.8693
       balanced_accuracy: 0.8664
              recall_ADL: 0.8917
        recall_Near_Fall: 0.7600
             recall_Fall: 0.9476

#### 1.2 详细情况

| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 28   | 0    | 2         | 30   |
| ADL        | DS       | 27   | 0    | 3         | 30   |
| ADL        | DSL      | 19   | 11   | 0         | 30   |
| ADL        | DSS      | 26   | 0    | 4         | 30   |
| ADL        | NW       | 26   | 0    | 4         | 30   |
| ADL        | POG      | 30   | 0    | 0         | 30   |
| ADL        | RSS      | 28   | 0    | 2         | 30   |
| ADL        | SQ       | 30   | 0    | 0         | 30   |
| Fall       | CS       | 1    | 26   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 2    | 28   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 1    | 26   | 3         | 30   |
| Fall       | trip     | 0    | 26   | 4         | 30   |
| Near_Fall  | CS       | 1    | 2    | 24        | 27   |
| Near_Fall  | HB       | 7    | 2    | 21        | 30   |
| Near_Fall  | ITCS     | 3    | 0    | 0         | 3    |
| Near_Fall  | ITRS     | 5    | 2    | 23        | 30   |
| Near_Fall  | slip     | 5    | 1    | 24        | 30   |
| Near_Fall  | trip     | 7    | 1    | 22        | 30   |
| All        |          | 246  | 218  | 136       | 600  |



### 2. 实验分析

#### **和 4 Resblock一起分析**





### 3. 使用的TCN模型

```python 
def ResBlock(x,filters,kernel_size,dilation_rate):
    r=tf.keras.layers.Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
    r=tf.keras.layers.Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r) #第二卷积
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    o=tf.keras.layers.Add()([r,shortcut])
    o=tf.keras.layers.Activation('relu')(o)  #激活函数
    return o





def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()
    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 第一层
    x=ResBlock(inputs,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)

    # 第3-6层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=8)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=16)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=32)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=64)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=128)

    #x=tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model
```





## STEP 09 TCN 4 ResBlock 100 epoch LOSO

### 1. 实验结果



#### 1.1 最终指标

最终 pooled held-out 指标：
                accuracy: 0.8933
                macro_f1: 0.8870
       balanced_accuracy: 0.8829
              recall_ADL: 0.9250
        recall_Near_Fall: 0.8000
             recall_Fall: 0.9238



#### 1.2 详细情况

| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 26   | 0    | 4         | 30   |
| ADL        | DSL      | 21   | 9    | 0         | 30   |
| ADL        | DSS      | 29   | 0    | 1         | 30   |
| ADL        | NW       | 26   | 0    | 4         | 30   |
| ADL        | POG      | 30   | 0    | 0         | 30   |
| ADL        | RSS      | 30   | 0    | 0         | 30   |
| ADL        | SQ       | 30   | 0    | 0         | 30   |
| Fall       | CS       | 0    | 27   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 4    | 26   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 5    | 23   | 2         | 30   |
| Fall       | trip     | 0    | 25   | 5         | 30   |
| Near_Fall  | CS       | 4    | 5    | 18        | 27   |
| Near_Fall  | HB       | 7    | 1    | 22        | 30   |
| Near_Fall  | ITCS     | 0    | 0    | 3         | 3    |
| Near_Fall  | ITRS     | 4    | 1    | 25        | 30   |
| Near_Fall  | slip     | 1    | 0    | 29        | 30   |
| Near_Fall  | trip     | 5    | 2    | 23        | 30   |
| All        |          | 252  | 212  | 136       | 600  |



### 2. 实验分析



这个结果说明了一件很重要的事：

> **TCN 的性能并不会随着 ResBlock 数量和感受野单调上升。**
> 感受野太短确实不行，但覆盖完整 1024 点也不代表所有类别都会继续改善。

先把 4、6、8 层放在一起：

| TCN        | 理论感受野 | Accuracy   | Macro-F1   | Balanced Acc. | ADL Recall | Near-Fall Recall | Fall Recall |
| ---------- | ---------- | ---------- | ---------- | ------------- | ---------- | ---------------- | ----------- |
| 4 ResBlock | 61         | **0.8933** | **0.8870** | **0.8829**    | 0.9250     | **0.8000**       | 0.9238      |
| 6 ResBlock | 253        | 0.8783     | 0.8693     | 0.8664        | 0.8917     | 0.7600           | **0.9476**  |
| 8 ResBlock | 1021       | 0.8917     | 0.8799     | 0.8763        | **0.9375** | 0.7533           | 0.9381      |

这里的感受野按“每块两次 `kernel_size=3` 卷积、dilation 按 1、2、4……增长”计算。

#### 2.1 4层和8层其实几乎打平

换算成600条 trial 的正确数量：

| 模型       | 正确数  | 错误数 |
| ---------- | ------- | ------ |
| 4 ResBlock | **536** | 64     |
| 6 ResBlock | 527     | 73     |
| 8 ResBlock | 535     | 65     |

4层比8层的 Accuracy 高，只相当于：

> **600条 trial 中净多预测正确了1条。**

所以目前不能说“4层总体明显优于8层”。更准确的说法是：

- 4层和8层总体表现基本相同；
- 但两者擅长的类别不同；
- 4层更擅长 Near-Fall；
- 8层更擅长 ADL，特别是 DSL。

#### 2.2 三种深度各自正确识别了多少条

| 模型       | ADL（240） | Near-Fall（150） | Fall（210） | 总正确数 |
| ---------- | ---------- | ---------------- | ----------- | -------- |
| 4 ResBlock | 222        | **120**          | 194         | **536**  |
| 6 ResBlock | 214        | 114              | **199**     | 527      |
| 8 ResBlock | **225**    | 113              | 197         | 535      |

4层相对于8层：

- Near-Fall 多识别正确 **7条**；
- ADL 少识别正确 **3条**；
- Fall 少识别正确 **3条**；
- 最终净多正确 **1条**。

因此，4层的优势不是“所有方面都更好”，而是它用一部分 ADL、Fall 表现，换来了更高的 Near-Fall recall。

#### 2.3 4层为什么Near-Fall更高

从场景拆开看：

| Near-Fall场景 | 4层正确 | 6层正确 | 8层正确 |
| ------------- | ------- | ------- | ------- |
| CS（27）      | 18      | **24**  | 20      |
| HB（30）      | 22      | 21      | **25**  |
| ITCS（3）     | **3**   | 0       | 约1     |
| ITRS（30）    | **25**  | 23      | 24      |
| slip（30）    | **29**  | 24      | 22      |
| trip（30）    | **23**  | 22      | 21      |
| 合计          | **120** | 114     | 113     |

4层的 Near-Fall 优势并不是每个动作都更好：

- CS反而明显弱于6层；
- HB弱于8层；
- 最大优势来自 Near-Fall `slip`：4层识别对29条，6层24条，8层22条；
- 只有3条的ITCS也贡献了一部分差距，但样本太少，不应过度解读。

所以更精确的结论是：

> 4层模型特别善于识别部分 Near-Fall 动作，尤其是 slip，而不是对所有 Near-Fall 场景都普遍更强。

#### 2.4 DSL确实受益于更长感受野

| 模型       | DSL正确识别为ADL | 错判为Fall |
| ---------- | ---------------- | ---------- |
| 4 ResBlock | 21/30            | 9/30       |
| 6 ResBlock | 19/30            | 11/30      |
| 8 ResBlock | **27/30**        | **3/30**   |

这一结果仍然支持此前的解释：

- DSL包含明显的身体下降过程；
- 短、中等感受野看到下降片段时，容易当成跌倒；
- 8层接近覆盖完整trial，可以进一步看到这是“受控地躺下”，因此DSL改善明显。

但是，对Near-Fall来说，更长的上下文不一定更有利。Near-Fall本身位于ADL与Fall之间，8层模型可能更倾向根据完整动作过程作出较明确的ADL/Fall判断，反而把一部分边界样本从Near-Fall分流出去。

#### 2.5 为什么6层反而最弱

感受野从61增加到253，并没有自动带来更好结果。可能同时受这些因素影响：

- 模型更深，优化难度增加；
- 数据量很小，每个LOSO折真正用于训练的trial有限；
- 不同随机初始化会产生波动；
- Early Stopping使每一折实际训练轮数不同；
- 不同深度可能学习到了不同的类别偏好：
  - 4层偏向Near-Fall；
  - 6层偏向Fall；
  - 8层偏向ADL和Fall。

因此，6层当前看起来是一个“中间深度但没有取得平衡”的版本，不能简单理解成感受野253本身不好。

#### 2.6 当前最严谨的实验结论

可以这样记录：

> Expanding the TCN receptive field from the shallow two-block architecture substantially improved classification performance. However, performance did not increase monotonically with network depth. The four-block and eight-block models achieved nearly identical overall accuracy, while showing different class-specific strengths. The four-block TCN obtained the highest Near-Fall recall, particularly for slip-related Near-Fall trials, whereas the eight-block TCN provided the most reliable ADL recognition and substantially reduced DSL-to-Fall confusion.

下一步不必再试5层、7层反复碰运气。最有价值的是把 **4层和8层分别跑3个随机种子**：

- 比较三次 `macro-F1` 和 Near-Fall recall 的均值与标准差；
- 检查4层的Near-Fall优势是否稳定存在；
- 检查8层的DSL优势是否每次都能复现。

在单次运行下，4层与8层只差1条trial，完全可能受到随机性的影响；但它们呈现出的“4层擅长Near-Fall、8层擅长DSL”的模式，值得通过多随机种子验证。









### 3. 使用的TCN模型



```python
def ResBlock(x,filters,kernel_size,dilation_rate):
    r=tf.keras.layers.Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
    r=tf.keras.layers.Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(r) #第二卷积
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=tf.keras.layers.Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    o=tf.keras.layers.Add()([r,shortcut])
    o=tf.keras.layers.Activation('relu')(o)  #激活函数
    return o





def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()
    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 第一层
    x=ResBlock(inputs,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)

    # 第3-4层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=8)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=16)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=32)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=64)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=128)

    #x=tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model 
```







## STEP 10 Conv1D TCN 4 ResBlock 100 epoch LOSO



### 1. 最终指标

最终 pooled held-out 指标：
                accuracy: 0.9083
                macro_f1: 0.8994
       balanced_accuracy: 0.8910
              recall_ADL: 0.9750
        recall_Near_Fall: 0.7600
             recall_Fall: 0.9381

### 2. 详细情况



| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 30   | 0    | 0         | 30   |
| ADL        | DSL      | 26   | 4    | 0         | 30   |
| ADL        | DSS      | 29   | 0    | 1         | 30   |
| ADL        | NW       | 29   | 0    | 1         | 30   |
| ADL        | POG      | 30   | 0    | 0         | 30   |
| ADL        | RSS      | 30   | 0    | 0         | 30   |
| ADL        | SQ       | 30   | 0    | 0         | 30   |
| Fall       | CS       | 1    | 26   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 3    | 27   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 2    | 26   | 2         | 30   |
| Fall       | trip     | 0    | 25   | 5         | 30   |
| Near_Fall  | CS       | 7    | 1    | 19        | 27   |
| Near_Fall  | HB       | 7    | 0    | 23        | 30   |
| Near_Fall  | ITCS     | 2    | 0    | 1         | 3    |
| Near_Fall  | ITRS     | 6    | 0    | 24        | 30   |
| Near_Fall  | slip     | 6    | 2    | 22        | 30   |
| Near_Fall  | trip     | 3    | 2    | 25        | 30   |
| All        |          | 271  | 206  | 123       | 600  |





### 3. 使用的Conv1D-TCN模型



```python
def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 额外加的卷积层
    x = tf.keras.layers.Conv1D(32, 7, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    #x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    # 第一层
    x=ResBlock(x,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)

    # 第3-N层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=8)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=16)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=32)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=64)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=128)
     
    #x=tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model
```











## STEP 11 Conv1D 2-layer TCN 4-ResBlock 100 epoch LOSO



### 1. 最终指标

终 pooled held-out 指标：
                accuracy: 0.9317
                macro_f1: 0.9256
       balanced_accuracy: 0.9232
              recall_ADL: 0.9667
        recall_Near_Fall: 0.8600
             recall_Fall: 0.9429



### 2. 详细情况

| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 28   | 0    | 2         | 30   |
| ADL        | DSL      | 27   | 3    | 0         | 30   |
| ADL        | DSS      | 29   | 0    | 1         | 30   |
| ADL        | NW       | 29   | 0    | 1         | 30   |
| ADL        | POG      | 30   | 0    | 0         | 30   |
| ADL        | RSS      | 30   | 0    | 0         | 30   |
| ADL        | SQ       | 29   | 0    | 1         | 30   |
| Fall       | CS       | 0    | 27   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 2    | 28   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 1    | 29   | 0         | 30   |
| Fall       | slip     | 0    | 27   | 3         | 30   |
| Fall       | trip     | 0    | 24   | 6         | 30   |
| Near_Fall  | CS       | 3    | 1    | 23        | 27   |
| Near_Fall  | HB       | 4    | 1    | 25        | 30   |
| Near_Fall  | ITCS     | 1    | 0    | 2         | 3    |
| Near_Fall  | ITRS     | 1    | 1    | 28        | 30   |
| Near_Fall  | slip     | 2    | 2    | 26        | 30   |
| Near_Fall  | trip     | 3    | 2    | 25        | 30   |
| All        |          | 249  | 208  | 143       | 600  |





### 3. 使用的Conv1D-TCN模型

```python
def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 额外加的卷积层
    # layer1
    x = tf.keras.layers.Conv1D(32, 7, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    #x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    #x = tf.keras.layers.Dropout(0.30)(x)

    # layer2
    x = tf.keras.layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    #x = tf.keras.layers.Dropout(0.30)(x)
    #layer3
    #x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Activation("relu")(x)

    # 第一层
    x=ResBlock(x,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)

    # 第3-N层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=8)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=16)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=32)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=64)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=128)
     
    #x=tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model
```







## STEP 12 Conv1D 3-layer TCN 4-ResBlock 100 epoch LOSO

### 1. 最终指标

最终 pooled held-out 指标：
                accuracy: 0.9100
                macro_f1: 0.9041
       balanced_accuracy: 0.9085
              recall_ADL: 0.8958
        recall_Near_Fall: 0.8867
             recall_Fall: 0.9429



### 2. 详细情况

| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 27   | 0    | 3         | 30   |
| ADL        | DS       | 22   | 0    | 8         | 30   |
| ADL        | DSL      | 28   | 2    | 0         | 30   |
| ADL        | DSS      | 26   | 0    | 4         | 30   |
| ADL        | NW       | 29   | 0    | 1         | 30   |
| ADL        | POG      | 28   | 0    | 2         | 30   |
| ADL        | RSS      | 25   | 0    | 5         | 30   |
| ADL        | SQ       | 30   | 0    | 0         | 30   |
| Fall       | CS       | 0    | 27   | 0         | 27   |
| Fall       | HB       | 1    | 29   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 2    | 28   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 0    | 26   | 4         | 30   |
| Fall       | trip     | 0    | 25   | 5         | 30   |
| Near_Fall  | CS       | 3    | 1    | 23        | 27   |
| Near_Fall  | HB       | 1    | 2    | 27        | 30   |
| Near_Fall  | ITCS     | 1    | 0    | 2         | 3    |
| Near_Fall  | ITRS     | 1    | 1    | 28        | 30   |
| Near_Fall  | slip     | 0    | 2    | 28        | 30   |
| Near_Fall  | trip     | 2    | 3    | 25        | 30   |
| All        |          | 226  | 209  | 165       | 600  |



### 3. 使用的Conv1D-TCN模型



```python
def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 额外加的卷积层
    # layer1
    x = tf.keras.layers.Conv1D(32, 7, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    #x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    #x = tf.keras.layers.Dropout(0.30)(x)

    # layer2
    x = tf.keras.layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    #x = tf.keras.layers.Dropout(0.30)(x)
    #layer3
    x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # 第一层
    x=ResBlock(x,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)

    # 第3-N层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=8)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=16)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=32)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=64)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=128)
     
    #x=tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model
```













## STEP 13 Conv1D 2-layer(dropout) TCN 4-ResBlock 100 epoch LOSO

### 1# 

最终 pooled held-out 指标：
                accuracy: 0.9067
                macro_f1: 0.8998
       balanced_accuracy: 0.9003
              recall_ADL: 0.9208
        recall_Near_Fall: 0.8467
             recall_Fall: 0.9333



| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 27   | 0    | 3         | 30   |
| ADL        | DSL      | 27   | 3    | 0         | 30   |
| ADL        | DSS      | 26   | 0    | 4         | 30   |
| ADL        | NW       | 28   | 0    | 2         | 30   |
| ADL        | POG      | 27   | 0    | 3         | 30   |
| ADL        | RSS      | 27   | 0    | 3         | 30   |
| ADL        | SQ       | 29   | 0    | 1         | 30   |
| Fall       | CS       | 1    | 26   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 3    | 27   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 1    | 29   | 0         | 30   |
| Fall       | slip     | 0    | 27   | 3         | 30   |
| Fall       | trip     | 0    | 24   | 6         | 30   |
| Near_Fall  | CS       | 5    | 3    | 19        | 27   |
| Near_Fall  | HB       | 1    | 2    | 27        | 30   |
| Near_Fall  | ITCS     | 0    | 0    | 3         | 3    |
| Near_Fall  | ITRS     | 2    | 1    | 27        | 30   |
| Near_Fall  | slip     | 2    | 2    | 26        | 30   |
| Near_Fall  | trip     | 3    | 2    | 25        | 30   |
| All        |          | 239  | 209  | 152       | 600  |



### 2#



最终 pooled held-out 指标：
                accuracy: 0.9067
                macro_f1: 0.8998
       balanced_accuracy: 0.9003
              recall_ADL: 0.9208
        recall_Near_Fall: 0.8467
             recall_Fall: 0.9333



| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 27   | 0    | 3         | 30   |
| ADL        | DSL      | 27   | 3    | 0         | 30   |
| ADL        | DSS      | 26   | 0    | 4         | 30   |
| ADL        | NW       | 28   | 0    | 2         | 30   |
| ADL        | POG      | 27   | 0    | 3         | 30   |
| ADL        | RSS      | 27   | 0    | 3         | 30   |
| ADL        | SQ       | 29   | 0    | 1         | 30   |
| Fall       | CS       | 1    | 26   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 3    | 27   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 1    | 29   | 0         | 30   |
| Fall       | slip     | 0    | 27   | 3         | 30   |
| Fall       | trip     | 0    | 24   | 6         | 30   |
| Near_Fall  | CS       | 5    | 3    | 19        | 27   |
| Near_Fall  | HB       | 1    | 2    | 27        | 30   |
| Near_Fall  | ITCS     | 0    | 0    | 3         | 3    |
| Near_Fall  | ITRS     | 2    | 1    | 27        | 30   |
| Near_Fall  | slip     | 2    | 2    | 26        | 30   |
| Near_Fall  | trip     | 3    | 2    | 25        | 30   |
| All        |          | 239  | 209  | 152       | 600  |







### 使用的Conv1D-TCN模型

```python
def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 额外加的卷积层
    # layer1
    x = tf.keras.layers.Conv1D(32, 7, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    #x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    #x = tf.keras.layers.Dropout(0.30)(x)

    # layer2
    x = tf.keras.layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dropout(0.30)(x)
    #layer3
    #x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Activation("relu")(x)

    # 第一层
    x=ResBlock(x,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)

    # 第3-N层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=8)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=16)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=32)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=64)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=128)
     
    #x=tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model
```







## STEP 14 Conv1D 2-layer(2dropout) TCN 4-ResBlock 100 epoch LOSO

### 1#

最终 pooled held-out 指标：
                accuracy: 0.9467
                macro_f1: 0.9414
       balanced_accuracy: 0.9407
              recall_ADL: 0.9792
        recall_Near_Fall: 0.9000
             recall_Fall: 0.9429



| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 30   | 0    | 0         | 30   |
| ADL        | DSL      | 28   | 2    | 0         | 30   |
| ADL        | DSS      | 29   | 0    | 1         | 30   |
| ADL        | NW       | 29   | 0    | 1         | 30   |
| ADL        | POG      | 30   | 0    | 0         | 30   |
| ADL        | RSS      | 29   | 0    | 1         | 30   |
| ADL        | SQ       | 30   | 0    | 0         | 30   |
| Fall       | CS       | 0    | 27   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 2    | 28   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 0    | 26   | 4         | 30   |
| Fall       | trip     | 0    | 24   | 6         | 30   |
| Near_Fall  | CS       | 2    | 4    | 21        | 27   |
| Near_Fall  | HB       | 0    | 1    | 29        | 30   |
| Near_Fall  | ITCS     | 1    | 0    | 2         | 3    |
| Near_Fall  | ITRS     | 1    | 2    | 27        | 30   |
| Near_Fall  | slip     | 2    | 1    | 27        | 30   |
| Near_Fall  | trip     | 0    | 1    | 29        | 30   |
| All        |          | 243  | 209  | 148       | 600  |



### 2#

最终 pooled held-out 指标：
                accuracy: 0.9467
                macro_f1: 0.9414
       balanced_accuracy: 0.9407
              recall_ADL: 0.9792
        recall_Near_Fall: 0.9000
             recall_Fall: 0.9429



| true_label | scenario | ADL  | Fall | Near_Fall | All  |
| ---------- | -------- | ---- | ---- | --------- | ---- |
| ADL        | AS       | 30   | 0    | 0         | 30   |
| ADL        | DS       | 30   | 0    | 0         | 30   |
| ADL        | DSL      | 28   | 2    | 0         | 30   |
| ADL        | DSS      | 29   | 0    | 1         | 30   |
| ADL        | NW       | 29   | 0    | 1         | 30   |
| ADL        | POG      | 30   | 0    | 0         | 30   |
| ADL        | RSS      | 29   | 0    | 1         | 30   |
| ADL        | SQ       | 30   | 0    | 0         | 30   |
| Fall       | CS       | 0    | 27   | 0         | 27   |
| Fall       | HB       | 0    | 30   | 0         | 30   |
| Fall       | ITCS     | 0    | 3    | 0         | 3    |
| Fall       | ITDS     | 2    | 28   | 0         | 30   |
| Fall       | ITRS     | 0    | 30   | 0         | 30   |
| Fall       | LCC      | 0    | 30   | 0         | 30   |
| Fall       | slip     | 0    | 26   | 4         | 30   |
| Fall       | trip     | 0    | 24   | 6         | 30   |
| Near_Fall  | CS       | 2    | 4    | 21        | 27   |
| Near_Fall  | HB       | 0    | 1    | 29        | 30   |
| Near_Fall  | ITCS     | 1    | 0    | 2         | 3    |
| Near_Fall  | ITRS     | 1    | 2    | 27        | 30   |
| Near_Fall  | slip     | 2    | 1    | 27        | 30   |
| Near_Fall  | trip     | 0    | 1    | 29        | 30   |
| All        |          | 243  | 209  | 148       | 600  |



### 使用的Conv1D-TCN模型

```python
def build_model(
    sequence_length: int,
    number_of_channels: int,
    number_of_classes: int,
    learning_rate: float,
) -> Any:
    """一个克制的小型 TCN baseline，输入形状为 [时间点, 通道]。"""
    require_tensorflow()

    inputs = tf.keras.Input(
        shape=(sequence_length, number_of_channels), name="waist_imu"
    )

    # 额外加的卷积层
    # layer1
    x = tf.keras.layers.Conv1D(32, 7, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    #x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    # layer2
    x = tf.keras.layers.Conv1D(64, 5, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dropout(0.30)(x)
    #layer3
    #x = tf.keras.layers.Conv1D(128, 3, padding="same", use_bias=False)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Activation("relu")(x)

    # 第一层
    x=ResBlock(x,filters=32,kernel_size=3,dilation_rate=1)

    # 第二层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=2)

    # 第3-N层
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=4)
    x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=8)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=16)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=32)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=64)
    #x=ResBlock(x,filters=16,kernel_size=3,dilation_rate=128)
     
    #x=tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # 输出
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="phase1_waist_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0,         # 再加一道“梯度过大就截断”的保险
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()
    return model
```
