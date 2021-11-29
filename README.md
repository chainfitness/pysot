
## 使用PySOT

### 为终端临时添加环境变量

```bash
export PYTHONPATH=~/liuChang/pysot/pysot/psot:$PYTHONPATH
```

### 在213服务器上调用封装好的track接口运行并保存测试视频

对视频还是图片进行跟踪取决于--video选项传入的是视频文件(.mp4或.avi)还是图片(存放图片的文件夹)可以自动识别路径并进行跟踪效果和可视化。

- 进入pysot文件夹

  ```bash
  cd win-pycharm/liuChang/pysot/pysot
  ```

- 有四个自定义参数可以决定demo的运行参数。

  --config 配置文件、--snapshot 预训练权重 --video 读取的视频，不给此参数则默认使用设备摄像头  --save_path 保存测试视频的路径，不给此参数默认不保存  --sfps 保存的视频存储的fps，默认值为25.

  如：

  ```bash
  python tools/track-interface.py
  --config
  ../experiments/siamrpn_r50_l234_dwxcorr/config.yaml
  --snapshot
  ../experiments/siamrpn_r50_l234_dwxcorr/model.pth
  --video
  ../testing_dataset/got10k/GOT-10k_Test_000002
  --save_path
  ../demo/output/GOT-10k_Test_000002.avi
  
  ```

## 接口功能（编辑tracker-interface最后的几行代码）

- 默认使用预测并可视化功能，可视化完成后会打印由每一帧的预测box位置组成的列表。

- ```python
  # 想要初始化第一帧的bdox坐标（X1,Y1,X2,Y2）
  bdbox = '316,792,404,123'
  # 传入bdbox坐标实例化Tracker类并调用预测并可视化接口，若不传入bdbox进行实例化，则需手动按s选择后按空格开始跟踪。
  pred_bdbox_list = Tracker(args.video_name,bdbox).pred_n_visualization()
  print(pred_bdbox_list)
  # Tracker(args.video_name, bdbox).just_get_pred_bdbox()
  ```

- 使用无需可视化仅预测box的代码，会在工作台实时打印每一帧预测的bdbox坐标

  ```python
  # 将上面的代码注释掉，只运行这一行即可
  Tracker(args.video_name, bdbox).just_get_pred_bdbox()
  ```

  

  

### 自定义选择不同的模型

#### 可供选择模型

在--config和--snapshot选项替换路径:

例如:

--**config**

../experiments/**替换部分**/config.yaml
--**snapshot**
../experiments/**替换部分**/model.pth

- **siamrpn_alex_dwxcorr**：

优点：

1. 模型小，只有23.8M；

2. 测试视频可以达到实时检测，fps最快；

3. 目标消失一段时间后再出现也可以继续跟踪;

缺点：

1. 目标框跟踪的不精准、当目标尺寸变化时不会动态调整跟踪框；

2. 当消失的目标重新出现时，有一段反应时间才能继续跟踪。

- **siamrpn_r50_l234_dwxcorr:**

优点：

1. fps不是最快，但是也能达到实时检测的效果；

2. 目标跟踪框精准而且随目标尺寸变化可以动态调整；

缺点：

1.208M的模型权重较大；

2．fps只有35；

3.当跟踪的目标消失的那段时间，目标框存在短暂游离的情况。

- **siammask_r50_l3**：

优点：

1.82.1M的模型大小适中；

2.fps56可达到实时检测，目标消失一段时间后再出现也可以继续跟踪; 

3.目标跟踪框精准而且随目标尺寸变化可以动态调整；

4.在跟踪框内还有进行了分割，将分割和跟踪集成在一起。

缺点：无明显缺点。

- **siamrpn_mobilev2_l234_dwxcorr**：

优点：

1. 模型权重42.8M较小；

2. fps75可达到实时检测，目标消失一段时间后再出现也可以继续跟踪;

3. 目标跟踪框精准而且随目标尺寸变化可以动态调整。

缺点：

1. 目标消失的那段时间内，目标框会出现大幅的变化，例如测试中穿蓝色衣服的人消失了，目标框就往蓝色玻璃靠近；

2. 目标跟踪精度还不够，目标尺寸变换时动态调整也不精准。

## 现场测试视频

测试视频~/dataset/all-tested-video，其目录结构如下:

```


all-tested-video/
├── pysot
│   └── safety-belt
│       ├── device
│       │   ├── mobilev2
│       │   │   ├── 20210427104348.avi
│       │   │   └── 20210427144815.avi
│       │   ├── res-50
│       │   │   ├── 20210427104348.avi
│       │   │   └── 20210427144815.avi
│       │   └── res-50-lt
│       │       └── 20210427104348.avi
│       └── mobilephone
│           ├── mobilev2
│           │   ├── VID_20210427_105751.avi
│           │   └── VID_20210427_144153.avi
│           ├── res-50
│           │   ├── VID_20210427_105751.avi
│           │   └── VID_20210427_144153.avi
│           └── res-50-lt
│               ├── VID_20210427_105751.avi
│               └── VID_20210427_144153.avi
└── RPTTrack
    └── safety-belt
        ├── device
        │   ├── 20210427104348.avi
        │   ├── 20210427144815.avi
        │   └── 20210428093650.avi
        └── mobilephone
            ├── VID_20210427_105751.avi
            └── VID_20210427_144153.avi


```



其中:

**safety-belt:**代表测试视频为安全带测试视频

**device:**代表执法仪拍摄的视频

**mobilephone:**代表手机拍摄视频
