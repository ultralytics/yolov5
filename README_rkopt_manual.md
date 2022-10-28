# YOLOv5 - rkopt 仓库

- 基于 https://github.com/ultralytics/yolov5 代码修改，设配 rknpu 设备的部署优化
- 切换分支 git checkout {分支名}
- 目前支持分支:
  - master
    - maxpool/ focus 优化，输出改为个branch分支的输出。以上优化代码使用插入宏实现，不影响原来的训练逻辑，这个优化兼容修改前的权重，故支持官方给的预训练权重。

    - 修改激活函数 silu 为 relu

    - 训练的相关内容请参考 README.md 说明

    - 导出模型时  python export.py --rknpu {rk_platform} 即可导出优化模型

      (rk_platform支持 rk1808, rv1109, rv1126, rk3399pro, rk3566, rk3568, rk3588, rv1103, rv1106)

