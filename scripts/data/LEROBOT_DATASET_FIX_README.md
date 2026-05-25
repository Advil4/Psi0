# LeRobot 数据集修复脚本

## 概述

这个脚本 (`fix_lerobot_dataset.py`) 用于将 LeRobot 数据集转换为与训练框架兼容的格式。

## 使用方法

### 基本用法

```bash
cd /mnt/data0/xhx/Psi0

# 修复默认数据集 (Pick_up_an_apple)
bash scripts/data/fix_lerobot_dataset.sh

# 或指定数据集名称
bash scripts/data/fix_lerobot_dataset.sh Your_Dataset_Name

# 直接使用 Python 脚本
python scripts/data/fix_lerobot_dataset.py --dataset Your_Dataset_Name
```

### 运行训练

修复完成后，可以直接运行训练：

```bash
bash scripts/train/psi0/finetune-real-psi0.sh Your_Dataset_Name
```

## 修复步骤详解

脚本按顺序执行以下 9 个步骤：

### Step 1: 转换 episodes.jsonl

- **输入**: `meta/episodes.jsonl` (原始格式)
- **输出**: `meta/episodes.jsonl` (LeRobot 兼容格式)
- **备份**: `meta/episodes_original.jsonl`
- **主要修改**:
  - 将任务名称转换为任务索引 (task_index)
  - 简化字段结构，移除冗余统计信息
  - 提取 chunk_index 和 file_index

### Step 2: 生成 episodes_stats.jsonl

- **来源**: 从备份的 episodes_original.jsonl 提取
- **内容**: 每个 episode 的 action 和 timestamp 统计信息
- **格式**: 每行一个 episode 的统计数据

### Step 3: 生成 modality.json

- **内容**: 定义 state、action、video 的维度映射
- **示例**:
  ```json
  {
    "state": {"names": ["dim"], "shape": [36], "dtype": "float32"},
    "action": {"names": ["dim"], "shape": [36], "dtype": "float32"},
    "video": {"names": ["c", "h", "w"], "shape": [3, 480, 640], "dtype": "uint8"}
  }
  ```

### Step 4: 生成 relative_stats.json

- **内容**: 空的相对动作统计对象 `{}`
- **用途**: PSI0 模型可能需要此文件（即使为空）

### Step 5: 生成 stats_psi0.json

- **来源**: 从所有 episodes 的 action 统计中计算全局 min/max
- **内容**: 
  ```json
  {
    "action": {"min": [...], "max": [...]},
    "states": {"min": [...], "max": [...]}
  }
  ```

### Step 6: 修复 parquet metadata

- **问题**: HuggingFace Datasets 不支持 `List` 类型
- **修复**: 将所有 `List` 类型改为 `Sequence` 类型
- **影响**: 所有 116 个 parquet 文件的 huggingface metadata

### Step 7: 重命名 parquet 列

- **映射**: `observation.state` → `states`
- **原因**: 训练配置 (`finetune-real-psi0.sh`) 中指定 `stat-state-key=states`
- **影响**: 所有 116 个 parquet 文件的列名

### Step 8: 更新 info.json

主要修改包括：

1. **路径模板**:
   ```json
   "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
   "video_path": "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4"
   ```

2. **特征名称**:
   - `observation.image` → `observation.images.egocentric`
   - `observation.state` → `states`

3. **视频 shape**:
   - 从 `[3, 480, 640]` (CHW) 改为 `[480, 640, 3]` (HWC)
   - names 从 `["c", "h", "w"]` 改为 `["height", "width", "channel"]`

4. **video_info 格式**:
   ```json
   "video_info": {
     "video.fps": 30.0,
     "video.codec": "av1",
     "video.pix_fmt": "yuv420p",
     "video.is_depth_map": false,
     "has_audio": false
   }
   ```

5. **添加缺失字段**:
   - `total_videos`: 116
   - `total_chunks`: 1

### Step 9: 重命名视频文件夹

- **操作**: `videos/chunk-*/observation.image/` → `videos/chunk-*/egocentric/`
- **原因**: 与 info.json 中的 video_path 保持一致

## 技术细节

### LeRobot v2.1 格式要求

1. **info.json 占位符**: 必须使用 `episode_chunk` 和 `episode_index`
2. **parquet 文件名**: 必须是 `episode_{index:06d}.parquet` 格式
3. **视频路径**: 必须与 info.json 中的 video_path 完全匹配
4. **特征类型**: 数组类型必须使用 `Sequence` 而非 `List`

### 列名约定

根据项目记忆，LeRobot 数据集的标准列名约定：
- **State**: `states` (不是 `observation.state`)
- **Action**: `action`
- **Video**: `observation.images.egocentric` (或其他摄像头名称)

### 训练配置匹配

`finetune-real-psi0.sh` 中的关键配置：
```bash
--data.transform.field.stat-state-key=states
--data.transform.field.stat-action-key=action
```

这些配置决定了数据集必须使用的列名。

## 故障排除

### 如果训练失败

1. **检查文件是否存在**:
   ```bash
   ls -la real/Pick_up_an_apple/meta/
   ls -la real/Pick_up_an_apple/data/chunk-000/*.parquet | head -5
   ls -la real/Pick_up_an_apple/videos/chunk-000/egocentric/ | head -5
   ```

2. **验证 info.json**:
   ```bash
   python -c "import json; print(json.dumps(json.load(open('real/Pick_up_an_apple/meta/info.json')), indent=2))"
   ```

3. **检查 parquet 列名**:
   ```bash
   python -c "
   import pyarrow.parquet as pq
   schema = pq.read_schema('real/Pick_up_an_apple/data/chunk-000/episode_000000.parquet')
   print('Columns:', schema.names)
   "
   ```

4. **重新运行修复脚本**:
   ```bash
   python scripts/data/fix_pickup_apple_dataset.py
   ```

### 常见错误

1. **KeyError: 'chunk_index'**: info.json 的 data_path 占位符不正确
2. **ValueError: Feature type 'List' not found**: parquet metadata 需要修复
3. **KeyError: Column states not in dataset**: parquet 列名需要重命名
4. **KeyError: Column observation.images.egocentric not in dataset**: 特征名称或视频路径不正确

## 注意事项

1. **备份**: 脚本会自动备份原始 episodes.jsonl
2. **幂等性**: 脚本可以多次运行，但建议只运行一次
3. **依赖**: 需要 `pyarrow`, `numpy` 库
4. **时间**: 处理 116 个文件大约需要 1-2 分钟

## 参考

- LeRobot 文档: https://github.com/huggingface/lerobot
- 参考数据集: `Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw`
- 训练脚本: `scripts/train/psi0/finetune-real-psi0.sh`
