# LeRobot 数据集修复 - 快速参考

## 🚀 快速开始

### 方法 1: 使用 Bash 脚本（推荐）

```bash
cd /mnt/data0/xhx/Psi0

# 修复默认数据集
bash scripts/data/fix_lerobot_dataset.sh

# 或指定数据集名称
bash scripts/data/fix_lerobot_dataset.sh Your_Dataset_Name
```

### 方法 2: 直接使用 Python 脚本

```bash
cd /mnt/data0/xhx/Psi0
python scripts/data/fix_lerobot_dataset.py --dataset Your_Dataset_Name
```

### 运行训练

```bash
bash scripts/train/psi0/finetune-real-psi0.sh Your_Dataset_Name
```

## 📁 创建的文件

| 文件 | 用途 |
|------|------|
| `scripts/data/fix_lerobot_dataset.py` | 主修复脚本（Python） |
| `scripts/data/fix_lerobot_dataset.sh` | Bash 包装器脚本 |
| `scripts/data/LEROBOT_DATASET_FIX_README.md` | 详细文档 |
| `scripts/data/LEROBOT_DATASET_QUICK_REF.md` | 本快速参考 |

## 🔧 修复内容

脚本自动执行以下 9 个步骤：

1. ✅ 转换 episodes.jsonl 为 LeRobot 格式
2. ✅ 生成 episodes_stats.jsonl
3. ✅ 生成 modality.json
4. ✅ 生成 relative_stats.json
5. ✅ 生成 stats_psi0.json
6. ✅ 修复 parquet metadata (List → Sequence)
7. ✅ 重命名 parquet 列 (observation.state → states)
8. ✅ 更新 info.json (特征名、路径、shape)
9. ✅ 重命名视频文件夹 (observation.image → egocentric)

## ⚠️ 注意事项

- **备份**: 原始 episodes.jsonl 会自动备份为 episodes_original.jsonl
- **幂等性**: 脚本可以安全地多次运行
- **时间**: 处理 116 个文件约需 1-2 分钟
- **依赖**: 需要 `pyarrow` 和 `numpy`

## 🐛 常见问题

### Q: 训练时出现 KeyError？

A: 确保已运行修复脚本，然后检查：
```bash
# 验证列名
python -c "import pyarrow.parquet as pq; print(pq.read_schema('real/Pick_up_an_apple/data/chunk-000/episode_000000.parquet').names)"

# 验证 info.json
python -c "import json; info=json.load(open('real/Pick_up_an_apple/meta/info.json')); print('Features:', list(info['features'].keys()))"
```

### Q: 如何重新运行修复？

A: 直接再次运行脚本即可：
```bash
bash scripts/data/fix_pickup_apple.sh
```

### Q: 其他数据集需要修复吗？

A: 这个脚本是专门为 `Pick_up_an_apple` 数据集定制的。其他数据集可能需要不同的修复步骤。

## 📚 更多信息

详细文档请查看：
- `scripts/data/FIX_PICKUP_APPLE_README.md`

技术细节请参考：
- LeRobot 文档: https://github.com/huggingface/lerobot
- 项目记忆: LeRobot 数据集列名约定

## ✨ 成功标志

修复成功后，你会看到：

```
==========================================
✓ All fixes completed successfully!
==========================================

Generated/Fixed files:
  episodes.jsonl                     - Converted to LeRobot format
  episodes_original.jsonl            - Backup of original
  episodes_stats.jsonl               - Per-episode statistics
  modality.json                      - Modality configuration
  relative_stats.json                - Relative action stats
  stats_psi0.json                    - PSI0 format statistics
  info.json                          - Updated with correct paths/names
  Parquet files                      - Fixed metadata and column names
  Video folders                      - Renamed to 'egocentric'

You can now run training:
  bash scripts/train/psi0/finetune-real-psi0.sh Your_Dataset_Name
```

---

**最后更新**: 2026-05-25
**版本**: 1.0
