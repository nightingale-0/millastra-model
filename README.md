# millastra-model

在《原神》千星奇域的客户端 Lua 沙箱里离线运行一个 2600 万参数的语言模型，做成一个可以聊天的派蒙（关卡「TeyvatAI 派蒙」）。

- 基座：[MiniMind2-Small](https://huggingface.co/jingyaogong/MiniMind2-Small)（Llama 结构，26M，Apache-2.0）
- 4 bit 量化感知训练，权重以文本形式嵌在一个约 18.8MB 的客户端脚本里（关卡存档上限 20MB）
- 纯 Lua 推理（无 JIT / SIMD / GPU / `require`），按帧切片；真机首字约 1 s（打字时预填），生成约 2.5 字/秒
- 派蒙人设靠教师模型（DeepSeek）蒸馏；冷门设定靠离线检索（RAG）；自制拼音输入法；问题过滤

评测结果见 [docs/results.md](docs/results.md)。

## 架构

```
游戏文本 + wiki ──► 继续预训练 ──► SFT（派蒙对话 + 设定 + 检索样本…）──► 4bit QAT ──► 导出 Lua 表
                                                                                   │
客户端脚本（一个文件）：界面 · 拼音输入法 · 问题过滤 · 词条检索 · BPE 分词 · 推理引擎 · 权重
```

推理端要点（`lua/minimind.lua`）：

- **位平面查表矩阵乘**：每 8 个输入预先算好 256 种子集和，4 bit 权重每 8 个只需 4 次查表，没有逐权重乘法
- **按轮前缀缓存**：人设 system 轮和推荐问题的 K/V、首步 logits 在导出时算好
- **草稿预填**：玩家打字时就把已输入的部分喂进模型，发送时只剩 7 个模板 token
- **GQA 合并注意力**、B=2..8 的专用 prefill 内核、回答内 6-gram 去重
- 帧预算按游戏空闲帧耗时自适应；所有入口在 `xpcall` 下运行，出错只结束当前回答

## 目录

| 路径 | 内容 |
|---|---|
| `data_prep/` | 游戏对话表 → 派蒙对话 SFT 集；继续预训练语料 |
| `distill/` | 教师数据生成、SFT 集构建、评审（`judge.py`）、分类探查（`probe_suite.py`）、wiki 词条抓取与资料生成 |
| `train/` | SFT / 继续预训练（`sft.py`）、权重平均（`soup.py`）、完整流程（`pipeline.sh`） |
| `quant/` | 4 bit 量化感知训练（`qat.py`）、8 bit log 块缩放 |
| `export/` | 检查点 → Lua 权重表、分词表、前缀缓存、golden 测试数据 |
| `lua/` | Lua 推理运行时、BPE 分词器、golden / 草稿预填测试、各项基准 |
| `game/` | 关卡：界面、输入法、过滤器、检索表、打包与模拟器测试（见 [game/README.md](game/README.md)） |
| `bench/` | 投机解码可行性模拟、轮次汇总 |

## 复现

```bash
npm install                                   # wasmoon / fengari（Lua 测试）
pip install torch transformers safetensors pypinyin regex
# DeepSeek key 放在 deepseek.env（API_KEY=...，已 gitignore）
bash train/pipeline.sh                        # 各阶段说明见脚本注释
```

需要自备（不在仓库里）：基座模型 `models/MiniMind2-Small/`、游戏文本表 `data/agd/`（AnimeGameData）、
模拟器 miliastra-beyond-simulator（`QXQY_SIM` 指向它）。

## 千星沙箱的坑

- 没有 `require`，所有模块打包进一个脚本；全局表和标准库只读（写入要包 `pcall`）
- `table.concat` 结果有大小上限：大块数据按约 48KB 分段，永不拼接
- `collectgarbage`、`_VERSION` 不可用；`string.format('%d')` 遇到非整数会报错
- 关卡审核会扫描脚本文本：敏感词只存哈希，权重用避开元音的 base85 编码，偶然拼出的英文词用 `=` 打断
- 导出在 Windows 上要用 `PYTHONIOENCODING=utf-8`，否则打印 golden 时崩溃、留下旧权重

## 数据与版权

仓库只含代码。游戏文本、wiki 抓取内容、教师生成的数据、模型权重和关卡封面素材均不包含。
游戏内容版权归 miHoYo 所有；wiki 内容来自 B 站原神 wiki，请遵守其许可。
