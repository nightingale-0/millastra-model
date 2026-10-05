# TeyvatAI 派蒙 · 关卡

一个客户端控件模板 + 一个客户端脚本（界面、输入法、过滤器、检索、分词器、推理引擎、权重全在里面）+ 服务端控件组上一个很小的 boot 脚本（实例化模板）。

## 构建与测试

```bash
python game/make_ime.py                       # 拼音词库 -> game/build/ime_dict.lua
python game/embed/train_filter.py             # 问题过滤器 -> game/build/filter.lua
python game/make_lore.py                      # 检索词条表 -> game/build/lore.lua
node game/build.mjs                           # 界面测试版（假模型）
LORE=1 TEMPLATE_ID=<模板ID> node game/build.mjs --full --install   # 完整版，复制到编辑器导入目录

node game/test.mjs [--full]                   # 模拟器点击 / 按键 / 截图 / 布局检查；日志里出现 [paimon] error 即失败
node game/filter_test.mjs                     # Lua 过滤器与检索 vs Python 参考
node game/sandbox_check.mjs game/out/scripts/paimon-chat-full.main.lua   # 只读全局 / 标准库
python game/scan_text.py game/out/scripts/paimon-chat-full.main.lua      # 审核敏感文本扫描，必须 clean
```

## 导入游戏

1. 编辑器导入 `paimon-chat-full.gia`，模板 ID 会重新分配（`node game/gil_peek.mjs <关卡.gil>` 可查）
2. 用新 ID 重新构建，把 `game/out/scripts/` 里的 main / boot 脚本覆盖到编辑器映射的脚本文件（保留 UTF-8 BOM）
3. 只改脚本、不改控件时，直接覆盖 main 脚本即可，不必重新导入

**按键**：关卡设置里把自定义按键 1–26 绑定 A–Z，27 删除，28 空格（选字 / 英文空格），29 回车（发送）。

**结算**：进入 30 秒后客户端发信号 `PaimonWin`，点「结算」发 `PaimonSettle`。服务端节点图：两个信号各接
【获取在场玩家实体列表】→【列表迭代循环】→【设置玩家结算成功状态 = 胜利】，`PaimonSettle` 在循环完成后接【结算关卡】。

## 功能

- 对话：多轮上下文 512 token，超出时压缩最早的轮次；「回退」到某一轮；推荐问题（首问零 prefill）
- 等待提示：「派蒙思考中…（x秒）」/「派蒙正在查阅资料…（x秒）」
- 输入法：全拼、首字母、音节自动切分、中 / 英切换
- 过滤：字 n-gram 线性分类器，阈值随句长从 0.97（短句）降到 0.7（长句），短句主要靠关键词哈希
- 检索：问题在问某个词条（或词条名作主语）时，最多带两条十几个字的资料；顺带提到的名字、推荐问题、
  七国和模型本来就答得对的词条（`distill/known_entries.py`）不触发
- 深色模式、30 秒结算按钮
