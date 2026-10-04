# 原始数据放置

将比赛提供的两份 CSV 按下列名称放到本目录：

```text
data/raw/训练集.csv
data/raw/测试集_X.csv
```

本地初始化时已经将原文件夹 `赛题五数据/` 中的两个文件移动到这里。文件内容保持原样。原始 CSV 不进入 Git；另一位成员克隆仓库后需自行复制比赛原始文件到本目录。

可用 PowerShell 核对文件校验值，并与 `data/manifest.json` 中记录的 SHA-256 比较：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'data/raw/训练集.csv'
Get-FileHash -Algorithm SHA256 -LiteralPath 'data/raw/测试集_X.csv'
```

不得把隐藏测试标签或自行生成的数据混入这两份原始文件。清洗结果和特征保存在 `data/processed/`。
