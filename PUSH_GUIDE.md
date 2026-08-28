# 推送到公开 GitHub 仓库 — 操作指引

本仓库已在本地初始化完成，随时可推送。本文件说明为什么需要一次手动操作，
以及三种推送方式。

## 本地仓库现状

- 分支：`main`
- 提交数：3
- 文件数：124（7.6 MB）
- 工作区：干净，无未提交内容
- 已排除（见 `.gitignore`）：`data/raw/`（原始图片 550 MB）、
  `experiments/checkpoints/*.pth`（模型权重 147 MB）

## 为什么需要你先手动创建一个空仓库

WorkBuddy 的 GitHub 连接器是一个 **GitHub App（integration）**，权限按项
授予。它具备 `Contents: write`（向已存在的仓库写文件）与读权限，但**不具备**
`Administration: write`（创建仓库）。

调用 `POST /user/repos` 创建仓库时返回：

```
403 Resource not accessible by integration
```

诊断依据：对不存在的仓库调用 `push_files`，返回的是 `404 Not Found`
（仓库不存在）而非 `403`（无权限）——证明写文件的通道是通的，唯一的卡点是
"仓库必须先存在"。

因此在 github.com 上手动创建一次空仓库（约 30 秒），即可打通后续流程。

---

## 方式 A：命令行 + Personal Access Token（推荐，保留完整 git 历史）

### 1. 创建空仓库

打开 <https://github.com/new> ，按下表填写：

| 字段 | 值 |
|------|-----|
| Repository name | `saint-george-classifier` |
| Description | Binary image classifier for detecting Saint George |
| 可见性 | **Public** |
| Add a README file | **不勾选**（否则会产生冲突的首次提交） |
| Add .gitignore | **不勾选** |
| Choose a license | 可留空，或选 MIT |

点击 **Create repository**。

### 2. 生成 Token

<https://github.com/settings/tokens> → **Generate new token (classic)**：

- 勾选 `repo`（完整仓库读写）
- 勾选 `workflow`（若后续要加 GitHub Actions）
- 生成后复制，**只显示一次**

### 3. 推送

```bash
cd "C:/Users/16610/Desktop/.workbuddy/.workbuddy/saint_george_classifier"

# 添加远端（把 <TOKEN> 换成刚才复制的 token）
git remote add origin https://1661032942:<TOKEN>@github.com/1661032942/saint-george-classifier.git

# 推送（保留 3 次提交的完整历史）
git push -u origin main
```

推送完成后验证：

```bash
git remote -v          # 确认远端地址
git log --oneline -3   # 确认提交
```

浏览器打开 `https://github.com/1661032942/saint-george-classifier` 应看到
124 个文件、3 次提交。

---

## 方式 B：GitHub CLI

若本机已安装 [gh](https://cli.github.com/) 并登录：

```bash
cd "C:/Users/16610/Desktop/.workbuddy/.workbuddy/saint_george_classifier"
gh auth login                                    # 首次使用需登录
gh repo create saint-george-classifier --public --source=. --remote=origin --push
```

一条命令完成创建 + 关联 + 推送。

---

## 方式 C：让 WorkBuddy 代传（无需 token）

仓库创建完成后，直接告诉 WorkBuddy "仓库已建好"，它会用连接器的
`push_files` 工具批量推送。**前提**：

- 仓库必须是**空的**（不要勾选 README/LICENSE 初始化）
- 用户名与仓库名需告知（默认 `1661032942/saint-george-classifier`）

注意：该方式走 GitHub Contents API，逐个文件提交，**不保留本地的 3 次
提交历史**（会合并为单次提交）。若在意提交历史的颗粒度，请用方式 A。

---

## 常见问题

**Q：推送时提示 `src refspec main does not match any`？**
本地分支是 `master` 而非 `main`。改推 `git push -u origin master`，
或先重命名：`git branch -M main`。

**Q：提示 `failed to push some refs`（远端有内容）？**
创建仓库时勾选了 README/LICENSE，导致远端已有提交。先拉取合并：
`git pull origin main --allow-unrelated-histories`，再 `git push -u origin main`。

**Q：想重新关联远端？**
```bash
git remote remove origin
git remote add origin <新的远端地址>
```

**Q：模型权重和原始图片没推送，别人怎么复现？**
这是刻意设计的——550 MB 原始图片和 147 MB 权重不适合进 git。复现方式见
`README.md`：运行 `scripts/prepare_data.py` 从两个 zip 重建数据，运行
`scripts/train.py` 重新训练。

**Q：Token 担心泄露？**
推送后可在 <https://github.com/settings/tokens> 删除该 token；
或改用 SSH 方式（生成密钥后上传公钥，见 GitHub 官方文档）。
