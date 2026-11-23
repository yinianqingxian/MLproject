# 团队贡献指南

A、B、C、D 的代码、实验结果和报告更新都从独立分支提交，再通过 Pull Request（PR）合入 `main`。

## 从最新的 `main` 开始

在仓库目录运行以下命令。`d/update-analysis` 是示例，请换成与你的任务对应的分支名。

```bash
git switch main
git pull --ff-only origin main
git switch -c d/update-analysis
```

如果 `git pull --ff-only` 提示本地 `main` 有分叉，先核对本地提交并把自己的改动保存在独立分支，联系队友确认如何同步。

## 检查和推送改动

- 运行与改动相关的测试或分析命令。在 PR 中填写实际命令、环境、结果以及未跑的检查。训练代码可在准备好数据和环境后运行 `python -m unittest discover -s experiments/tests -v`，DGX Spark 可用 `bash SPARK.sh tests`。
- 涉及数据和实验结果时，核对数据划分、模型版本、配置、随机种子和运行设备。说明使用了哪次实验的输出，指标属于开发验证还是独立测试，以及改动是否需要重跑。相关约定见 [训练协议](PROTOCOL.md)。
- 检查将提交的文件。凭据、个人路径、虚拟环境、原始数据和大型模型留在仓库外。结果文件只提交本次讨论需要且可以公开的内容。[项目说明](README.md)

以下命令以修改指南为例。将 `CONTRIBUTING.md` 换成实际改动的文件，将分支名和提交信息换成这次任务对应的内容。

```bash
git status --short
git diff --check
git add -- CONTRIBUTING.md
git diff --cached
git commit -m "Update contribution guide"
git push -u origin d/update-analysis
```

没有仓库写权限时，可以先 Fork 仓库，再将功能分支推到自己的 Fork。 [从 Fork 创建 PR](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request-from-a-fork)

## 发起 PR 并同步合并结果

1. 在仓库网页打开 **Pull requests → New pull request**，将 `base` 选为 `main`，将 `compare` 选为本次的功能分支。
2. 看一遍差异，填写标题和 [PR 模板](.github/pull_request_template.md)，再点击 **Create pull request**。
3. 在 **Reviewers** 里邀请熟悉该改动的队友审阅。按反馈修改后提交并推送到同一功能分支，PR 会自动更新。
4. PR 合并后，在本地同步 `main`。

```bash
git switch main
git pull --ff-only origin main
```

网页操作说明见 [GitHub PR 快速入门](https://docs.github.com/en/pull-requests/get-started/pull-request-quickstart)。
