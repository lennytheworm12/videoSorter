# Incremental GitHub workflow

videoSorter project work belongs on focused project branches. Retired course
notebooks, reports, generated exports, and homework-only tests do not belong in
the repository.

## Start from the remote main branch

```bash
git switch main
git pull --ff-only
git switch -c agent/<short-task-name>
```

Use one branch for one coherent task. Do not resume development from an old
`homework/*` branch.

## Commit and push logical checkpoints

Before each commit:

```bash
git status --short
git diff --check
```

Run the tests relevant to the files changed, then stage only that logical
slice with explicit paths:

```bash
git add path/to/file path/to/test
git commit -m "Describe the completed slice"
```

Publish the first checkpoint and establish tracking:

```bash
git push -u origin "$(git branch --show-current)"
```

For later validated checkpoints on the same task:

```bash
git push
```

Open a draft pull request early. Continue pushing small, tested commits to the
same branch so GitHub records progress incrementally and the pull request stays
reviewable.

## Keep local and retired material out of Git

Never stage `.env` files, credentials, cookies, authentication state, local
databases, parser assets, browser caches, generated build output, or retired
course deliverables. Check the staged diff before every commit:

```bash
git diff --cached --stat
git diff --cached
```

After a pull request is merged, update local main with a fast-forward pull and
start the next task from a fresh branch.
