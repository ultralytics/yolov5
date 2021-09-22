---
name: "🐛 Bug report"
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

Before submitting a bug report, please be aware that your issue **must be reproducible** with all of the following,
otherwise it is non-actionable, and we can not help you:

- **Current repo**: run `git fetch && git status -uno` to check and `git pull` to update repo
- **Common dataset**: coco.yaml or coco128.yaml
- **Common environment**: Colab, Google Cloud, or Docker image. See https://github.com/ultralytics/yolov5#environments

If this is a custom dataset/training question you **must include** your `train*.jpg`, `val*.jpg` and `results.png`
figures, or we can not help you. You can generate these with `utils.plot_results()`.

## 🐛 Bug

A clear and concise description of what the bug is.

## To Reproduce (REQUIRED)

Input:

```
import torch

a = torch.tensor([5])
c = a / 0
```

Output:

```
Traceback (most recent call last):
  File "/Users/glennjocher/opt/anaconda3/envs/env1/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 3331, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-5-be04c762b799>", line 5, in <module>
    c = a / 0
RuntimeError: ZeroDivisionError
```

## Expected behavior

A clear and concise description of what you expected to happen.

## Environment

If applicable, add screenshots to help explain your problem.

- OS: [e.g. Ubuntu]
- GPU [e.g. 2080 Ti]

## Additional context

Add any other context about the problem here.
