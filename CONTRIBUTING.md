## Contributing to YOLOv5 🚀

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing a new feature
- Becoming a maintainer

YOLOv5 works so well due to our combined community effort, and for every small improvement you contribute you will be helping push the frontiers of what's possible in AI 😃!


## Submitting a Pull Request (PR) 🛠️

To allow your work to be integrated as seamlessly as possible, we advise you to:
- ✅ Verify your PR is **up-to-date with origin/master.** If your PR is behind origin/master an automatic [GitHub actions](https://github.com/ultralytics/yolov5/blob/master/.github/workflows/rebase.yml) rebase may be attempted by including the /rebase command in a comment body, or by running the following code, replacing 'feature' with the name of your local branch:
```bash
git remote add upstream https://github.com/ultralytics/yolov5.git
git fetch upstream
git checkout feature  # <----- replace 'feature' with local branch name
git merge upstream/master
git push -u origin -f
```
- ✅ Verify all Continuous Integration (CI) **checks are passing**.
- ✅ Reduce changes to the absolute **minimum** required for your bug fix or feature addition. _"It is not daily increase but daily decrease, hack away the unessential. The closer to the source, the less wastage there is."_  -Bruce Lee


## Submitting a Bug Report 🐛

For us to investigate an issue we would need to be able to reproduce it ourselves first. We've created a few short guidelines below to help users provide what we need in order to get started investigating a possible problem.

When asking a question, people will be better able to provide help if you provide **code** that they can easily understand and use to **reproduce** the problem. This is referred to by community members as creating a [minimum reproducible example](https://stackoverflow.com/help/minimal-reproducible-example). Your code that reproduces the problem should be:

* ✅ **Minimal** – Use as little code as possible that still produces the same problem
* ✅ **Complete** – Provide **all** parts someone else needs to reproduce your problem in the question itself
* ✅ **Reproducible** – Test the code you're about to provide to make sure it reproduces the problem

In addition to the above requirements, for [Ultralytics](https://ultralytics.com/) to provide assistance your code should be:

* ✅ **Current** – Verify that your code is up-to-date with current GitHub [master](https://github.com/ultralytics/yolov5/tree/master), and if necessary `git pull` or `git clone` a new copy to ensure your problem has not already been resolved by previous commits.
* ✅ **Unmodified** – Your problem must be reproducible without any modifications to the codebase in this repository. [Ultralytics](https://ultralytics.com/) does not provide support for custom code ⚠️.

If you believe your problem meets all of the above criteria, please close this issue and raise a new one using the 🐛 **Bug Report** [template](https://github.com/ultralytics/yolov5/issues/new/choose) and providing a [minimum reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) to help us better understand and diagnose your problem. 


## License

By contributing, you agree that your contributions will be licensed under the [GPL-3.0 license](https://choosealicense.com/licenses/gpl-3.0/)
