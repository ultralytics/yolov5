# GitHub Release Process

## Steps

1. Update the version in `yolov5/version.py`.

3. Run the release script:

    ```bash
    ./scripts/release.sh
    ```

    This will commit the changes to the CHANGELOG and `version.py` files and then create a new tag in git
    which will trigger a workflow on GitHub Actions that handles the rest.

## Fixing a failed release

If for some reason the GitHub Actions release workflow failed with an error that needs to be fixed, you'll have to delete both the tag and corresponding release from GitHub. After you've pushed a fix, delete the tag from your local clone with

```bash
git tag -l | xargs git tag -d && git fetch -t
```

Then repeat the steps above.
