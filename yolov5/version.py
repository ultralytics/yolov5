_MAJOR = "0"
_MINOR = "1"
# On main and in a nightly release the patch should be one ahead of the last
# released build.
_PATCH = "0"
# This is mainly for nightly builds which have the suffix ".dev$DATE". See
# https://semver.org/#is-v123-a-semantic-version for the semantics.
_SUFFIX = ""

VERSION_SHORT = f"{_MAJOR}.{_MINOR}"
VERSION = f"{_MAJOR}.{_MINOR}.{_PATCH}{_SUFFIX}"
