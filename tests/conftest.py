
_REQUIRED_ARGUMENTS = ('weights',)


def pytest_addoption(parser):
    for argument in _REQUIRED_ARGUMENTS:
        parser.addoption(f'--{argument}', default=None)
