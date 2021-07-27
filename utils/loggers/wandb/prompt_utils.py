from pathlib import Path
import sys
import getpass
import wandb
from wandb.errors import term
from wandb.util import _is_databricks, isatty

from inputimeout import inputimeout, TimeoutOccurred

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[3].as_posix())  # add yolov5/ to path

from utils.general import colorstr, emojis


TIMEOUT_CODE = -2
LOGIN_CHOICE_ENABLED = "Visualize Experiment on W&B (recommended)"
LOGIN_CHOICE_DISABLED = "Don't visualize Experiment on W&B"
LOGIN_CHOICES = [LOGIN_CHOICE_ENABLED, LOGIN_CHOICE_DISABLED]


def _prompt_choice_with_timeout():
    # If we're not in an interactive environment, default to dry-run.
    jupyter = wandb.Settings()._jupyter
    if (
        not jupyter and (not isatty(sys.stdout) or not isatty(sys.stdin))
    ) or _is_databricks():
        return 1 # LOGIN_CHOICE_DISABLED
    
    try:
        choice = inputimeout(prompt="%s: Enter your choice: " % term.LOG_STRING, timeout=15)
        return int(choice) - 1
        #return int(input("%s: Enter your choice: " % term.LOG_STRING)) - 1
    except TimeoutOccurred:
        return TIMEOUT_CODE
    except ValueError:
        return -1


def _prompt_choice():
    try:
        return int(input("%s: Enter your choice: " % term.LOG_STRING)) - 1  # noqa: W503
    except ValueError:
        return -1


def prompt_user_choices():
    for i, choice in enumerate(LOGIN_CHOICES):
        wandb.termlog("(%i) %s" % (i + 1, choice))


def get_user_choice():
    prompt_user_choices()
    choice = _prompt_choice_with_timeout()
    while choice < 0 or choice > len(LOGIN_CHOICES) - 1:
        if choice == TIMEOUT_CODE:
            wandb.termwarn('Read Timeout! You can log into W&B by running `wandb login` on your terminal')
            return None
        if choice < 0 or choice > len(LOGIN_CHOICES) - 1:
            wandb.termwarn("Invalid choice")
        choice = _prompt_choice()
    return choice


def get_api_key():
    wandb.termlog(
        "Paste your W&B API Key from here: https://wandb.ai/authorize?signup=true"
    )
    key = getpass.getpass().strip()
    return key

def setup_wandb():
    wandb.ensure_configured()
    if wandb.api.api_key:
        return True
    choice = get_user_choice()
    if choice is not None and choice == 0:
        key = get_api_key()
        wandb.login(key=key)
        return True
    else:
        print("W&B Disabled")
        return False


if __name__ == "__main__":
    wandb_enabled = setup_wandb()
