from .train import run as train
from .detect import run as detect

import fire


def main():
  fire.Fire({
      'train': train,
      'detect': detect
  })

if __name__=="__main__":
    main()