import fire

from .detect import run as detect
from .export import run as export
from .train import run as train
from .val import run as val


def main():
  fire.Fire({
      'train': train,
      'detect': detect,
      'val': val,
      'export': export
  })

if __name__=="__main__":
    main()
