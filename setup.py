from setuptools import setup

setup(name='yolov5',
      version='1.0.1',
      author="Psycle Research",
      description="Fork of yolov5",
      url="https://github.com/PsycleResearch/yolov5",
      packages=['yolov5', 'yolov5.models', 'yolov5.utils', 'yolov5.data'],
      python_requires='>=3.6'
      )
