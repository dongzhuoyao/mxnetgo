# Author: Tao Hu <taohu620@gmail.com>
import sys,os
class classb():
    def __init__(self):
        print("classb init")

    def hh(self):
        print ("hh")

def say_basename():
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    print basename