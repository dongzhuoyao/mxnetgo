# Author: Tao Hu <taohu620@gmail.com>

import sys,os


if __name__ == '__main__':
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    print basename