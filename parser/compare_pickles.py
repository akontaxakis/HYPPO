from parser.sub_parser import compare_pickles

if __name__ == '__main__':
    import sys
    import os

    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    pairs = compare_pickles()
    print(pairs)