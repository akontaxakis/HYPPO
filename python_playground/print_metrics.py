from libs.sub_parser import print_metrics, compute_loading_times

if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")


    print_metrics()
    lt=compute_loading_times()
    print(lt)