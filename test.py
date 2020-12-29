from global_search_BO.es_interface import es_loop


print(es_loop)


def f(*arg):
    for i in arg:
        print(i)


f(*[1,2,3])