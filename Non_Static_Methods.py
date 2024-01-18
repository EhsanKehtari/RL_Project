import inspect


def get_non_static_methods_instances(obj):
    """
    Get all non-static methods of a given class, except __init__.
    :param obj: the obj to be investigated.
    :return: a list containing all non-static methods of a given class (__init__ excluded)
    """
    non_static_methods = [method for method in dir(obj)
                          if inspect.ismethod(getattr(obj, method)) and
                          not inspect.ismethod(getattr(obj, method).__func__) and
                          not method == '__init__']
    return [getattr(obj, method) for method in non_static_methods]

