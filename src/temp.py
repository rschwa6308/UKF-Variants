from Systems import AutoDiffSystemModel


def f_new(...):
    ...

def h_new(...):
    ...


NL_system = AutoDiffSystemModel(3, 3, 3, f_new, h_new)


