import banditcoot.algorithms as a

a.hello_world()

def test_ind_max():
    assert a.ind_max([1,2,3,2,4.5,2.3,4]) == 4

