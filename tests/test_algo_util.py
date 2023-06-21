import banditcoot.algorithms as a

def test_ind_max():
    assert a.ind_max([1,2,3,2,4.5,2.3,4]) == 4

def test_epsilon_greedy_param_lengths():
    algo = a.EpsilonGreedy(epsilon = 0.1, n_arms = 5, rewards = [1,2,3,4,5])
    assert len(algo.conv_rates) == len(algo.rewards) == len(algo.counts) == len(algo.values)

def test_epsilon_greedy_select_arm():
    algo1 = a.EpsilonGreedy(epsilon = 0.1, n_arms = 5, rewards = [1,1,1,1,5], conv_rates = [0.1,0.1,0.1,0.1,0.5])
    algo2 = a.EpsilonGreedy(epsilon = 0.2, n_arms = 5, rewards = [1,1,1,1,5], conv_rates = [0.1,0.1,0.1,0.1,0.5])
    algo3 = a.EpsilonGreedy(epsilon = 0.3, n_arms = 5, rewards = [1,1,1,1,5], conv_rates = [0.1,0.1,0.1,0.1,0.5])

    algo1_observed_epsilon = sum([algo1.select_arm()!=4 for i in range(10000)]) / len(range(10000))
    algo2_observed_epsilon = sum([algo2.select_arm()!=4 for i in range(10000)]) / len(range(10000))
    algo3_observed_epsilon = sum([algo3.select_arm()!=4 for i in range(10000)]) / len(range(10000))
        