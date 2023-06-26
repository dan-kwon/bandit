import banditcoot.algorithms as a

def test_param_lengths():
    algo = a.EpsilonGreedy(epsilon = 0.1, n_arms = 5, rewards = [1,2,3,4,5])
    assert len(algo.conv_rates) == len(algo.rewards) == len(algo.counts) == len(algo.values)

def test_calc_values():
    algo = a.EpsilonGreedy(epsilon = 0.1, n_arms = 3, rewards = [10,0.1,1], conv_rates = [0.1,0.2,0.3], counts = [90, 90, 90])
    assert [i*j for i,j in zip(algo.conv_rates, algo.rewards)] == algo.values

def test_update_counts():
    errors = []
    for i in range(0,3):
        algo = a.EpsilonGreedy(epsilon = 0.1, n_arms = 3, rewards = [1,1,1], conv_rates = [0.0,0.0,0.0], counts = [90, 90, 90])
        prev_counts = algo.counts.copy()
        algo.batch_update(chosen_arm=i, num_times_chosen=10, num_successes=1, observed_reward=1)
        new_counts = algo.counts.copy()
        
        if not new_counts[i] == prev_counts[i] + 10:
            errors.append(f"Error when testing arm {i}. Selected arm did not increment correctly.")
        if not [y for x,y in enumerate(prev_counts) if x!=i] == [y for x,y in enumerate(new_counts) if x!=i]:
            errors.append(f"Error when testing arm {i}. Non-selected arm incremented when it should not have.")
        assert not errors

def test_update_conv_rates():
    errors = []
    for i in range(0,3):
        algo = a.EpsilonGreedy(epsilon = 0.1, n_arms = 3, rewards = [1,1,1], conv_rates = [0.1,0.1,0.1], counts = [90, 90, 90])
        prev_conv_rates = algo.conv_rates.copy()
        algo.batch_update(chosen_arm=i, num_times_chosen=10, num_successes=6, observed_reward=1)
        new_conv_rates = algo.conv_rates.copy()
        if not new_conv_rates[i] == 0.15:
            errors.append(f"Error when testing arm {i}. Conversion rate for selected arm did not update correctly.")
        if not [y for x,y in enumerate(prev_conv_rates) if x!=i] == [y for x,y in enumerate(new_conv_rates) if x!=i]:
            errors.append(f"Error when testing arm {i}. Conversion rate for selected arm did not update correctly.")
        assert not errors

def test_update_values():
    errors = []
    for i in range(0,3):
        observed_reward = 5
        algo = a.EpsilonGreedy(epsilon = 0.1, n_arms = 3, rewards = [1,1,1], conv_rates = [0.1,0.1,0.1], counts = [50, 50, 50])
        prev_values = algo.values.copy()
        algo.batch_update(chosen_arm=i, num_times_chosen=50, num_successes=5, observed_reward=observed_reward)
        new_values = algo.values.copy()
        assert new_values[i] == 0.03
