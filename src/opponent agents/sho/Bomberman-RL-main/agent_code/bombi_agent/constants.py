import os

t_policy = 'epsgreedy'
t_policy_epsilon = 0.15

# Construct unique id for naming of persistent data.
t_training_id = "{}_{}_{}".format(t_policy, t_policy_epsilon, "bestguess")

# Save weights to file
print("TRAINING ID:", t_training_id)
cwd = os.getcwd()
weights_file = os.path.join(cwd, "{}_weights.npy".format(t_training_id))
