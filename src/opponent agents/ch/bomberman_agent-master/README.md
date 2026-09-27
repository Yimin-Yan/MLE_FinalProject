# bomberman_agent
A reinforcement learning agent for the Bomberman framework found at https://github.com/ukoethe/bomberman_rl
using Q-learning with gradient boosted trees and an ε-greedy exploration-exploitation strategy, created as a final project within the scope of the lecture "Fundamentals of Machine Learning" at the Heidelberg University.

The folder "agent_code" contains three versions of the agent:
1. coinllector is coin collecting agent, developed for an arena without crates or enemies
2. cratemate is an advanced agent, able to place bombs in order to destroy crates, possibly hiding coins
3. cubi_bot is the final agent supposed to compete against 3 enemies

The first agent furthermore contains a "backup" subfolder including a good sample dataset,
recorded logs across ten test games and the corresponding replays. The second and third agent only contain backups of sample datasets.

The bash script "n_rounds.sh", when placed inside the "logs" folder of an agent, outputs the current round in the terminal every 60 seconds in order to keep track of the training progress.

Inside the folder "report" you can find the report written for the project, including our approaches, experiments and results. My part of the report can be read in "my_report.pdf". The final report, put together by my partner, who is also responsible for the final version of cubi_bot, can be read in "report_final.pdf".
