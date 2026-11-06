## Evaluation (100 greedy episodes per agent, seed=0)

| Agent | Success | Avg steps | Avg reward | Avg collisions |
|---|---|---|---|---|
| Random | 17% | 188.0 | -652.0 | 53.5 |
| Q-learning | 100% | 18.0 | 83.0 | 0.0 |
| A* | 100% | 18.0 | 83.0 | 0.0 |

## Q-learning training convergence
- Episodes trained: 1000
- Reward curve flattened (50-ep average within 5 of final) at episode: 473
- Collisions per episode: first 100 eps = 24.6, last 100 eps = 0.2, total over training = 3395
- A* optimal path length: 18 steps
