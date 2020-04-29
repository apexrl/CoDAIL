# Multi-Agent Adversarial Inverse Reinforcement Learning

Run Multi-Agent ACKTR to obtain experts:
```
python -m sandbox.mack.run_simple
```

Run CoDAIL / NCDAIL / MA-GAIL / MA-AIRL:

```
python -m irl.mack.run_mack_gail
python -m irl.mack.run_mack_airl
python -m irl.mack.run_mack_ncdail
python -m irl.mack.run_mack_codail
```

Render results (see './irl/render.py' for more information):

```
python -m irl.render
```
