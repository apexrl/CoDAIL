# Multi-Agent Adversarial Inverse Reinforcement Learning

Frist you should install requirements.
Type the following codes under this forlder:
```
pip install -r requirements.txt
```
Then cd into the particle environment folder and type:
```
pip install -e .
```
to install the env.

Run Multi-Agent ACKTR to obtain experts:
```
python -m sandbox.mack.run_simple
python -m sandbox.mack.run_simple_om
```
The former generates interactions that agents do not consider others, while the latter generates interactions that agents model others when making decisions. Notice you should be aware of the args parameters to run it successfully.

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
