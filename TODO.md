# TODO

## Simplicity cutover

- [x] one common Isaac scene builder and asset physical metadata
- [x] one asset method shared by task goals
- [x] task goal, initial distribution and episode limit, plus necessary Isaac overrides
- [x] one screw specification for reset, mimic and expert motion
- [x] bounded M30 regrasp cycle without repeated recipe definitions
- [x] shared defaults without per-asset identity, zero and false clutter
- [x] one robot home, contact binding and actuator-group definition
- [x] compare all eight resolved goals, joint physics, timing and canonical target geometry
- [x] final Franka ball and ANYmal M30 bank/replay sanity runs after canonical cutover
- [x] remove semantic classes, scene aliases and actuator binding boilerplate
- [x] consistent asset section ordering and short YAML headings
- [x] one mechanism-relative panel pose and reset path
- [x] task start drives reset without duplicated joint state
- [x] incorporate expert procedures, execution settings and bank diversity in Python
- [x] incorporate shared task pose ranges and zero joint-offset defaults
- [x] incorporate method-specific observation, reward and tracking wiring

## Geometry and method

- [x] canonical TCP geometry shared by expert targets and RL measurements
- [x] remove tool compensation while preserving upright and grasp-symmetry selection
- [ ] isolate reported quaternion-sign sensitivity with controlled equivalent inputs
- [ ] verify each method's supported goals before adding opposite-direction tasks
- [ ] justify M30 regrasp budget and task-specific completion tolerances
- [ ] full training and held-out evaluation per robot and task
- [ ] Franka small valve and M30 had no successful experts, breaker had 6 percent
- [ ] diagnose failed Franka methods before assuming robot-only tuning is sufficient

## Physics

- [x] common authored contact-material ranges across expert and RL
- [x] small-valve solver iterations shared by expert, training and play
- [x] unified collision-disabled panel attachment and full-pose reset
- [ ] persist sampled contact materials in banks and restore them during replay
- [ ] validate held-out training after canonical grasp measurements and shared physics change
- [ ] measure expert-bank yield under the shared contact-material distributions

## Retire and upstream

- [ ] remove robustness gain, delay and registration knobs when study ends
- [ ] select and retain one arm-command method after performance evidence
- [ ] upstream object metadata and robot bindings without per-pair environments
- [ ] review upstream saved JSON setups before removing unrelated consumers
- [ ] tabula rasa progress reward remains separate research scope
- [ ] review and delete attic after user review
