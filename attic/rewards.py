def reach_tcp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """``1 - tanh(d / std)`` on the TCP-to-grasp distance."""
    dist, _ = tcp_grasp_error(env)
    return 1.0 - torch.tanh(dist / std)


def align_tcp(env: ManagerBasedRLEnv, std: float, gate_dist: float) -> torch.Tensor:
    """Orientation alignment to the grasp, counted only within ``gate_dist`` [m]."""
    dist, ang = tcp_grasp_error(env)
    return (1.0 - torch.tanh(ang / std)) * (dist < gate_dist).float()


def grasp_lever(env: ManagerBasedRLEnv, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """1 while the gripper is closed on the lever at the grasp pose."""
    return lever_held(env, dist_threshold, ang_threshold).float()


def align_held(env: ManagerBasedRLEnv, std: float, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """Tight orientation alignment [rad] while the lever is held.

    Teacher v3 held the lever at 0.34 rad, just inside the 0.35 rad hold gate,
    so its distilled student fell out of the gate on small imitation errors.
    A narrow kernel while holding pulls the grasp to the expert orientation.
    """
    _, ang = tcp_grasp_error(env)
    return (1.0 - torch.tanh(ang / std)) * lever_held(env, dist_threshold, ang_threshold).float()


def close_far_penalty(env: ManagerBasedRLEnv, dist_threshold: float) -> torch.Tensor:
    """1 while the gripper is closed away from the lever (closing on air)."""
    dist, _ = tcp_grasp_error(env)
    return (dist > dist_threshold).float() * _gripper_closed(env)


def valve_unheld_motion(env: ManagerBasedRLEnv, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """Squared valve joint rate [rad^2/s^2] while the lever is *not* held.

    Teacher v2 struck the lever on its approach, flinging it half open at
    ~11 rad/s before grasping. Any valve motion before the grasp is an impact.
    """
    return valve_rate(env).square() * (~lever_held(env, dist_threshold, ang_threshold)).float()


def track_valve_reference(
    env: ManagerBasedRLEnv, std: float, dist_threshold: float, ang_threshold: float, command_name: str = "valve_turn"
) -> torch.Tensor:
    """``exp(-(q - q_ref)^2 / std^2)`` while the lever is held; ``std`` [rad]."""
    error = valve_angle(env) - _turn_command(env, command_name).ref_angle
    held = lever_held(env, dist_threshold, ang_threshold).float()
    return torch.exp(-(error / std).square()) * held


def valve_opened_on_schedule(
    env: ManagerBasedRLEnv,
    threshold_rad: float,
    dist_threshold: float,
    ang_threshold: float,
    command_name: str = "valve_turn",
) -> torch.Tensor:
    """1 while the valve is open, held, and the reference has also reached open.

    Opening ahead of the reference earns nothing extra, so there is no reward
    for turning faster than commanded.
    """
    opened = _is_open(env, valve_angle(env), threshold_rad)
    on_schedule = _turn_command(env, command_name).reference_done
    return (opened & on_schedule & lever_held(env, dist_threshold, ang_threshold)).float()


def valve_rate_deviation(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Squared deviation [rad^2/s^2] of the valve rate from the commanded rate while turning.

    Tracking the reference angle alone let the teacher chatter the lever at
    20 Hz around the reference (mean |rate| ~1 rad/s for a 0.3 rad/s command);
    this asks for a steady turn at the commanded speed.
    """
    term = _turn_command(env, command_name)
    return (valve_rate(env) - _opening_sign(env) * term.rate).square() * term.turning.float()


class ValveTurnRateCommand(CommandTerm):
    @property
    def turning(self) -> torch.Tensor:
        """True while the reference is advancing (engaged and not yet open)."""
        return self.engaged & ~self.reference_done
