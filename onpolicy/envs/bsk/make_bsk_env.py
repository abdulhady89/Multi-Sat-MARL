import random

import numpy as np
from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw
from bsk_rl import GeneralSatelliteTasking
from bsk_rl.utils.orbital import random_orbit
from bsk_rl.utils.orbital import walker_delta_args
import pdb


def make_BSK_Cluster_env(args, satellite_names):
    # Common orbital parameters for all satellites
    inclination = 50.0          # degrees, fixed for all satellites
    altitude = 500              # km, fixed for all satellites
    eccentricity = 0            # Circular orbit
    # Longitude of Ascending Node (Omega), fixed for all
    LAN = 0
    arg_periapsis = 0           # Argument of Periapsis (omega), fixed for all

    # True anomaly offsets for spacing satellites along the Cluster orbit
    true_anomaly_offsets = [225 - 0.0001*i for i in range(
        len(satellite_names))]  # degrees
    orbit_ls = []
    for offset in true_anomaly_offsets:
        orbit = random_orbit(
            i=inclination, alt=altitude, e=eccentricity, Omega=LAN, omega=arg_periapsis, f=offset
        )
        orbit_ls.append(orbit)

    if args.scenario_name == "ideal":
        battery_sizes = [1e6]*len(satellite_names)
        memory_sizes = [1e6]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = 500

    elif args.scenario_name == "limited_batt":
        battery_sizes = [50]*len(satellite_names)
        memory_sizes = [int(args.memory_size)]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "limited_mem":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_sizes = [5000]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "limited_baud":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_sizes = [args.memory_size]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = 0.5
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "limited_img":
        battery_sizes = [int(args.battery_capacity)
                         ]*len(satellite_names)
        memory_sizes = [int(args.memory_size)]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = 125

    elif args.scenario_name == "limited_all":
        battery_sizes = [50]*len(satellite_names)
        memory_sizes = [5000]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = 0.5
        instr_baud_rate = 125

    elif args.scenario_name == "default":
        battery_sizes = [int(args.battery_capacity)
                         ]*len(satellite_names)
        memory_sizes = [int(args.memory_size)]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "random_all":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_sizes = [args.memory_size]*len(satellite_names)
        random_init_memory = True
        random_init_battery = True
        random_disturbance = True
        random_RW_speed = True
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "random_batt":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_sizes = [args.memory_size]*len(satellite_names)
        random_init_memory = False
        random_init_battery = True
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "random_mem":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_sizes = [args.memory_size]*len(satellite_names)
        random_init_memory = True
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "random_dist":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_sizes = [args.memory_size]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = True
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "random_rw":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_sizes = [args.memory_size]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = True
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "hetero_batt":
        battery_sizes = [50, 100, 200, 400]
        memory_sizes = [int(args.memory_size)]*len(satellite_names)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "hetero_mem":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_sizes = [5000, 10000, 250000, 500000]
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    else:
        print("Scenario name not available")
        NotImplementedError

    # Define four satellites in a "train" Cluster formation along the same orbit
    multiSat = []
    index = 0

    for orbit, battery_size, memory_size in zip(orbit_ls, battery_sizes, memory_sizes):
        sat_args = dict(
            # Power
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * args.init_battery_level / 100 * 3600) if not random_init_battery else np.random.uniform(
                battery_size * 3600 * 0.4, battery_size * 3600 * 0.5),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30,
            transmitterPowerDraw=-25,
            thrusterPowerDraw=-80,
            # Data Storage
            dataStorageCapacity=memory_size * 8e6,  # MB to bits,
            storageInit=int(memory_size *
                            args.init_memory_percent/100) * 8e6 if not random_init_memory else np.random.uniform(memory_size * 8e6 * 0.2, memory_size * 8e6 * 0.8),
            instrumentBaudRate=instr_baud_rate * 1e6,
            transmitterBaudRate=-1*baud_rate * 1e6,
            # Attitude
            imageAttErrorRequirement=0.1,
            imageRateErrorRequirement=0.1,
            disturbance_vector=lambda: np.random.normal(
                scale=0.0001, size=3) if random_disturbance else np.array([0.0, 0.0, 0.0]),
            maxWheelSpeed=6000.0,  # RPM
            wheelSpeeds=lambda: np.random.uniform(
                -3000, 3000, 3) if random_RW_speed else np.array([0.0, 0.0, 0.0]),
            desatAttitude="nadir",
            u_max=0.4,
            K1=0.25,
            K3=3.0,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150,
            # Orbital elements
            oe=orbit
        )

        class ImagingSatellite(sats.ImagingSatellite):
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),

                ),
                obs.Eclipse(norm=5700),
                obs.OpportunityProperties(
                    dict(prop="priority"),
                    dict(prop="opportunity_open", norm=5700.0),
                    n_ahead_observe=args.n_obs_image,
                ),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700),
                    dict(prop="opportunity_close", norm=5700),
                    type="ground_station",
                    n_ahead_observe=1,
                ),
                obs.Time(),
            ]
            action_spec = [act.Image(n_ahead_image=args.n_act_image),
                           act.Downlink(duration=20.0),
                           act.Desat(duration=20.0),
                           act.Charge(duration=20.0),
                           ]
            dyn_type = dyn.ManyGroundStationFullFeaturedDynModel
            fsw_type = fsw.SteeringImagerFSWModel

        sat = ImagingSatellite(f"EO-{index}", sat_args)
        multiSat.append(sat)
        index += 1

    duration = args.orbit_num * 5700.0  # About 2 orbits

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.UniformTargets(args.uniform_targets),
        rewarder=data.UniqueImageReward(),
        time_limit=duration,
        # Note that dyn must inherit from LOSCommunication
        communicator=comm.LOSCommunication(),
        log_level="WARNING",
        terminate_on_time_limit=True,
        failure_penalty=args.failure_penalty,
        vizard_dir="./tmp_cluster/vizard" if args.use_render else None,
        vizard_settings=dict(showLocationLabels=-
                             1) if args.use_render else None,
    )
    return env


def make_BSK_Walker_env(args, satellite_names):
    # Define four satellites in walker delta orbits
    sat_arg_randomizer = walker_delta_args(
        altitude=500.0, inc=50.0, n_planes=args.n_satellites, randomize_lan=False, randomize_true_anomaly=False)

    if args.scenario_name == "ideal":
        battery_sizes = [1e6]*len(satellite_names)
        memory_size = 1e6
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = 500

    elif args.scenario_name == "limited":
        battery_sizes = [int(args.battery_capacity/4)
                         ]*len(satellite_names)
        memory_size = int(args.memory_size/20)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "default":
        battery_sizes = [int(args.battery_capacity)
                         ]*len(satellite_names)
        memory_size = int(args.memory_size)
        random_init_memory = False
        random_init_battery = False
        random_disturbance = False
        random_RW_speed = False
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    elif args.scenario_name == "random":
        battery_sizes = [args.battery_capacity]*len(satellite_names)
        memory_size = args.memory_size
        random_init_memory = True
        random_init_battery = True
        random_disturbance = True
        random_RW_speed = True
        baud_rate = args.baud_rate
        instr_baud_rate = args.instr_baud_rate

    else:
        print("Scenario name not available")
        NotImplementedError

    # Define four satellites in a "train" Cluster formation along the same orbit
    multiSat = []
    index = 0
    for battery_size in battery_sizes:
        sat_args = dict(
            # Power
            batteryStorageCapacity=battery_size * 3600,
            storedCharge_Init=int(battery_size * args.init_battery_level / 100 * 3600) if not random_init_battery else np.random.uniform(
                battery_size * 3600 * 0.4, battery_size * 3600 * 0.5),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30.0,
            transmitterPowerDraw=-25.0,
            thrusterPowerDraw=-80.0,
            # Data Storage
            dataStorageCapacity=memory_size * 8e6,  # MB to bits,
            storageInit=int(memory_size *
                            args.init_memory_percent/100) * 8e6 if not random_init_memory else np.random.uniform(memory_size * 8e6 * 0.2, memory_size * 8e6 * 0.8),
            instrumentBaudRate=instr_baud_rate * 1e6,
            transmitterBaudRate=-1*baud_rate * 1e6,
            # Attitude
            imageAttErrorRequirement=0.1,
            imageRateErrorRequirement=0.1,
            disturbance_vector=lambda: np.random.normal(
                scale=0.0001, size=3) if random_disturbance else np.array([0.0, 0.0, 0.0]),
            maxWheelSpeed=6000.0,  # RPM
            wheelSpeeds=lambda: np.random.uniform(
                -3000, 3000, 3) if random_RW_speed else np.array([0.0, 0.0, 0.0]),
            desatAttitude="nadir",
            u_max=0.4,
            K1=0.25,
            K3=3.0,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150,
        )

        class ImagingSatellite(sats.ImagingSatellite):
            observation_spec = [
                obs.SatProperties(
                    dict(prop="storage_level_fraction"),
                    dict(prop="battery_charge_fraction"),
                    dict(prop="wheel_speeds_fraction"),

                ),
                obs.Eclipse(),
                obs.OpportunityProperties(
                    dict(prop="priority"),
                    dict(prop="opportunity_open", norm=5700.0),
                    n_ahead_observe=args.n_obs_image,
                ),
                obs.OpportunityProperties(
                    dict(prop="opportunity_open", norm=5700),
                    dict(prop="opportunity_close", norm=5700),
                    type="ground_station",
                    n_ahead_observe=1,
                ),
                obs.Time(),
            ]
            action_spec = [act.Image(n_ahead_image=args.n_act_image),
                           act.Downlink(duration=20.0),
                           act.Desat(duration=20.0),
                           act.Charge(duration=20.0),
                           ]
            dyn_type = dyn.ManyGroundStationFullFeaturedDynModel
            fsw_type = fsw.SteeringImagerFSWModel

        sat = ImagingSatellite(f"EO-{index}", sat_args)
        multiSat.append(sat)
        index += 1

    duration = args.orbit_num * 5700.0  # About 2 orbits

    env = GeneralSatelliteTasking(
        satellites=multiSat,
        scenario=scene.UniformTargets(args.uniform_targets),
        rewarder=data.UniqueImageReward(),
        time_limit=duration,
        # Note that dyn must inherit from LOSCommunication
        communicator=comm.LOSCommunication(),
        sat_arg_randomizer=sat_arg_randomizer,
        log_level="WARNING",
        terminate_on_time_limit=True,
        failure_penalty=args.failure_penalty,
        vizard_dir="./tmp_cluster/vizard" if args.use_render else None,
        vizard_settings=dict(showLocationLabels=-
                             1) if args.use_render else None,
    )
    return env
