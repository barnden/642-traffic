import os
import xml.etree.ElementTree as ET
import numpy as np
import sys
from resco_benchmark.config.map_config import map_configs
from pathlib import Path
import re
from collections import defaultdict
from matplotlib import pyplot as plt
import json

cwd = os.path.dirname(os.getcwd())
log_dir = Path(cwd, "..", "results")
env_base = Path(cwd, "environments")

dirs = [folder for folder in next(os.walk(log_dir.absolute()))[1]]
metrics = ["timeLoss", "duration", "waitingTime"]

def parse_csv(experiment: Path, episode: int):
    num_steps = 0
    total = 0

    trip_file = experiment.joinpath(f'metrics_{episode}.csv')

    if not trip_file.exists():
        return None

    with open(trip_file) as file:
        for line in file:
            line = line.split('}')

            queues = line[2]
            signals = queues.split(':')
            step_total = 0

            for s, signal in enumerate(signals):
                if s == 0:
                    continue

                queue = signal.split(',')
                queue = int(queue[0])
                step_total += queue

            step_avg = step_total / len(signals)
            total += step_avg
            num_steps += 1

    average = total / num_steps

    return average

def parse_xml(metric: str, experiment: Path, episode: int):
    try:
        tree = ET.parse(experiment.joinpath(f'tripinfo_{episode}.xml').absolute())
        root = tree.getroot()

        num_trips = 0
        total = 0
        last_departure_time = 0
        last_departure_id = ''

        for child in root:
            try:
                num_trips += 1
                total += float(child.attrib[metric])

                if metric == 'timeLoss':
                    total += float(child.attrib['departDelay'])
                    depart_time = float(child.attrib['depart'])

                    if depart_time > last_departure_time:
                        last_departure_time = depart_time
                        last_departure_id = child.attrib['id']
            except:
                break

        route_file = Path(env_base, map_name, f"{map_name}_{episode}.rou.xml").absolute()

        if metric == 'timeLoss':
            try:
                tree = ET.parse(route_file)
            except FileNotFoundError:
                route_file = Path(env_base, map_name, f"{map_name}.rou.xml").absolute()
                tree = ET.parse(route_file)

            root = tree.getroot()
            last_departure_time = None

            for child in root:
                if child.attrib['id'] == last_departure_id:
                    last_departure_time = float(child.attrib['depart'])

            never_departed = []
            if last_departure_time is None:
                raise RuntimeError("Wrong Trip File")

            for child in root:
                if child.tag != 'vehicle':
                    continue

                depart_time = float(child.attrib['depart'])

                if depart_time > last_departure_time:
                    never_departed.append(depart_time)

            never_departed = np.asarray(never_departed)
            never_departed_delay = np.sum(float(map_configs[map_name]['end_time']) - never_departed)

            total += never_departed_delay
            num_trips += len(never_departed)

        average = total / num_trips

        return average
    except ET.ParseError as e:
        print("Failed to parse episode", i)
        return None

    except FileNotFoundError:
        return None

    except Exception as e:
        print(e)
        return None

for metric in metrics:
    run_avg = defaultdict(list)

    for path in dirs:
        spath = path.split('-')

        map_name = spath[2]

        average_per_episode = []
        experiment = log_dir.joinpath(path)
        i = 1

        while True:
            if metric not in ['queue'] and (average := parse_xml(metric, experiment, i)) is None:
                break
            elif (average := parse_csv(experiment, i)) is None:
                break

            average_per_episode.append(average)

            i += 1

        del spath[1]
        run_name = ' '.join(spath)
        average_per_episode = np.asarray(average_per_episode)

        run_avg[run_name] += [average_per_episode]

    data = {}
    for run_name, runs in run_avg.items():
        min_len = min(map(len, runs))
        runs = [run[:min_len] for run in runs]

        avg = np.sum(runs, 0) / len(runs)
        err = np.std(runs, axis=0)

        print(run_name, metric, avg[-1])

        # plt.title(run_name)
        # plt.plot(avg)
        # plt.fill_between(np.arange(avg.shape[0]), avg - err, avg + err, interpolate=True, alpha=.25)
        # plt.show()

        data[run_name] = {
            'avg': avg.tolist(),
            'err': err.tolist()
        }

    print()

    with open(f'dump_{metric}.py', 'w') as file:
        file.write(json.dumps(data))
