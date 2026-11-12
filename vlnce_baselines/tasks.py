"""The supported benchmark registry; no simulator imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    name: str
    config_name: str
    task_type: str
    timeout: int

    @property
    def config_path(self):
        return f'vlnce_baselines/config/{self.config_name}.yaml'


BENCHMARKS = {
    'vlnce_r2r': Benchmark('vlnce_r2r', 'r2r', 'VLN', 259200),
    # OpenNav R2R-CE 100-episode val_unseen subset with its own start headings.
    'vlnce_r2r_rand100': Benchmark('vlnce_r2r_rand100', 'r2r_rand100', 'VLN', 86400),
    'vlnce_rxr': Benchmark('vlnce_rxr', 'rxr', 'VLN', 129600),
    'hm3d_v2': Benchmark('hm3d_v2', 'objectnav_v2', 'OBJNAV', 172800),
    'hm3d_ovon': Benchmark('hm3d_ovon', 'objectnav_ovon', 'OBJNAV', 259200),
    # LangMap single-goal (HieraNav) on HM3D v0.2 val, scored as language navigation.
    'langmap': Benchmark('langmap', 'langmap', 'VLN', 259200),
}
ALIASES = {'r2r': 'vlnce_r2r', 'r2r_rand100': 'vlnce_r2r_rand100', 'rand100': 'vlnce_r2r_rand100',
           'rxr': 'vlnce_rxr', 'ovon': 'hm3d_ovon'}


def get_benchmark(name):
    key = str(name).lower()
    key = ALIASES.get(key, key)
    if key not in BENCHMARKS:
        raise ValueError(f'Unsupported benchmark {name!r}; choose ' + ', '.join(BENCHMARKS))
    return BENCHMARKS[key]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    args = parser.parse_args()
    try:
        benchmark = get_benchmark(args.task)
    except ValueError as error:
        parser.error(str(error))
    print(benchmark.name, benchmark.config_path, benchmark.timeout)
