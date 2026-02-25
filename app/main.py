from app.config import load_config
from app.experiment import Experiment


def main() -> None:
    """
    Minimal entry point for hybrid P300/SSVEP stimulus.
    All settings are defined in app/config.py.
    """
    config = load_config()
    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
