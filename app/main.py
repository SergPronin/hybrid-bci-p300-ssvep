from gui.gui import StimulusApp


def main(
    *,
    auto_random_trials: bool = False,
    inter_trial_s: float = 1.0,
) -> None:
    app = StimulusApp(
        auto_random_trials=auto_random_trials,
        inter_trial_s=inter_trial_s,
    )
    app.run()


if __name__ == "__main__":
    main()
