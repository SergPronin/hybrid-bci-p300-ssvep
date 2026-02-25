from psychopy import core
from core.grid import Grid
from core.stimulus_controller import StimulusController


def main():
    grid = Grid(size=3)
    controller = StimulusController(grid)

    controller.start()

    print("Stimulus engine started. Press Ctrl+C to stop.")

    try:
        while True:
            event = controller.update()
            if event:
                print(event)

            core.wait(0.001)  # минимальная нагрузка CPU

    except KeyboardInterrupt:
        controller.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()