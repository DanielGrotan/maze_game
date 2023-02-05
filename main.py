# from level_editor import LevelEditor
from trainer import Trainer


def main():
    # level_editor = LevelEditor()
    # level_editor.run()

    trainer = Trainer("neat-config", "training_output", "grid.json")
    trainer.train(False, generations=10000)


if __name__ == "__main__":
    main()
