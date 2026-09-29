"""Run the complete M2-5 collection, risk fitting, and validation."""

from .run_training import main as train
from .validate import main as validate


def main():
    train()
    validate()


if __name__ == "__main__":
    main()
