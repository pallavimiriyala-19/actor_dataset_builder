import argparse
from src.pipeline import build_actor_dataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", "-a", nargs="+", required=True)
    args = p.parse_args()

    for name in args.actor:
        build_actor_dataset(name)

if __name__ == "__main__":
    main()
