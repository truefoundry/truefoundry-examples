
import argparse
import time


def main(limit: int):
    for i in range(limit):
        print(f"Running Job: {i + 1}")
        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", default=10, type=int)
    args = parser.parse_args()
    main(args.limit)
