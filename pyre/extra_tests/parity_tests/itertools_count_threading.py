import itertools
import threading


def check(step):
    counter = itertools.count(step=step)

    def consume():
        for _ in range(10_000):
            next(counter)

    threads = [threading.Thread(target=consume) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert next(counter) == 100_000 * step


check(1)
check(5)
print("OK")
