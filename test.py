import asyncio
from time import sleep
import time
import concurrent.futures


async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")

async def main():
    await asyncio.gather(count(), count(), count())




def main_thread():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pool.map(thread, range(3))

def thread(name):
    print("One")
    sleep(1)
    print("Two")

if __name__ == "__main__":
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")


    s = time.perf_counter()
    main_thread()
    elapsed = time.perf_counter() - s

    print(f"{__file__} executed in {elapsed:0.2f} seconds.")


