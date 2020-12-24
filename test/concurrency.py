import concurrent.futures
import logging
import threading
import time
import asyncio

# [rest of code]

def thread_function(name, before, after):
    logging.info("Thread %s: starting", name)
    print(before)
    time.sleep(5)
    print(after)
    logging.info("Thread %s: finishing", name)
    return name


async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")

async def main():
    await asyncio.gather(count(), count(), count())




if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # context manager will apply join method in the end
        for i, ans in enumerate(executor.map(thread_function, ("a", "b", "c"), range(1,4), [7,8,9])):
            print(ans))

    
    # benchmark async io
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds."


# import logging
# import threading
# import time
# import concurrent.futures

# def thread_function(name):
#     logging.info("Thread %s: starting", name)
#     time.sleep(5)
#     logging.info("Thread %s: finishing", name)

# if __name__ == "__main__":
#     format = "%(asctime)s: %(message)s"
#     logging.basicConfig(format=format, level=logging.INFO,
#                         datefmt="%H:%M:%S")

#     logging.info("Main    : before creating thread")
#     x = threading.Thread(target=thread_function, args=(1,), daemon=True)
#     logging.info("Main    : before running thread")
#     x.start()
#     logging.info("Main    : wait for the thread to finish")
#     x.join()  # main thread will wait for this thread, 
#     logging.info("Main    : all done")
