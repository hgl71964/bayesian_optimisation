import numpy as np
import torch as tr
import copy
from time import sleep
import os
import concurrent.futures
import multiprocessing
import asyncio

class api_utils:

    @staticmethod
    def wrapper(api_func: callable):  """IO bound"""
        def wrapper(x: tr.tensor,  #  shape[q,d]; q query, d-dimensional
                    r0: float,  #  unormalised reward
                    device: str,
                    ):
            """
            Returns:
                neg_rewards: [q, 1]
            """
            x = x.cpu(); q = x.shape[0]; neg_rewards = tr.zeros((q, ))

            for _ in range(5):  # handle potential network disconnection issue
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        for i, r in enumerate(executor.map(api_func, x)):  # multi-threading
                            neg_rewards[i] = -(r/r0)   

                except TypeError as ter:
                    print(f"api has error {ter}")
                    print("query is:", repr(x))
                    sleep(10)
                else:
                    break

            return neg_rewards.view(-1, 1).to(device)  # assume dtype == torch.float() overall

        return wrapper

def qpso(loss_f, size, ranges, logger, init_particles=None, iteration=100, g=0.8):
    """
    QPSO algorithm implemetation.
    Return (list): the global optimal vector.
    """
    ranges = np.array(ranges)
    N = len(ranges)
    particles = init_particles
    min_ranges = ranges[:, 0]
    max_ranges = ranges[:, 1]
    if particles is None:
        #calculate variances and means based on given ranges
        mean = (min_ranges+max_ranges)/2
        variance = ((max_ranges-min_ranges)/4)**2
        conv = np.zeros((N, N))
        np.fill_diagonal(conv, variance)
        #Generate starting partilces, which will also be local optimals.
        particles = [particle_generation(mean, conv) for i in range(size)]
        particles = np.clip(particles, min_ranges, max_ranges)
    local_optimal = particles.copy()


    # functions to update particle state and local optimal.
    def update_particles(index, global_optimal, iteration):
        try:
            phi_1 = np.random.rand(N)
            phi_2 = np.random.rand(N)
            mu = np.random.rand(N)
            sign = np.random.choice([-1, 1], N)
            potential = (phi_1*local_optimal[index]+phi_2*global_optimal)/(phi_1+phi_2)
            L = 1/g*np.abs(particles[index]-potential)
            particles[index] = np.clip(potential+sign*L*np.log(1/mu), min_ranges, max_ranges)
            new_loss = loss_f(particles[index], index, iteration+1)
            particle_losses[index] = new_loss
            if new_loss < local_losses[index]:
                local_optimal[index] = particles[index]
                local_losses[index] = new_loss
        except Exception as e:
            print("Got Exception in update partices")
            print(e)
       
    with futures.ThreadPoolExecutor(max_workers=size) as executor:
        # calculate initial losses in parallel.
        future_to_loss = [executor.submit(loss_f, p, i, 0) for i, p in enumerate(particles)]
        particle_losses = [f.result() for f in future_to_loss]
        local_losses = particle_losses.copy()
        global_optimal = local_optimal[local_losses.index(min(local_losses))]
        logger.log_particle(particles, particle_losses, 0)
        logger.log_local_optimal(local_optimal, local_losses, 0)
        logger.log_global_optimal(global_optimal, min(local_losses), 0)
        for it in range(iteration):
            print("start of iteration: {}".format(it))
            # calculate global optimal based on local optimals
            # update all particles based on global and local optimals of last iteration in parallel.
            future_to_loss = [executor.submit(update_particles, i, global_optimal, it) for i in range(len(particles))]
            futures.wait(future_to_loss)
            global_optimal = local_optimal[local_losses.index(min(local_losses))]
            logger.log_particle(particles, particle_losses, it+1)
            logger.log_local_optimal(local_optimal, local_losses, it+1)
            logger.log_global_optimal(global_optimal, min(local_losses), it+1)