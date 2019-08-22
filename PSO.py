import numpy as np


class Particle:

    def __init__(self, length, max_pos, min_pos, max_vel, min_vel, w, c1, c2, problem):
        self.length = length
        self.max_pos = max_pos
        self.min_pos = min_pos
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.problem = problem

        self.position = min_pos + np.random.rand(length)*(max_pos-min_pos)
        self.velocity = np.zeros(length)
        self.fitness = self.problem.worst_fitness()
        self.loss = self.problem.worst_fitness()
        self.dist = self.problem.worst_fitness()

        self.pbest_pos = np.zeros(length)
        self.pbest_fit = self.problem.worst_fitness()
        self.pbest_loss = self.problem.worst_fitness()
        self.pbest_dist = self.problem.worst_fitness()

        self.gbest_pos = np.zeros(length)
        self.gbest_fit = self.problem.worst_fitness()
        self.gbest_loss = self.problem.worst_fitness()
        self.gbest_dist = self.problem.worst_fitness()

    def update(self):
        # Update velocity
        self.velocity = self.w * self.velocity + \
                        self.c1 * np.random.rand(self.length) * (self.pbest_pos-self.position) + \
                        self.c2 * np.random.rand(self.length) * (self.gbest_pos-self.position)

        self.velocity[self.velocity < self.min_vel] = self.min_vel[self.velocity < self.min_vel]
        self.velocity[self.velocity > self.max_vel] = self.max_vel[self.velocity > self.max_vel]

        #update position
        self.position = self.position + self.velocity
        self.position[self.position < self.min_pos] = self.min_pos[self.position < self.min_pos]
        self.position[self.position > self.max_pos] = self.max_pos[self.position > self.max_pos]


class Swarm:

    def __init__(self, n_particle, length, problem, n_iterations,
                 max_pos, min_pos, max_vel, min_vel, verbose=False):
        self.verbose = verbose
        self.n_particle = n_particle
        self.prob = problem
        self.n_iterations = n_iterations
        # self.pop = [Particle(length, max_pos=5.12, min_pos=-5.12, max_ve=2.5, min_ve=-2.5,
        #                      w=0.72984, c1=1.496172, c2=1.496172, problem=self.prob)

        self.pop = [Particle(length,
                             max_pos=max_pos, min_pos=min_pos,
                             max_vel=max_vel, min_vel=min_vel,
                             w=0.72984, c1=1.496172, c2=1.496172, problem=self.prob)
                    for _ in range(n_particle)]

    def iterate(self):
        for i in range(self.n_iterations):
            new_w = 0.9-i*(0.9-0.4)/self.n_iterations

            # star topology
            gbest_fit = self.pop[0].gbest_fit
            gbest_index = self.pop[0].gbest_pos
            gbest_updated = False

            # Evaluate all particles
            for index, par in enumerate(self.pop):
                par.w = new_w
                par.c1 = 2.0
                par.c2 = 2.0
                par.fitness, par.loss, par.dist = self.prob.fitness(par.position)

                # update pbest if necessary
                if self.prob.is_better(par.fitness, par.pbest_fit):
                    par.pbest_fit = par.fitness
                    par.pbest_pos = np.copy(par.position)
                    par.pbest_loss = par.loss
                    par.pbest_dist = par.dist

                # check if gbest needs to be updated
                if self.prob.is_better(par.pbest_fit, gbest_fit):
                    gbest_fit = par.pbest_fit
                    gbest_index = index
                    gbest_updated = True

            # now update gbest if necessary
            if gbest_updated:
                for par in self.pop:
                    par.gbest_fit = self.pop[gbest_index].pbest_fit
                    par.gbest_pos = np.copy(self.pop[gbest_index].pbest_pos)
                    par.gbest_loss = self.pop[gbest_index].pbest_loss
                    par.gbest_dist = self.pop[gbest_index].pbest_dist

            # now update particle position:
            for par in self.pop:
                par.update()

        return self.pop[0].gbest_pos, self.pop[0].gbest_fit, self.pop[0].gbest_loss, self.pop[0].gbest_dist
