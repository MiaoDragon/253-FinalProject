"""
this implements the memory recording past trajectories
each memory records: (s_t, a_t, r_t, p(a_t|s_t))
memory is a circular array of array
"""
class Memory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.memory = []
        self.b = 0.
    def remember(self, traj):
        # given new trajectory in the form [(s_t, a_t, r_t, p(a_t|s_t))]
        # add this into memory
        self.idx = (self.idx + 1) % self.capacity
        if len(self.memory) < self.capacity:
            self.memory.append(traj)
        else:
            self.memory[self.idx] = traj
    def compute_b(self):
        # compute baseline
        b = 0.
        for traj in self.memory:
            for exp in traj:
                b += exp[2]
        self.b = b / len(self.memory)

    def loss(self, net):
        self.compute_b()
        b = self.b
        sum_exps = 0.
        for exp_i in range(len(self.memory)):
            exp = exps[exp_i]
            R = 0.
            sum_exp = 0.
            bt = b / len(exp)
            # compute is for entire trajectory (is: important sampling)
            log_is = 0.
            for t in range(len(exp)):
                (s,a,r,p) = exp[t]
                log_is += torch.log(net(s)[a])-torch.log(p)
            # treat the IS term as data, don't compute gradient w.r.t. it
            log_is = log_is.detach()
            for t in range(len(exp)-1,-1,-1):
                (s,a,r,p) = exp[t]
                R += r - bt
                # future reward * current loglikelihood * past IS
                sum_exp += torch.log(net(s)[a])*R*torch.exp(log_is)
                log_is -= (torch.log(net(s)[a])-torch.log(p))
                # the predicted prob is treated as constant
                log_is = log_is.detach()
            sum_exps += sum_exp
        return sum_exps / len(self.memory)
