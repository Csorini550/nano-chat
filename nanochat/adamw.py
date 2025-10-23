import torch  # import torch to access tensor utilities and optimizer primitives
import torch.distributed as dist  # import torch.distributed to coordinate work across multiple ranks
from torch import Tensor  # import Tensor alias to make type annotations concise


class DistAdamW(torch.optim.Optimizer):  # declare a distributed AdamW optimizer that inherits from torch's Optimizer
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):  # initialize the optimizer with parameter groups and hyperparameters
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)  # collect the provided hyperparameters into the defaults dictionary
        super().__init__(param_groups, defaults)  # call the Optimizer base class constructor with parameters and defaults

    @torch.compile  # compile the step method to speed up repeated executions when supported
    @torch.no_grad()  # disable gradient tracking because optimizer steps should not build autograd history
    def step(self):  # perform one distributed AdamW update across all parameter shards
        rank = dist.get_rank()  # obtain the integer identifier for the current distributed rank
        world_size = dist.get_world_size()  # determine how many ranks participate in the distributed group
        reduce_scatter_futures: list[torch.Future] = []  # maintain futures that track reduce-scatter completion for each gradient
        all_reduce_futures: list[torch.Future] = []  # maintain futures that track all-gather completion for each parameter shard
        grad_slices = []  # store locally owned gradient slices aligned with parameter shards
        for group in self.param_groups:  # iterate over each parameter group managed by this optimizer
            params: list[Tensor] = group["params"]  # pull out the list of tensors belonging to the current group
            for base_i in range(len(params)):  # loop over each parameter tensor in this group by index
                grad = params[base_i].grad  # access the gradient tensor accumulated on the parameter
                rank_size = grad.shape[0] // world_size  # compute the length of the shard assigned to each rank
                grad_slice = torch.empty_like(grad[:rank_size])  # allocate a tensor to receive the local gradient portion
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())  # launch asynchronous reduce-scatter to average gradients and record the future
                grad_slices.append(grad_slice)  # stash the gradient shard so it can be used during the parameter update phase

        idx = 0  # initialize an index that walks the gradient slice list as parameters are updated
        for group in self.param_groups:  # iterate through parameter groups again to apply AdamW updates
            beta1, beta2 = group['betas']  # unpack the exponential decay rates for first and second moments
            eps = group['eps']  # retrieve the epsilon constant for numerical stability in denominators
            wd = group['weight_decay']  # grab the decoupled weight decay coefficient for the group
            params = group['params']  # reference the parameters belonging to this group
            for base in range(len(params)):  # update each parameter tensor within the current group
                reduce_scatter_futures[idx].wait()  # ensure the corresponding gradient slice has finished reducing before use
                p = params[base]  # access the parameter tensor about to be updated
                rank_size = p.shape[0] // world_size  # compute the length of the parameter shard local to this rank
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]  # slice the tensor to isolate the shard owned by this rank
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)  # determine the effective learning rate, honoring optional per-parameter multipliers
                state = self.state[p]  # retrieve the mutable optimizer state dictionary associated with this parameter
                g_slice = grad_slices[idx]  # look up the gradient shard aligned with the current parameter index
                if not state:  # lazily initialize optimizer state the first time the parameter is updated
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)  # allocate a tensor counter that tracks the number of updates applied
                    state['exp_avg'] = torch.zeros_like(p_slice)  # allocate the first moment accumulator for the local parameter shard
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)  # allocate the second moment accumulator for the local parameter shard
                exp_avg = state['exp_avg']  # create a short alias to the first moment accumulator for readability
                exp_avg_sq = state['exp_avg_sq']  # create a short alias to the second moment accumulator for readability
                state['step'] += 1  # increment the step counter to reflect the upcoming update
                t = state['step']  # store the updated step counter for reuse in bias correction calculations
                if wd != 0:  # apply decoupled weight decay when a nonzero coefficient is configured
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)  # compute the effective decay value considering per-parameter overrides
                    p_slice.mul_(1 - eff_weight_decay)  # scale the parameter shard to implement weight decay before applying Adam update
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)  # update the exponential moving average of gradients using beta1
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)  # update the exponential moving average of squared gradients using beta2
                bias1 = 1 - beta1 ** t  # compute the first moment bias correction denominator based on step count
                bias2 = 1 - beta2 ** t  # compute the second moment bias correction denominator based on step count
                denom = exp_avg_sq.sqrt().add_(eps)  # compute the denominator by taking the square root of the second moment and adding epsilon
                step_size = lr * (torch.sqrt(bias2) / bias1)  # determine the bias-corrected step scaling factor for this update
                update = exp_avg.div(denom).mul_(step_size)  # combine first moment, denominator, and step size to form the Adam update
                p_slice.add_(other=update, alpha=-1.0)  # apply the update by subtracting it from the parameter shard in place
                idx += 1  # advance the gradient slice index so the next parameter reads the correct shard
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())  # launch asynchronous all-gather to rebuild the full parameter across ranks
        torch.futures.collect_all(all_reduce_futures).wait()  # wait for every all-gather to finish so all ranks share synchronized parameters before returning
