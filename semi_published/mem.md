## Training Workload Optimization

### The Problem

What’s a deep learning training workload's memory usage profile? 1. Adam optimizer states (moment, etc.) 2. Gradients 3. Intermediate result  (attention scores, linearized output/activation memory) 4. Model weights. Attention scores dominate transformer intermediate results. According to Deepspeed tech blog, 1, 2 and 4 each respectively takes roughly 80%, 10%, 5%. 



Most tensors are used only twice, once in forward and backward passes respectively. In other words, egregious waste in storing everything in VRAM. It comes down to the time to complete the round trip forward to calculate loss and then back to compute the gradient, versus loading or recomputing, and chip stalls caused by them.

Because the training workload is the same across all epochs, it's sometimes possible to benefit from overriding operating systems built-in optmizations (such as avoiding CPU logical cores and paging).

### Strategies
#### Trade compute for memory

Suitable for models not having issues with overall training time.

Checkpointing:  in other words, [recompute](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)/rematerialize intermediate linearization results/activation memory. Estimates on its runtime impact range is “50% more expensive” training time.

Flash Attention, do tensor calculations using small tiles

#### Reserve GPUs for only tensor multiplies

Zero Offload: offload weight updates to CPU, then overlap CPU and GPU execution. Zero Offload claims to achieve roughly 10X larger model.

A similar method is to optimise the PyTorch compiler to fuse multiple point-wise operators (sigmoid) into one, and recompute them, since they are extremely slow on GPUs. 

#### Memory Allocation

Assign a large zero tensor as a buffer. Use it store all tensors needed.

```
self.buffer = torch.zeros(size, dtype=dtype, device=device)

# source: Deepspeed Zero runtime
```


To use each tensor, use `torch.narrow` to generate a view of the buffer. Also, periodically defragment the large buffer.


#### Memory Leak Prevention

Avoid all references to “self” or other external variables inside any callback function.