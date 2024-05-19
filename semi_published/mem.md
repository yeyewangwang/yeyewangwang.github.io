## Training Workload Optimization

The primary goal for analyzing neural net memory properties is to train larger or higher-dimensional models on smaller GPU's for longer period of time.

### The Problem

What’s a deep learning training workload's memory usage profile? 

<ul style="list-style-type:none;">
  <li>1. Adam optimizer states (moment, etc.)</li>
  <li>2. Gradients</li>
  <li>3. Intermediate result  (attention scores, linearized output/activation memory/data) </li>
  <li>4. Model weights</li>
</ul>

According to Deepspeed tech blog, 1, 2 and 4 each respectively takes roughly 80%, 10%, 5%. 
Attention scores dominate transformer intermediate results. 

Most tensors are used only when the forward passes reaches the layer, and after the backward pass reaches the layer. In other words, there is egregious waste in storing everything in VRAM, if the time to complete the round trip forward to calculate loss and then back to compute the gradient is greater than the time for loading or recomputing, and chip stalls caused by them. 

Because the training workload is periodical and predictable, it's possible to benefit from overriding systems built-in optimizations (such as avoiding CPU logical cores, paging, or cuDNN).

### Strategies
#### Trade compute for memory

Checkpointing:  in other words, [recompute](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)/rematerialize intermediate linearization results/activation memory. Estimates on its runtime impact range is “50% more expensive” training time.

Flash Attention, do tensor calculations using small tiles

#### Reserve GPUs for only tensor multiplications

Zero Offload: offload weight updates to CPU, then overlap CPU and GPU execution. Zero Offload claims to achieve roughly 10X larger model size.

A similar method is to optimise the PyTorch compiler to fuse multiple point-wise operators (sigmoid) into one, and recompute them, since they are extremely slow on GPUs. 

#### Memory Allocation

Assign a large zero tensor as a buffer. Use it store all tensors needed.

```
self.buffer = torch.zeros(size, dtype=dtype, device=device)

# Source: Deepspeed Zero runtime
```


To use each tensor, use `torch.narrow` to generate a view of the buffer. Also, periodically defragment the large buffer.


#### Memory Leak Prevention

Avoid all references to “self” or other external variables inside any callback function.