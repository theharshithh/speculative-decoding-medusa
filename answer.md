Q. Why did i choose Onnx:

- Hardware and software support: Onnx is supportd by both pytorch and tensorflow, it supports various hardware acclerations. 
- Community support: Its very compatible with huggingface transformers, only drawback is the its still in beta versions. So version compatibility is an issue. 
- Performance: It matches the performance of NVIDIA's TensorRT, and is much simpler to setup. Ideal for concurrent requests. 
- Acceleration: It can be ran on GPU and is independent of the arch of the GPU. It was built for GPU acceleration.
- High throughput: It can support high throughput of many concurrent requests.

Tradeoffs:
- Increased compilation time
- Have to write custom wrappers over the base class for compiling the medusa with its medusa heads. 

Q. Include a brief explanation of how speculative decoding is implemented and its advantages.

A: Speculative decoding is a technique to introducre additional(medusa_heads) heads to the last layer of the model. The last layer will be passed on to a full connected linear layer (last_layer_size * last_layer_size) and then passed to an activation function(SiLU - Sigmoid Linear Activation.) ![SiLU activation function](https://miro.medium.com/v2/resize:fit:794/0*zzwSGGzn8ZhSsLFN). The activation function will be a softmax over the medusa_heads. The medusa_heads will be passed on to a linear layer (last_layer_size * vocab_size) and then passed to a softmax over the vocab_size.

It introduces a "draft" phase, where a smaller (or additional) set of heads predicts multiple tokens at once. These candidate tokens are then re-verified by the main model, eliminating the need to iterate token-by-token. In our Medusa approach, we attach multiple parallel heads to generate tokens in "branches," then confirm or reject them with a quick verification pass.

Advantages:
- Better throughput while keeping the accuracy of the model.
- Can be parallelized.

Disadvantages:
- Additional training of medusa heads.

Q: Explain your approach to dynamic batching and its benefits in serving LLMs.

A: Dynamic batching is a technique to efficiently manage concurrent requests to the LLM by intelligently grouping them for processing. My implementation uses a queue-based system that collects incoming requests and processes them in optimized batches.
Implementation:
- Request queue: All inference requests are added to a central queue
- Async processing: A dedicated thread continuously monitors the queue, forming batches when available
- Flexible batch size: The system adapts to process variable-sized batches based on incoming traffic
- Per-request response queues: Each request has its own async response queue, allowing independent completion

Advantages:
- GPU utilization: Maximizes GPU memory usage by processing multiple requests simultaneously
- Adaptive throughput: Scales from single requests to maximum batch size based on load
- Latency management: Minimizes waiting time by processing batches as soon as they're available
- Memory efficiency: Better memory utilization compared to spawning separate model instances

Tradeoffs:
- Implementation complexity: Requires careful queue management and thread coordination
- Race Condition: There can be a race condition between the main thread and the async thread.
- First-request latency: Initial requests might wait briefly to form batches in low-traffic scenarios
- Batch formation time: Finding optimal balance between waiting for batch completion vs. processing partial batches

The dynamic batching approach is particularly effective with Medusa's speculative decoding, as it allows parallel processing of multiple requests while still benefiting from Medusa's accelerated token generation. This combination delivers both high throughput and reduced latency compared to traditional token-by-token generation.