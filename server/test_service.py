import asyncio
import aiohttp
import time
from typing import List
import statistics

async def send_request(session, prompt: str):
    start_time = time.time()
    async with session.post(
        "http://localhost:8000/generate",
        json={"prompt": prompt, "temperature": 0.7, "max_tokens": 50}
    ) as response:
        result = await response.json()
        end_time = time.time()
        return end_time - start_time, result

async def run_concurrent_requests(prompts: List[str], concurrency: int):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts:
            tasks.append(send_request(session, prompt))
            if len(tasks) >= concurrency:
                results = await asyncio.gather(*tasks)
                tasks = []
                yield results

        if tasks:
            results = await asyncio.gather(*tasks)
            yield results

async def main():
    test_prompts = [
        "Explain what is machine learning in simple terms.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "What is the process of evolution?"
    ]
    
    print("Startingserver test...")
    print("Testing for concurrency levels:")
    
    for concurrency in [1, 2, 3]:
        print(f"\nTesting with {concurrency} concurrent requests:")
        response_times = []
        
        async for batch_results in run_concurrent_requests(test_prompts, concurrency):
            for response_time, result in batch_results:
                response_times.append(response_time)
                print(f"Request completed in {response_time:.2f}s")
                print(f"Generated text: {result['generated_text'][:100]}...")
        
        avg_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        print(f"\nResults for concurrency {concurrency}:")
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Standard deviation: {std_dev:.2f}s")
        print(f"Throughput: {len(response_times) / sum(response_times):.2f} requests/second")

if __name__ == "__main__":
    asyncio.run(main()) 