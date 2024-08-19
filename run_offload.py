import torch
# from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
import time
from accelerate import init_empty_weights
import numpy as np
import os
import logging
from transformers.cache_utils import DynamicCache


from models import NaiveWrapper, HuggingFaceWrapper, OffloadWrapper
# python run_offload.py --mode ol -s --max-new-token 5
# python run_offload.py --mode ol 


logging.getLogger().setLevel(logging.INFO)
# allocating 40MB to match L2 cache size on A100
x = torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device='cuda')
def flush_cache():
    x.zero_()

def prepare_data(config, batch_size, prev_tokens, new_tokens, dtype=torch.float16, device="cuda"):
    head_dim = config.hidden_size // config.num_attention_heads
    past_key_values = DynamicCache()
    for i in range(0, config.num_hidden_layers):
        cache_k = torch.randn(batch_size, config.num_attention_heads, prev_tokens, head_dim, dtype=dtype, device=device)
        cache_v = torch.randn(batch_size, config.num_attention_heads, prev_tokens, head_dim, dtype=dtype, device=device)
        past_key_values.update(cache_k, cache_v, i)
    
    tokens = torch.randint(100, (batch_size, new_tokens), device=device)
    return past_key_values, tokens

@torch.no_grad() # Time per output token
def benchmark_tpot(model, past_key_values, tokens, repetitions=100):
    # Get the number of previous tokens
    prev_tokens = past_key_values.get_seq_length()

    # Warmup steps
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(repetitions):
            _ = model.run(tokens, past_key_values=past_key_values)
            past_key_values.crop(prev_tokens)
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _ = model.run(tokens, past_key_values=past_key_values)
    # actually not required to crop past_key_values, since cudagraph will replay and read and write to the same memory locations
    # past_key_values.crop(prev_tokens)
    
    # Start and end events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    
    # Run the benchmark
    for i in range(repetitions):
        flush_cache()
        start_events[i].record()
        graph.replay() 
        # _ = model.run(tokens, past_key_values=past_key_values)
        end_events[i].record()
    torch.cuda.synchronize()
    
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    latency = np.median(times) # median is more robust to outliers
    
    return latency

def main(args):
    # deterministic
    torch.manual_seed(0)

    print("Loading model...")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)

    # load LLM
    if args.mode == "offload" or args.mode == "ol":
        #  (use init_empty_weights in offload case)
        # Load model using config
        config = AutoConfig.from_pretrained(args.llm_path)
        with init_empty_weights():
            llm = AutoModelForCausalLM.from_config(config)
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )  

    # mode
    if args.mode == "naive":
        model = NaiveWrapper()
    elif args.mode == "huggingface" or args.mode == "hf":
        model = HuggingFaceWrapper()
    elif args.mode == "offload" or args.mode == "ol":
        model = OffloadWrapper()
    else:
        raise ValueError("Invalid mode.")

    # dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")

    # set model
    model.set_tokenizer(tokenizer)
    model.set_llm(llm)
    model.eval()

    print("Warming up model...")

    if args.simple:

        # Define the prompt
        prompt = "hi"

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt")

        # # Generate text
        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #     on_trace_ready=trace_handler  # 設置回調函數
        # ) as prof:
        #     with torch.no_grad():
        #         outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
        start_time = time.time()
        outputs = model.generate(inputs["input_ids"], max_length=args.max_new_tokens, num_return_sequences=1)
        end_time = time.time()

        if not args.no_print_message:
            # Decode the generated text
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\nPrompt:")
            print(prompt)
            print("\nModel response:")
            print(output)
            print("\n-----------------------------------")
            # print("Input tokens:", len(input_ids[0]))
            # print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
        
        if not args.no_print_time:
            print("Time:", end_time - start_time)

    elif 1:
        # Test for number of new tokens from 1 to max_new_tokens
        for prev_tokens in args.prev_tokens:
            latencies = []
            for i in range(1, args.max_new_tokens+1):
                # past_key_values, tokens = prepare_data(model.config, args.batch_size, prev_tokens, i, dtype=dtype, device=args.device) 
                past_key_values, tokens = prepare_data(model.llm.config, 1, prev_tokens, i, dtype=dtype, device='cuda') 
                latency = benchmark_tpot(model, past_key_values, tokens, repetitions=args.repetitions)
                
                logging.info(f"Finished. \nprevious_tokens: {prev_tokens} \nnew_tokens: {i} \nlatency: {latency:.2f} milliseconds")
                latencies.append(latency)
            
            # convert to numpy array, plot and save
            latencies = np.array(latencies)

            # save latencies
            np.save(os.path.join(args.save_folder, f"llm_prev_{prev_tokens}.npy"), latencies)
    else:
        # input message
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        input_message = "Hello."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_message},
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        _  = model.generate(input_ids, temperature=args.temp, max_length=args.max_new_tokens, do_sample=args.do_sample)

        # generate response
        print("Generating response...")

        # input message
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        input_message = "What's the best way to start learning a new language?"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_message},
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        prompt = tokenizer.decode(input_ids[0])
        
        start_time = time.time()
        output_ids = model.generate(input_ids, temperature=args.temp, max_length=args.max_new_tokens, do_sample=args.do_sample)
        end_time = time.time()
        

        if not args.no_print_message:
            output = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:])
            print("\nPrompt:")
            print(prompt)
            print("\nModel response:")
            print(output)
            print("\n-----------------------------------")
            print("Input tokens:", len(input_ids[0]))
            print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
        
        if not args.no_print_time:
            print("Time:", end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--llm-path",
        "-llm",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="LLM model path.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to do sampling. (Default is False)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="naive",
        help="The mode of model generation.",
    )
    parser.add_argument(
        "-nm",
        "--no-print-message",
        action="store_true",
        help="Print the message.",
    )
    parser.add_argument(
        "-nt",
        "--no-print-time",
        action="store_true",
        help="Record the time.",
    )
    parser.add_argument(
        "-s",
        "--simple",
        action="store_true",
        help="the simple mode(only one prompt)",
    )
    parser.add_argument(
        "--prev_tokens", 
        type=int, 
        nargs="+", 
        default=[128,256,512,1024,2048,4096]
    )
    parser.add_argument(
        "--repetitions", 
        "-rep", 
        type=int, 
        default=10
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="float16"
    )

    args = parser.parse_args()
    
    main(args)