import torch
# from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
import time

from models import NaiveWrapper, HuggingFaceWrapper, OffloadWrapper

from accelerate import init_empty_weights

# python run_offload.py --mode ol -s --max-new-token 5
# python run_offload.py --mode ol 

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

    if args.mode == "naive":
        model = NaiveWrapper()
    elif args.mode == "huggingface" or args.mode == "hf":
        model = HuggingFaceWrapper()
    elif args.mode == "offload" or args.mode == "ol":
        model = OffloadWrapper()
    else:
        raise ValueError("Invalid mode.")
    
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
        outputs = model.generate(inputs["input_ids"], max_length=args.max_new_token, num_return_sequences=1)
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

    else:
        # input message
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        input_message = "Hello."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_message},
        ]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        _  = model.generate(input_ids, temperature=args.temp, max_length=args.max_new_token, do_sample=args.do_sample)

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
        output_ids = model.generate(input_ids, temperature=args.temp, max_length=args.max_new_token, do_sample=args.do_sample)
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
        "--max-new-token",
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
    args = parser.parse_args()
    
    main(args)