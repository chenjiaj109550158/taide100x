import torch
from .base import WrapperBase
# from accelerate import load_checkpoint_and_dispatch, infer_auto_device_map
from ..offload_tools import load_checkpoint_and_dispatch, infer_auto_device_map


def t(module_name, module):
    for child_name, child in module.named_children():
        # child_name = f"{llm}.{child_name}" if len(llm) > 0 else child_name
        # layers_to_hook.append(child_name)
        child_name = f"{module_name}.{child_name}"
        print(f"module_name:{child_name}")
        try:
            print(f"module._hf_hook: {type(child._hf_hook)}")
        except:
            print(f"module._hf_hook: None")

        try:
            if child._hf_hook.offload:
                print(f"module._hf_hook.offload: {child._hf_hook.offload} ************************** ")
            else:
                print(f"module._hf_hook.offload: {child._hf_hook.offload}")
        except:
            print(f"module._hf_hook.offload: None")
        print('----------------------')
        t(child_name, child)

def print_child_module_names(module):
    for child_name, child in module.named_children():
        # child_name = f"{llm}.{child_name}" if len(llm) > 0 else child_name
        # layers_to_hook.append(child_name)
        child_name = f"{child_name}"
        print(f"module_name:{child_name}")
        try:
            print(f"module._hf_hook: {type(child._hf_hook)}")
        except:
            print(f"module._hf_hook: None")

        try:
            if child._hf_hook.offload:
                print(f"module._hf_hook.offload: {child._hf_hook.offload} ************************** ")
            else:
                print(f"module._hf_hook.offload: {child._hf_hook.offload}")

        except:
            print(f"module._hf_hook.offload: None")
        print('----------------------')
        t(child_name, child)

class OffloadWrapper(WrapperBase):
    def __init__(self):
        super(OffloadWrapper, self).__init__()
    
    def set_llm(self, llm):
        # self.llm = llm
        device_map = infer_auto_device_map(
            llm,
            max_memory={
                0: "8GiB",
                "cpu": "22GiB"
            },
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype=torch.float16,
        )

        layers_to_be_hooked = list(device_map.keys())
        print(device_map)
        
        self.llm = load_checkpoint_and_dispatch(
            llm,
            checkpoint="C:\\Users\\JustinChen\\.cache\\huggingface\\hub\\models--meta-llama--Llama-2-7b-chat-hf\\snapshots\\f5db02db724555f92da89c216ac04704f23d4590\\model.safetensors.index.json", 
            # checkpoint="/home/chenjiaj/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/model.safetensors.index.json", 
            device_map=device_map,
            # no_split_module_classes=["LlamaDecoderLayer"],
            dtype=torch.float16,
            layers_to_be_hooked = layers_to_be_hooked
        )
        # print_child_module_names(self.llm)
        # exit()
        
    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        temperature=None, top_p=None, top_k=None, 
        max_length=2048, do_sample=True, 
        *args, 
        **kwargs
    ):
        assert self.llm is not None, "LLM model must be provided"
        
        return self.llm.generate(
            input_ids=input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=max_length,
            do_sample=do_sample,
            *args,
            **kwargs,
        )


