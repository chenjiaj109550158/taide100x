import torch
from .base import WrapperBase
# from accelerate import load_checkpoint_and_dispatch, infer_auto_device_map
from ..offload_tools import load_checkpoint_and_dispatch, infer_auto_device_map


def t(module_name, module):
    for child_name, child in module.named_children():
        # child_name = f"{llm}.{child_name}" if len(llm) > 0 else child_name
        # layers_to_hook.append(child_name)
        child_name = f"{module_name}.{child_name}"
        print(f"child_name:{child_name}")
        t(child_name, child)

def get_layers_to_hook(llm):
    for child_name, child in llm.named_children():
        # child_name = f"{llm}.{child_name}" if len(llm) > 0 else child_name
        # layers_to_hook.append(child_name)
        child_name = f"{child_name}"
        print(f"child_name:{child_name}")
        t(child_name, child)

class OffloadWrapper(WrapperBase):
    def __init__(self):
        super(OffloadWrapper, self).__init__()
    
    def set_llm(self, llm):
        # self.llm = llm
        device_map = infer_auto_device_map(
            llm,
            max_memory={
                0: "5GiB",
                "cpu": "22GiB"
            },
            no_split_module_classes=["LlamaDecoderLayer"],
        )

        layers_to_be_hooked = list(device_map.keys())
        # print(device_map)
        # for k in device_map.keys():
        #     print(k)
        # print('---------------------------------------------')
        # get_layers_to_hook(llm)
        # print('---------------------------------------------')
        # if 'model.embed_tokens' in layers_to_be_hooked:
        #     print('hi')
        # else:
        #     print('no')
        
        # exit()
        
        self.llm = load_checkpoint_and_dispatch(
            llm,
            checkpoint="C:\\Users\\JustinChen\\.cache\\huggingface\\hub\\models--meta-llama--Llama-2-7b-chat-hf\\snapshots\\f5db02db724555f92da89c216ac04704f23d4590\\model.safetensors.index.json", 
            # checkpoint="/home/chenjiaj/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/model.safetensors.index.json", 
            device_map=device_map,
            # no_split_module_classes=["LlamaDecoderLayer"],
            dtype=torch.float16,
            # layers_to_be_hooked = layers_to_be_hooked
        )
        
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


