from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': r'C:\Users\qinla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\outpus\BEV_test\isolated pictures\0-case2'},
    {'text': 'Is it possible that the vehicle in the picture was involved in a collision?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)

# 2nd dialogue turn
response, history = model.chat(tokenizer, 'output the objects in a potential collision', history=history)
print(response)

image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    # Modify the save path here
    image.save(r'C:\Users\qinla\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples\outpus\BEV_test\isolated pictures\model_output\output.jpg')
else:
    print("no box")