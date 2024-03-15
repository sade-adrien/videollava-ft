from llama_cpp import Llama
from tqdm import tqdm
import random
import torch
import json
import csv


RECIPE_DICT = {}
with open('./label_foodtype.csv', mode='r', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        RECIPE_DICT[row.pop('Key')] = row.pop('Name')


file_path = "./test_data_curated_and_enhanced.json"
with open(file_path, 'r') as file:
    data = json.load(file)

model = Llama(model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Download the model file first
            n_ctx=1024,             # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=-1,         # The number of layers to offload to GPU, if you have GPU acceleration available, if -1 all layers are offloaded
            verbose=False,
            )

def main():
    list_videos = []
    for i, (name, info) in tqdm(enumerate(data.items())):
        instruction = get_instruction()
        annotation = get_annotation(video_dict=info, instruction=instruction, using_bloke_quantized_model=True)

        video = {'id': i, 
                'video': f"{info['subset']}/{name}.mp4", 
                'segments': [info['annotations'][i]['segment'] for i in range(len(info['annotations']))],
                'conversations':[{'from': 'human', 'value': '<video>\n' + instruction},
                                {'from': 'gpt', 'value': annotation}]}

        list_videos.append(video)

        print(video) ##for check
    
    output_path = "./test_data_final.json"
    with open(output_path, 'w') as file:
        json.dump(list_videos, file, indent=4)



def get_instruction():
    possible_prompts = ["What is being cooked in this video?",
                        "What recipe is being cooked?",
                        "What is the person preparing?",
                        "What food is being prepared?",
                        "What dish is being made in this video?",
                        "What meal is being prepared in this video?",
                        "What is the chef preparing?",
                        "What type of dish is being cooked up?",
                        "What food item is being prepared?",
                        "Which recipe is the cook following in this video?",
                        "Describe the recipe video.",
                        "Explain what is being prepared and how to cook it.",
                        "Give the recipe of the video.",
                        "Describe the steps to follow to cook this.",
                        "Explain how to reproduce this recipe.",
                        "Give indications on how to reproduce this dish.",
                        "How to cook this meal?",
                        "What specific food item is being prepared?",
                        "Which recipe is the chef following in this video?",
                        "Summarize the content of the recipe video.",
                        "Detail what is being prepared and the cooking process.",
                        "Provide the recipe featured in the video.",
                        "Outline the steps necessary to prepare this dish.",
                        "Elaborate on how to replicate this recipe.",
                        "Offer instructions on how to recreate this dish.",
                        "How can one prepare this meal?",
                        "In this video, what is the chef cooking?",
                        "In this video, which recipe is followed?",
                        "In the video, what dish is being cooked by the person?",
                        "In the video, the person is preparing what?",
                        "The video is presenting what meal?",
                        "The video showcase what recipe?"]
    
    return random.choice(possible_prompts)


def get_annotation(video_dict, instruction, using_bloke_quantized_model=False):

    prompt = f"Below are steps extracted from a {RECIPE_DICT[video_dict['recipe_type']]} recipe. Summarize them and write a short coherent paragraph as a cooking recipe. {instruction}"

    #recipe = random.sample(video_dict['annotations'], k=8)
    recipe = video_dict['annotations']
    recipe = sorted(recipe, key=lambda x: (x['segment'][0]+x['segment'][0])/2, reverse=False) #sorting as unordered since we added steps in data augmentation
    recipe = '\n'.join([step['sentence'] for step in recipe])

    full_prompt = f'[INST]{prompt}\n{recipe}[/INST]'
    
    if using_bloke_quantized_model:
        annotation_dict = model(full_prompt, # Prompt
                            max_tokens=512,  # Generate up to 512 tokens
                            stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
                            echo=False,       # Whether to echo the prompt
                            )
        annotation = annotation_dict['choices'][0]['text']

    else:
        inputs = tokenizer(full_prompt, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        annotation = tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    return annotation
    

if __name__ == '__main__':
    main()