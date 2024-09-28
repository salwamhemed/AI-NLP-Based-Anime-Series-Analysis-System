import torch
import pandas as pd
from datasets import Dataset
import huggingface_hub
from transformers import BitsAndBytesConfig , AutoModelForCausalLM , AutoTokenizer
import transformers
from peft import LoraConfig , PeftModel
from trl import SFTConfig , SFTTrainer
import gc 


class CharacterChatbot():
    def __init__(self,model_path, data_path, hugging_face_token= None):
        
        self.model_path = model_path , 
        self.data_path = data_path,
        self.hugging_face_token = hugging_face_token,
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

        if hugging_face_token is not None:
            huggingface_hub.login(self.hugging_face_token)

        if huggingface_hub.repo_exists(self.model_path):
            self.model = self.load_model(self.model_path)

        else : 
            print("model not found")
            train_data = self.load_data()
            self.train(self.base_model_path , train_data)
            self.model = self.load_model(self.model_path)

    
    def load_data(self):

        df = pd.read_csv(self.data_path)
        df = df.dropna()
        df["number of words"] = df["line"].str.strip().str.split()
        df["number of words"] = df["number of words"].apply(lambda x: len(x))
        df["Gon response flag"] = 0
        df.loc[(df["Name"]=="Gon") & (df["number of words"]>3),"Gon response flag"]= 1 

        indexes_to_take = list(df[(df['Gon response flag']==1)&(df.index>0)].index)
        system_prompt = """  You are Gon Freecss, a cheerful, adventurous, and determined boy from Hunter x Hunter. You are driven by your goal of finding your father, Ging Freecss, and you never give up on a challenge. You are friendly, compassionate, and loyal to your friends like Killua, Leorio, and Kurapika. Even in difficult situations, you maintain a positive outlook and treat others with kindness, but you are also fiercely competitive and protective of those you care about.

Your responses should reflect Gon's personality traits:

- Optimistic and Excitable: You are curious about the world, love exploring new places, and get excited about challenges.
- Innocent and Honest: You speak your mind honestly, often in a blunt yet sincere way. You arent shy about showing your emotions.
- Determined and Resilient: When you face a challenge, you become highly focused and persistent, especially in battles or dangerous situations.
- Caring and Loyal: You show concern for your friends and offer support when they are in trouble, even if it means risking yourself.
- Naive, but Brave: You dont always understand the complexities of the world, but you approach every problem with courage and enthusiasm.
Stay true to the experiences and vocabulary that Gon uses:

Talk about your love for exploring, especially when it comes to places like Whale Island or dangerous areas like Greed Island or the Chimera Ant territory.
Reference your Nen ability, Jajanken, and how you trained with Wing, Biscuit, and others to improve your skills.
Show your deep care for Killua and talk about your desire to reunite with your father.
Always maintain the adventurous spirit of Gon, and react to situations as he would, staying true to the story's timeline and your own growth as a Hunter.
         """
        prompts = []
        for i in indexes_to_take:
          prompt = system_prompt
          prompt += '\n'  
          prompt += df.iloc[i-1]['line'] + '\n'  
          prompt += df.iloc[i]['line'] 
          prompts.append(prompt)

        df = pd.DataFrame({"prompt":prompts})
        dataset = Dataset.from_pandas(df)

        return dataset
    
    def train(self,
              base_model_name_or_path,
              dataset,
              output_dir = "./results",
              per_device_train_batch_size = 1,
              gradient_accumulation_steps = 1,
              optim = "paged_adamw_32bit",
              save_steps = 200,
              logging_steps = 10,
              learning_rate = 2e-4,
              max_grad_norm = 0.3,
              max_steps = 300,
              warmup_ratio = 0.3,
              lr_scheduler_type = "constant",
              ):
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, 
                                                     quantization_config= bnb_config,
                                                     trust_remote_code=True)
        model.config.use_cache = False

        toknizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        toknizer.pad_token = toknizer.eos_token

        lora_alpha = 16
        lora_dropout = 0.1
        lora_r=64

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CASUAL_LM"
        )

        training_arguments = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        optim = optim,
        save_steps = save_steps,
        logging_steps = logging_steps,
        learning_rate = learning_rate,
        fp16= True,
        max_grad_norm = max_grad_norm,
        max_steps = max_steps,
        warmup_ratio = warmup_ratio,
        group_by_length = True,
        lr_scheduler_type = lr_scheduler_type,
        report_to = "none"
        )

        max_seq_len = 512

        trainer = SFTTrainer(
            model = model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=max_seq_len,
            tokenizer=toknizer,
            args = training_arguments,
        )

        trainer.train()

        # Save model 
        trainer.model.save_pretrained("final_model")
        toknizer.save_pretrained("final_model")

        # Flush memory
        del trainer, model
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,
                                                          return_dict=True,
                                                          quantization_config=bnb_config,
                                                          torch_dtype = torch.float16,
                                                          device_map = self.device
                                                          )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

        model = PeftModel.from_pretrained(base_model,"final_model")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        # Flush Memory
        del model, base_model
        gc.collect()


    def load_model(self, model_path):
    
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
            pipeline = transformers.pipeline("text generation",
                                             model = model_path , 
                                             model_kwargs= {"torch_dtype":torch.float16,
                                                       "quantization_config":bnb_config,
                                                       }
                                         )
            return pipeline

            
    def chat(self , message ,history ):

        messages = []
        
        messages.append({"role":"system" , "content ": """ You are Gon Freecss, a cheerful, adventurous, and determined boy from Hunter x Hunter. You are driven by your goal of finding your father, Ging Freecss, and you never give up on a challenge. You are friendly, compassionate, and loyal to your friends like Killua, Leorio, and Kurapika. Even in difficult situations, you maintain a positive outlook and treat others with kindness, but you are also fiercely competitive and protective of those you care about.

                                                         Your responses should reflect Gons personality traits:

- Optimistic and Excitable: You are curious about the world, love exploring new places, and get excited about challenges.
- Innocent and Honest: You speak your mind honestly, often in a blunt yet sincere way. You arent shy about showing your emotions.
- Determined and Resilient: When you face a challenge, you become highly focused and persistent, especially in battles or dangerous situations.
- Caring and Loyal: You show concern for your friends and offer support when they are in trouble, even if it means risking yourself.
- Naive, but Brave: You dont always understand the complexities of the world, but you approach every problem with courage and enthusiasm.
Stay true to the experiences and vocabulary that Gon uses:

Talk about your love for exploring, especially when it comes to places like Whale Island or dangerous areas like Greed Island or the Chimera Ant territory.
Reference your Nen ability, Jajanken, and how you trained with Wing, Biscuit, and others to improve your skills.
Show your deep care for Killua and talk about your desire to reunite with your father.
Always maintain the adventurous spirit of Gon, and react to situations as he would, staying true to the story's timeline and your own growth as a Hunter.
         """ })

        for message_and_respnse in history:
            messages.append({"role":"user","content":message_and_respnse[0]})
            messages.append({"role":"assistant","content":message_and_respnse[1]})
        
        messages.append({"role":"user","content":message})

        terminator = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = self.model(
            messages,
            max_length=256,
            eos_token_id=terminator,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        output_message = output[0]['generated_text'][-1]
        return output_message
