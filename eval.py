import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from datasets import Dataset
from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

colab = False
if(colab):
    from google.colab import drive # type: ignore
    from google.colab import files   # type: ignore
    drive.mount('/content/drive')

print(torch.cuda.device_count())
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def get_embeddings(text_list, tokenizer, model, device):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # GPT-2 için, hidden_states[-1] son katmandaki embeddinglerdir
        hidden_states = outputs.hidden_states[-1]
        # Mean pooling kullanarak cümle embeddingi elde edelim
        sentence_embedding = hidden_states.mean(dim=1).cpu().numpy()
        embeddings.append(sentence_embedding)
    return np.vstack(embeddings)

############################################################
# last_model_path = "sgd_deneme/gpt/checkpoint-4000"

# last_model_path = "sgd/gpt/checkpoint-10000"

# last_model_path = "sgd_deneme/gpt/checkpoint-4000"

# last_model_path = "/home/alperen/Documents/msc/computational_semantics/hw3/hw4_dataset_v2_medium/30plus/checkpoint-120000"
last_model_path = "models/medium_models/v1_180k"
last_model_path2 = "models/medium_models/v2_390k"
last_model_path3 = "models/medium_models/merged_linear"
last_model_path4 = "models/medium_models/merged_slerp"
last_model_path5 = "models/medium_models/merged_ties"
 

config = AutoConfig.from_pretrained(last_model_path)

tokenizer = AutoTokenizer.from_pretrained(last_model_path)
model = GPT2LMHeadModel.from_pretrained(last_model_path, config= config)
model = model.to(device)


#text = "Bir abla ile kardeşin yaş farkı 10'dur, 5 yıl sonra yaş farkları kaç olur?"
text = "Şu listeyi tamamlayın: Televizyon, radyo, gazete"

preds = []
inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=50,return_dict_in_generate=True, output_scores=True)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=175,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
)

print("SORU model1 ", text)
print("------------")
# Extract generated sequence and its corresponding scores
generated_sequence = outputs.sequences[0].cpu().numpy()
generated_scores = outputs.scores[0].cpu().numpy()

# Decode the generated sequence
decoded_sequence = tokenizer.decode(generated_sequence, skip_special_tokens=True)
print("Generated text:", decoded_sequence)
print("Generated scores:", generated_scores)

# Get embeddings for questions and answers
soru_embedding1 = get_embeddings(text, tokenizer, model, device)
cevap_embedding1 = get_embeddings(decoded_sequence, tokenizer, model, device)
similarities = cosine_similarity(soru_embedding1, cevap_embedding1)[0][0] 
print("Model 1 Cos Sim: ", similarities)

print("###############################")

config = AutoConfig.from_pretrained(last_model_path2)

tokenizer = AutoTokenizer.from_pretrained(last_model_path2)
model = GPT2LMHeadModel.from_pretrained(last_model_path2, config= config)
model = model.to(device)


preds = []
inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=50,return_dict_in_generate=True, output_scores=True)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
)

print("SORU model2 ", text)
print("------------")
# Extract generated sequence and its corresponding scores
generated_sequence = outputs.sequences[0].cpu().numpy()
generated_scores = outputs.scores[0].cpu().numpy()

# Decode the generated sequence
decoded_sequence = tokenizer.decode(generated_sequence, skip_special_tokens=True)
print("Generated text:", decoded_sequence)
print("Generated scores:", generated_scores)


# Get embeddings for questions and answers
soru_embedding2 = get_embeddings(text, tokenizer, model, device)
cevap_embedding2 = get_embeddings(decoded_sequence, tokenizer, model, device)

# Calculate cosine similarities
similarities2 = cosine_similarity(soru_embedding2, cevap_embedding2)[0][0] 
print("Model 2 Cos Sim: ", similarities2)

print("###############################")
########## Model 3
config = AutoConfig.from_pretrained(last_model_path3)

tokenizer = AutoTokenizer.from_pretrained(last_model_path3)
model = GPT2LMHeadModel.from_pretrained(last_model_path3, config= config)
model = model.to(device)


preds = []
inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=50,return_dict_in_generate=True, output_scores=True)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
)

print("SORU model3 Linear :  ", text)
print("------------")
# Extract generated sequence and its corresponding scores
generated_sequence = outputs.sequences[0].cpu().numpy()
generated_scores = outputs.scores[0].cpu().numpy()

# Decode the generated sequence
decoded_sequence = tokenizer.decode(generated_sequence, skip_special_tokens=True)
print("Generated text:", decoded_sequence)


# Get embeddings for questions and answers
soru_embedding3 = get_embeddings(text, tokenizer, model, device)
cevap_embedding3 = get_embeddings(decoded_sequence, tokenizer, model, device)

# Calculate cosine similarities
similarities3 = cosine_similarity(soru_embedding3, cevap_embedding3)[0][0] 
print("Model 3 Cos Sim: ", similarities3)

print("###############################")
########## Model 4

config = AutoConfig.from_pretrained(last_model_path4)

tokenizer = AutoTokenizer.from_pretrained(last_model_path4)
model = GPT2LMHeadModel.from_pretrained(last_model_path4, config= config)
model = model.to(device)


preds = []
inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=50,return_dict_in_generate=True, output_scores=True)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
)

print("SORU model4 Slerp : ", text)
print("------------")
# Extract generated sequence and its corresponding scores
generated_sequence = outputs.sequences[0].cpu().numpy()
generated_scores = outputs.scores[0].cpu().numpy()

# Decode the generated sequence
decoded_sequence = tokenizer.decode(generated_sequence, skip_special_tokens=True)
print("Generated text:", decoded_sequence)

# Get embeddings for questions and answers
soru_embedding4 = get_embeddings(text, tokenizer, model, device)
cevap_embedding4 = get_embeddings(decoded_sequence, tokenizer, model, device)

# Calculate cosine similarities
similarities4 = cosine_similarity(soru_embedding4, cevap_embedding4)[0][0] 
print("Model 4 Cos Sim: ", similarities4)



print("###############################")
########## Model 5

config = AutoConfig.from_pretrained(last_model_path5)

tokenizer = AutoTokenizer.from_pretrained(last_model_path5)
model = GPT2LMHeadModel.from_pretrained(last_model_path5, config= config)
model = model.to(device)


preds = []
inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=50,return_dict_in_generate=True, output_scores=True)

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=150,
    pad_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
    output_scores=True,
)

print("SORU model5 Ties: ", text)
print("------------")
# Extract generated sequence and its corresponding scores
generated_sequence = outputs.sequences[0].cpu().numpy()
generated_scores = outputs.scores[0].cpu().numpy()

# Decode the generated sequence
decoded_sequence = tokenizer.decode(generated_sequence, skip_special_tokens=True)
print("Generated text:", decoded_sequence)

# Get embeddings for questions and answers
soru_embedding5 = get_embeddings(text, tokenizer, model, device)
cevap_embedding5 = get_embeddings(decoded_sequence, tokenizer, model, device)

# Calculate cosine similarities
similarities5 = cosine_similarity(soru_embedding5, cevap_embedding5)[0][0] 
print("Model 4 Cos Sim: ", similarities5)


###################




# # Embeddingleri birleştir
# all_embeddings = np.vstack((soru_embedding1, cevap_embedding1, soru_embedding2, cevap_embedding2,soru_embedding3, cevap_embedding3, soru_embedding4, cevap_embedding4 ))

# Embeddingleri birleştir
# all_embeddings = np.vstack((soru_embedding1, cevap_embedding1, soru_embedding2, cevap_embedding2,
#                             soru_embedding3, cevap_embedding3, soru_embedding4, cevap_embedding4, soru_embedding5, cevap_embedding5))

all_embeddings = np.vstack(( cevap_embedding1, cevap_embedding2,
                              cevap_embedding3, cevap_embedding4, 
                              cevap_embedding5))

# TSNE ile boyut indirgeme
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(all_embeddings)

# TSNE sonuçlarını görselleştirme
plt.figure(figsize=(10, 6))
# colors = ['red', 'red', 'green', 'green', 'orange', 'orange', 'blue', 'blue', 'purple', 'purple']
# labels = ['Soru 1', 'Cevap 1', 'Soru 2', 'Cevap 2', 'Soru Linear', 'Cevap Linear', 'Soru Slerp', 'Cevap Slerp', 'Soru Ties', 'Cevap Ties']

colors = ['red', 'green', 'orange', 'blue', 'purple']
labels = ['Cevap 1', 'Cevap 2', 'Cevap Linear', 'Cevap Slerp', 'Cevap Ties']


for i in range(5):
    plt.scatter(tsne_embeddings[i, 0], tsne_embeddings[i, 1], c=colors[i], label=labels[i])
    plt.annotate(labels[i], (tsne_embeddings[i, 0], tsne_embeddings[i, 1]), fontsize=12, color=colors[i])

plt.legend()
plt.title("TSNE Visualization of Question and Answer Embeddings")
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.show()