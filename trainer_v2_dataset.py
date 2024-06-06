import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, AutoConfig, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
from datasets import Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.trainer_callback import TrainerCallback


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

if(colab):
    df = pd.read_csv("/content/drive/MyDrive/ComputationalSemantics/Hw3/instructions.csv")
else :
    df = pd.read_csv("instructions.csv")


def plot_sentence_embeddings_length(text_list, tokenizer):  # reference: https://www.kaggle.com/code/ozcan15/turkish-sentiment-analysis-with-berturk-93-acc
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t), text_list))
    tokenized_texts_len = list(map(lambda t: len(t), tokenized_texts))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tokenized_texts_len, bins=40)
    ax.set_xlabels("Length of Comment Embeddings")
    ax.set_ylabels("Number of Comments")
    plt.title("Length of Embeddings")
    plt.show()



# df = df.head(500)
df = df.drop(df.columns[-21:], axis=1)


df = df.fillna('')

df['soru'] = df['talimat'] + df['giris']
df['labels'] = df['cikti']

for index, row in df.iterrows():
    if row['giris'] != '':
        df.at[index, 'soru'] = row['talimat'] + row['giris']
        df.at[index, 'labels'] = row['cikti']
        
new_df = df[["soru", "labels"]]

train_df, test_df = train_test_split(new_df, test_size = 0.25, random_state = 42)

dataset = Dataset.from_pandas(train_df)
dataset_eval = Dataset.from_pandas(test_df)

model_id = "/home/alperen/Documents/msc/computational_semantics/hw3/hw4_dataset_v2_medium/30plus/120plus/checkpoint-60000"
#model_id = "asafaya/kanarya-750m"
optim = "adamw_torch"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


class MetricsCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_metrics = {'loss': []}
        self.eval_metrics = {'eval_accuracy': []}

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.train_metrics['loss'].append(logs.get('loss'))

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.eval_metrics['eval_accuracy'].append(logs.get('eval_accuracy'))

def plot_metrics(train_metrics, eval_metrics, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot training loss
    axes[0].plot(train_metrics['loss'], label='Training Loss', color='blue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot evaluation accuracy
    axes[1].plot(eval_metrics['eval_accuracy'], label='Evaluation Accuracy', color='green')
    axes[1].set_title('Evaluation Accuracy')
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Trainer'ı eğitirken metrikleri saklamak için boş bir sözlük oluşturun
metrics_callback = MetricsCallback()

args = TrainingArguments(
    output_dir=f"/home/alperen/Documents/msc/computational_semantics/hw3/hw4_dataset_v2_medium/30plus/120plus/60plus",
    max_steps=180000,
    per_device_train_batch_size=1,
    optim=optim,
    eval_steps=250,  # Her 500 adımda bir değerlendirme yap
    logging_steps=250,  # Her 500 adımda bir metrikleri kaydet
    save_steps=20000,  # Her 500 adımda bir modeli kaydet
    eval_accumulation_steps=1,  # Değerlendirme için birikmiş gradyanlar
    disable_tqdm=False,  # İlerleme çubuğunu devre dışı bırak
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['soru'])):
        text = f"### Question: {example['soru'][i]}\n ### Answer: {example['labels'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


trainer = SFTTrainer(
    model,
    args=args,
    train_dataset=dataset,
    eval_dataset= dataset_eval,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=300,
    neftune_noise_alpha=5,
    callbacks=[metrics_callback]  # Metrikleri toplamak için geri çağrıyı ekleyin
)

trainer.train()

# Metrikleri grafiğe dök
save_path = "metrics_plot_gpthw4_v2_med.png"  # Buraya dosya yolunu ve adını istediğiniz gibi ayarlayabilirsiniz
plot_metrics(metrics_callback.train_metrics, metrics_callback.eval_metrics, save_path)