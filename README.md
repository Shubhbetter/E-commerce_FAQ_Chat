# E-commerce FAQ Chat

## Local Development

1. Clone the repository
2. Create virtual environment: `python3 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `streamlit run main.py`

## Streamlit Cloud Deployment

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select this repository
4. Set main file path to: `main.py`
5. Deploy!

# E-commerce FAQ Chatbot

![MADE WITH PYTHON](https://img.shields.io/badge/MADE_WITH-PYTHON-blue)

An automated FAQ chatbot for an e-commerce platform.

**Live site: [ecommercefaq.streamlit.app](https://e-commercefaqchat-ehw9eyzmd9sgqgxlyd3ali.streamlit.app/)**

## Technology Stack

This project leverages several powerful technologies to deliver an intelligent and informative FAQ experience:

* **LangChain**
* **Gemini Pro**
* **ChromaDB**
* **Streamlit**

<img align="left" src="https://github.com/imsoumya18/imsoumya18/blob/main/assets/langchain.png"  alt="LangChain"  height="42px">
<img align="left" src="https://github.com/imsoumya18/imsoumya18/blob/main/assets/gemini.png"  alt="Gemini"  height="42px">
<img align="left" src="https://github.com/imsoumya18/imsoumya18/blob/main/assets/chromadb.png"  alt="ChromaDB"  height="42px">
<img src="https://github.com/imsoumya18/imsoumya18/blob/main/assets/streamlit.png"  alt="Streamlit"  height="42px">

## Dataset
* **[Kaggle: Ecommerce-FAQ-Chatbot-Dataset [JSON]](https://www.kaggle.com/datasets/saadmakhdoom/ecommerce-faq-chatbot-dataset)**
* **[Kaggle: Ecommerce-FAQ-Chatbot-Dataset [CSV]](https://github.com/imsoumya18/E-commerce_FAQ/blob/main/Ecommerce_FAQs.csv)**

The model is just for **demonstration purpose only**. It is trained using **only 80 Question - Answer pairs**. So, expecting it to answer any question other then used for training with high accuracy is not a good idea. You can [have a look at all the 80 questions](https://github.com/imsoumya18/E-commerce_FAQ/blob/main/Ecommerce_FAQs.csv) and ask something similar or combined of multiple questions.

## New: Production-style LLM Fine-Tuning Module (addresses review feedback)

This repository now includes a complete fine-tuning pipeline so you can build one LLM model on the project dataset and extend it in your next module/research paper.

### Model choice
- **Base model (Hugging Face):** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Fine-tuning method:** **QLoRA** (4-bit quantization + LoRA adapters)

### Dataset choice
- **Public dataset source:** Kaggle e-commerce FAQ dataset (the same source represented here as `Ecommerce_FAQs.csv`)
- Pipeline converts it to instruction format and creates train/validation/test splits.

### Files added for the pipeline
- `llm_pipeline_preprocess.py` → data cleaning, deduplication, instruction formatting, split generation
- `llm_pipeline_train_qlora.py` → QLoRA fine-tuning script
- `llm_pipeline_evaluate.py` → relevance/factuality-oriented evaluation (ROUGE-L + hallucination proxy)

### End-to-end commands

1. **Preprocess data**
   ```bash
   python llm_pipeline_preprocess.py --input_csv Ecommerce_FAQs.csv --output_dir training_data
   ```

2. **Train QLoRA adapters**
   ```bash
   python llm_pipeline_train_qlora.py \
     --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --data_dir training_data \
     --output_dir artifacts/tinyllama-ecom-qlora
   ```

3. **Evaluate for relevance/factuality + hallucination reduction proxy**
   ```bash
   python llm_pipeline_evaluate.py \
     --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --adapter_path artifacts/tinyllama-ecom-qlora \
     --test_file training_data/test.jsonl
   ```

### Hallucination reduction strategy used
- Instruction template explicitly enforces concise factual answers.
- FAQ-specific supervised fine-tuning biases output toward known domain facts.
- Deterministic decoding during evaluation (`do_sample=False`) reduces unsupported generations.
- A hallucination proxy metric is reported to track unsupported-token drift against references.

Any type of feedback is appreciated: ssshubham.147.sp@gmail.com

![forthebadge made-by-shubham](https://img.shields.io/badge/CREATED_BY-SHUBHAM-blue)

![forthebadge hosted-on-streamlit](https://img.shields.io/badge/HOSTED_ON-STREAMLIT-red)
