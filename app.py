import streamlit as st
import pandas as pd
import torch
import warnings
import sys
from transformers import pipeline

# --- 0. SETUP ---
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="AI Guru", layout="wide")

# --- CONFIGURATION ---
CSV_FILE = 'ai_guru_unique_dsa_with_difficulty.csv'
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_FILE)
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_FILE, encoding='latin1')
    except FileNotFoundError:
        st.error(f"❌ File not found: {CSV_FILE}. Please check the filename.")
        return pd.DataFrame()

    try:
        df.columns = [c.lower().strip() for c in df.columns]
    except:
        pass
    return df

df = load_data()

# --- 2. LOAD LOCAL SLM (FIXED LOADING LOGIC) ---
@st.cache_resource
def load_local_slm():
    print("--------------------------------------------------")
    print("INITIALIZING MODEL...")
    
    # Check for GPU
    if torch.cuda.is_available():
        device_type = "cuda"
        print(f"✅ SUCCESS: GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device_type = "cpu"
        print("⚠️ WARNING: GPU not found. Using CPU (Slow).")
    
    try:
        if device_type == "cuda":
            # GPU LOADING (Accelerate handles the device, so we DO NOT pass 'device=0')
            pipe = pipeline(
                "text-generation",
                model=MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto" 
            )
        else:
            # CPU LOADING (We explicitly pass device=-1)
            pipe = pipeline(
                "text-generation",
                model=MODEL_ID,
                torch_dtype=torch.float32,
                device=-1
            )
            
        print("✅ Model Loaded Successfully.")
        return pipe, device_type
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, "error"

# Initialize model
slm_pipeline, device_used = load_local_slm()

# --- 3. GENERATION LOGIC ---
def generate_question_internal(difficulty, topic, dataset):
    if slm_pipeline is None:
        return "Error: Model not loaded."

    # RAG: Get examples
    subset = dataset[(dataset['difficulty'] == difficulty) & (dataset['topic'] == topic)]
    context_text = ""
    if not subset.empty:
        row = subset.sample(1).iloc[0]
        context_text = f"Reference Example:\nQuestion: {row['question']}\nAnswer: {row['correct_answer']}"
    
    # Prompt
    system_msg = "You are a Computer Science teacher. Create a multiple-choice question."
    user_msg = f"""
    Topic: {topic} ({difficulty})
    
    {context_text}
    
    INSTRUCTIONS:
    Write a NEW question.
    Use this EXACT format:
    
    QUESTION: [Your Question Text]
    OPTION A: [Option 1]
    OPTION B: [Option 2]
    OPTION C: [Option 3]
    OPTION D: [Option 4]
    ANSWER: [Correct Option Content]
    EXPLANATION: [Short Reasoning]
    """
    
    prompt = f"<|system|>\n{system_msg}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>\n"

    # Inference
    sequences = slm_pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=slm_pipeline.tokenizer.eos_token_id
    )
    
    generated_text = sequences[0]['generated_text']
    result_part = generated_text.split("<|assistant|>")[-1].strip()
    return result_part

# --- 4. UI PAGES ---

# Session State
if 'page' not in st.session_state: st.session_state.page = 'login'
if 'sel_diff' not in st.session_state: st.session_state.sel_diff = None
if 'sel_topic' not in st.session_state: st.session_state.sel_topic = None

# PAGE: Login
def page_login():
    st.title("🔐 AI Guru SLM - Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Sign In"):
        if u: 
            st.session_state.page = 'difficulty'
            st.rerun()

# PAGE: Difficulty
def page_difficulty():
    st.title("Choose Difficulty")
    st.caption(f"Running on: {device_used.upper()}")
    
    if df.empty: 
        st.warning("Data not loaded. Check CSV file.")
        return
    
    diffs = df['difficulty'].unique()
    cols = st.columns(len(diffs))
    for idx, d in enumerate(diffs):
        if cols[idx].button(d, use_container_width=True):
            st.session_state.sel_diff = d
            st.session_state.page = 'topic'
            st.rerun()

# PAGE: Topic
def page_topic():
    d = st.session_state.sel_diff
    st.title(f"Select Topic ({d})")
    if st.button("Back"):
        st.session_state.page = 'difficulty'
        st.rerun()
        
    topics = df[df['difficulty'] == d]['topic'].unique()
    for t in topics:
        if st.button(t, use_container_width=True):
            st.session_state.sel_topic = t
            st.session_state.page = 'quiz'
            st.rerun()

# PAGE: Quiz
def page_quiz():
    d = st.session_state.sel_diff
    t = st.session_state.sel_topic
    st.title(f"Quiz: {t} - {d}")
    
    if st.button("Change Topic"):
        st.session_state.page = 'topic'
        st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    # Button 1: Get from CSV
    if col1.button("Get Question from Data"):
        subset = df[(df['difficulty'] == d) & (df['topic'] == t)]
        if not subset.empty:
            row = subset.sample(1).iloc[0]
            st.info(f"**Q:** {row['question']}")
            st.write(f"**A:** {row['correct_answer']}")
            with st.expander("Explanation"):
                st.write(row['explanation'])
        else:
            st.warning("No data found.")

    # Button 2: Generate via Local SLM
    if col2.button("Generate New AI Question"):
        with st.spinner("AI is thinking..."):
            raw_response = generate_question_internal(d, t, df)
            
            # --- PARSER ---
            try:
                question = "Could not parse question."
                answer = "Unknown"
                explanation = "None"
                options = []

                lines = raw_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.upper().startswith("QUESTION:"):
                        question = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("OPTION"):
                        options.append(line)
                    elif line.upper().startswith("ANSWER:"):
                        answer = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("EXPLANATION:"):
                        explanation = line.split(":", 1)[1].strip()

                st.success(f"**AI Q:** {question}")
                
                if options:
                    for opt in options: st.text(opt)
                else:
                    st.warning("No specific options generated")
                    
                with st.expander("Reveal AI Answer"):
                    st.write(f"**Answer:** {answer}")
                    st.write(f"**Explanation:** {explanation}")
                    
            except Exception as e:
                st.warning("Could not parse AI output automatically.")
                st.write("Raw Output:")
                st.write(raw_response)

# Router
if st.session_state.page == 'login': page_login()
elif st.session_state.page == 'difficulty': page_difficulty()
elif st.session_state.page == 'topic': page_topic()
elif st.session_state.page == 'quiz': page_quiz()