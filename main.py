import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from groq import Groq
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from gtts import gTTS
from fpdf import FPDF
from docx import Document
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# App 1: YOLOv8 Defect Detection and Report Generation
class EngineInspectionApp:
    def __init__(self):
        # Initialize YOLO model
        self.model_path = 'yolov8n_model/best.pt'  # Path to your trained YOLOv8 model
        self.model = YOLO(self.model_path)

        # Initialize Groq LLM Client
        self.groq_client = Groq(api_key=st.secrets["groq"]["api_key"])

        # Initialize Faiss-based vector store
        self.vector_store_folder = "visionLLM_vector_store"
        self.vector_store_file = os.path.join(self.vector_store_folder, "faiss_index.pkl")
        if os.path.exists(self.vector_store_file):
            with open(self.vector_store_file, "rb") as f:
                self.vector_store = pickle.load(f)
        else:
            self.vector_store = None

        # Preprocessing parameters
        self.input_size = (640, 640)

    def preprocess_image(self, image):
        """
        Preprocess input image for YOLO detection.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        h, w = image.shape[:2]
        aspect_ratio = w / h
        if w > h:
            new_w = 640
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = 640
            new_w = int(new_h * aspect_ratio)

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        start_x = (640 - new_w) // 2
        start_y = (640 - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_image

        return canvas

    def detect_defects(self, image):
        """
        Detect defects in the image using YOLO.
        """
        results = self.model(image)
        annotated_image = np.copy(image)

        labels = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{self.model.names[cls]} {conf:.2f}"
                labels.append((self.model.names[cls], conf))

                # Draw bounding box on the image
                x1, y1, x2, y2 = map(int, box.xywh[0])
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box, thickness = 2
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated_image, labels

    def generate_report(self, detected_labels):
        """
        Generate inspection report using Groq LLM with Llama model.
        """
        prompt = f"Generate a detailed inspection report for gas turbine blade defects. Detected defects: {', '.join([label for label, _ in detected_labels])}. "
        prompt += "For each detected defect, provide:"  # Define your prompt format here
        prompt += "1. Defect type and description"
        prompt += "2. Potential causes"
        prompt += "3. Recommended maintenance actions"
        prompt += "4. Safety warnings and precautions"

        chat_completion = self.groq_client.chat.completions.create(
            messages=[{"role": "system", "content": "You are an expert in gas turbine blade inspection and maintenance."},
                      {"role": "user", "content": prompt}],
            model="llama3-70b-8192"  # Llama model via Groq
        )

        return chat_completion.choices[0].message.content

    def run(self):
        """
        Streamlit app main function for App 1.
        """
        st.title("Engine Component Inspection System")
        st.header("Gas Turbine Blade Defect Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)

            # Preprocess image
            preprocessed_image = self.preprocess_image(image)

            # Detect defects
            annotated_image, detected_labels = self.detect_defects(preprocessed_image)

            # Display images side by side (row 1, 2 columns)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            with col2:
                st.image(annotated_image, caption="Defect Detection Results with Bounding Boxes", use_column_width=True)

            st.subheader("Detected Defects")
            for label, confidence in detected_labels:
                st.write(f"- {label} (Confidence: {confidence:.2%})")

            if st.button("Generate Detailed Report"):
                with st.spinner("Generating Report..."):
                    report = self.generate_report(detected_labels)
                    st.markdown("### Inspection Report")
                    st.write(report)


# App 2: LLM-based Chat and Report Generation
@st.cache_resource(show_spinner=False)
def setup_vectorstore(working_dir):
    """Set up and cache vector store for App 2."""
    persist_directory = os.path.join(working_dir, "chatLLM_vector_store")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

def create_chain(vectorstore):
    """Create and cache the conversational retrieval chain for App 2."""
    if "chain" not in st.session_state:
        groq_api_key = st.secrets["groq"]["api_key"]  # Fetch the Groq API key from secrets.toml
        llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, groq_api_key=groq_api_key)  # Pass the API key here
        retriever = vectorstore.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            memory=memory,
            verbose=False,
            return_source_documents=False
        )
    return st.session_state.chain

def generate_audio(message):
    """Generate audio for the chat message using gTTS."""
    tts = gTTS(message)
    audio_file = f"audio_{len(st.session_state.chat_history)}.mp3"
    tts.save(audio_file)
    return audio_file

def create_pdf(message):
    """Create PDF from the chat message."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, message)
    pdf_output = f"pdf_{len(st.session_state.chat_history)}.pdf"
    pdf.output(pdf_output)
    return pdf_output

def create_docx(message):
    """Create DOCX from the chat message."""
    doc = Document()
    doc.add_paragraph(message)
    doc_output = f"docx_{len(st.session_state.chat_history)}.docx"
    doc.save(doc_output)
    return doc_output

def add_chat_message(message):
    """Add a new chat message to the session state and update display.""" 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append(message)

def app2():
    """Streamlit app main function for App 2."""
    st.title("Interactive Chat with Report Generation 📝")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize as an empty list

    vectorstore = setup_vectorstore(".")
    chain = create_chain(vectorstore)

    user_input = st.text_area("Enter your query:")
    if user_input:
        response = chain.run(question=user_input)
        add_chat_message({"role": "assistant", "content": response})

        # Display chat response
        st.write(f" {response}")

        # Buttons for generating and downloading files
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Play Audio"):
                audio_file = generate_audio(response)
                st.audio(audio_file, format='audio/mp3')
        with col2:
            if st.button("📄 Generate PDF"):
                pdf_file = create_pdf(response)
                st.download_button("Download PDF", pdf_file)
        with col3:
            if st.button("📄 Generate DOCX"):
                docx_file = create_docx(response)
                st.download_button("Download DOCX", docx_file)

        # Display chat history
        for chat_message in st.session_state.chat_history:
            st.write(f"**{chat_message['role']}**: {chat_message['content']}")

# Run the app
if __name__ == "__main__":
    st.set_page_config(page_title="Defect Detection and Chatbot App", layout="wide")
    selected_app = st.sidebar.selectbox("Choose App", ["Engine Inspection", "Interactive Chat"])

    if selected_app == "Engine Inspection":
        app1 = EngineInspectionApp()
        app1.run()
    elif selected_app == "Interactive Chat":
        app2()
