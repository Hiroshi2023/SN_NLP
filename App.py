import streamlit as st
from streamlit_lottie import st_lottie
import requests
import streamlit_authenticator as stauth
from PIL import Image
from streamlit_option_menu import option_menu
import io
from langchain_groq import ChatGroq
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
import json
from groq import Groq
import pyttsx3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from deep_translator import GoogleTranslator
import fitz  # PyMuPDF
from fpdf import FPDF
import base64
from dotenv import load_dotenv
from email.message import EmailMessage

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load .env variables
load_dotenv()

# Configuration LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "LanguePro AI"

SENDER_EMAIL = os.environ.get("EMAIL_USER")
SENDER_PASSWORD = os.environ.get("EMAIL_PASS")

# Configuration Groq
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

# LangSmith tracing
def encode_image_to_base64(image_file):
    image = Image.open(image_file).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@traceable(name="TranslationChain")
def translate_text(prompt):
    return llm.invoke([HumanMessage(content=prompt)]).content

from langsmith.run_helpers import traceable

@traceable(name="TextToAudio")
def generate_audio(text, lang):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    voice_map = {"fr": voices[0], "en": voices[1] if len(voices) > 1 else voices[0]}
    engine.setProperty('voice', voice_map.get(lang, voices[0]).id)
    engine.save_to_file(text, "output_audio.mp3")
    engine.runAndWait()


client = Groq()
# Fonction de génération de description d’image
@traceable(name="ImageCaptioning")
def generate_caption(uploaded_file):
    # Lire l'image directement depuis l'objet UploadedFile
    img_bytes = uploaded_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    messages = [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
    ]

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # ou autre modèle
        messages=[{"role": "user", "content": messages}]
    )

    return response.choices[0].message.content

    
@traceable(name="PDFTranslation")
def translate_pdf_text(text, source_lang, target_lang):
    max_len = 4500  # Deep Translator limite ~5000, on prend un peu moins pour sécurité
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    translated = ""
    for chunk in chunks:
        translated += GoogleTranslator(source=source_lang, target=target_lang).translate(chunk) + "\n"
    return translated



@traceable(name="PDFToAudio")
def read_pdf_to_audio(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.save_to_file(text, "pdf_audio.mp3")
    engine.runAndWait()


@traceable(name="SendContactEmail")
def send_email(sender_email, sender_password, recipient_email, subject, body):
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        return True
    except Exception as e:
        print(f"Erreur : {e}")
        return False





# UI Streamlit
st.set_page_config(page_title="LanguePro AI", layout="wide")

def toast():
    st.markdown("""
    <style>
    #custom-toast {
    position: fixed;
    top: 60px;
    center: 20px;
    background-color: #28a745;
    color: white;
    padding: 14px 22px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-size: 15px;
    font-weight: 600;
    z-index: 9999;
    animation: fadeOut 4s forwards;
    }

    @keyframes fadeOut {
    0% {opacity: 1;}
    80% {opacity: 1;}
    100% {opacity: 0;}
    }
    </style>
    <div id="custom-toast">📡 LangSmith Monitoring Activé</div>
    """, unsafe_allow_html=True)

toast()

st.sidebar.markdown("""
<style>
@keyframes pulse {
  0% {opacity: 0.4;}
  50% {opacity: 1;}
  100% {opacity: 0.4;}
}
.sidebar-center {
  text-align: center;
}
#monitoring-indicator {
  animation: pulse 2s infinite;
  background-color: #28a745;
  color: white;
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: bold;
  width: fit-content;
  margin: 0 auto 10px auto;
  box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
</style>
<div class="sidebar-center">
  <div id="monitoring-indicator">🔎 Monitoring actif</div>
</div>
""", unsafe_allow_html=True)



# Style global
st.markdown("""
    <style>
        .block-container {
            max-width: 90% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .box {
            background-color: #B0E0E6;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        }
        h3 {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    selection = option_menu(
        menu_title="Menu",
        options=["Dashboard", "Traduction", "Text-to-Audio", "Image-to-Text", "Traduction PDF","Chatbot", "PDF to Audio", "À propos", "Contact Us"],
        icons=["check", "translate", "volume-up", "image", "file-earmark-text", "file-music", "info-circle", "envelope"],
        menu_icon="globe",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "blue", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e0e0e0"
            },
            "nav-link-selected": {
                "background-color": "#3399ff",
                "color": "white",
                "font-weight": "bold"
            }
        }
    )

toast()
# Pages
if selection == "Dashboard":
    toast()
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: #339CFF;'>🌎 LanguePro AI 🗣️</h1></div>""", unsafe_allow_html=True)
    def load_lottie_file(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)
    lottie_animation = load_lottie_file("animation.json")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""<div style="display: flex; justify-content: center;">""", unsafe_allow_html=True)
        st_lottie(lottie_animation, speed=1, width=380, height=380, key="lottie1")
        st.markdown("</div>", unsafe_allow_html=True)
    with col1:
        st.markdown("""<div class="box"> <h4>Traduction</h4></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="box"> <h4>Text-Audio</h4></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="box"> <h4>Image-Text</h4></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="box"> <h4>PDF-Audio</h4></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="box"> <h4>PDF-Traduction</h4></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="box"> <h4>À propos</h4></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="box"> <h4>Bienvenue dans notre application LanguePro AI ! Sélectionnez une fonctionnalité dans le menu à gauche pour commencer.</h4></div>""", unsafe_allow_html=True)

elif selection == "Traduction":
    toast()
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: #339CFF;'>🌍 LanguePro AI / Traduction</h1></div>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("Langue source", ["Français", "Anglais", "Espagnol", "Allemand", "Arabe", "Chinois", "Portugais"])
        c1,c2,c3,c4= st.columns(4)
        with c1:
            st.image("fran.jpg", width=50)
        with c2:
            st.image("ang.jpg", width=50)
        with c3:
            st.image("ar.jpg", width=50)  
        with c4:
            st.image("esp.jpg", width=50)
    
    with col2:
        target_lang = st.selectbox("Langue de sortie", ["Français", "Anglais", "Espagnol", "Allemand", "Arabe", "Chinois", "Portugais"])
        c1,c2,c3,c4= st.columns(4)
        with c1:
            st.image("all.jpg", width=50)
        with c2:
            st.image("por.jpg", width=50)
        with c3:
            st.image("ita.jpg", width=50)
        with c4:
            st.image("ch.jpg", width=50) 
    text = st.text_area("Entrez le texte à traduire")
    if st.button("Traduire"):
        prompt = f"Traduis du {source_lang} vers le {target_lang} : {text}"
        result = translate_text(prompt)
        st.success("Traduction réussie")
        st.text_area("Résultat", value=result)

elif selection == "Text-to-Audio":
    toast()
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: #339CFF;'>🔊 LanguePro AI/ Texte vers Audio</h1></div>""", unsafe_allow_html=True)

    source_lang_audio = st.selectbox("Langue source", ["Français", "Anglais", "Espagnol", "Allemand", "Arabe", "Chinois", "Portugais"])
    text_audio = st.text_area("Entrez le texte à convertir en audio")
    if st.button("Générer Audio"):
        generate_audio(text_audio, source_lang_audio)
        st.audio("output_audio.mp3")
    
# --- Image to Text ---
elif selection == "Image-to-Text":
    toast()
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: #339CFF;'>🖼️ LanguePro AI/ Image vers Texte</h1></div>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Choisissez une image", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2 style='text-decoration: underline; color: sky-blue;'>🔍 Aperçu de l'image </h2>", unsafe_allow_html=True)
        if uploaded_file:
            st.image(uploaded_file, caption="🖼️ Image chargée", use_container_width=True)
        else:
            st.warning("Veuillez charger une image")
    with col2:

        st.markdown("<h2 style='text-decoration: underline; color:sky-blue;'>📝 Description de l'image </h2>", unsafe_allow_html=True)

        if uploaded_file:
            

            if st.button("Générer la description"):
                with st.spinner("⏳ Génération en cours..."):
                    try:
                        description = generate_caption(uploaded_file)
                        st.session_state["description"] = description  # Sauvegarde dans session_state
                    except Exception as e:
                        st.error(f"❌ Erreur : {str(e)}")
            # Affiche la description si elle existe
            if "description" in st.session_state:
                st.success("✅ Description en anglais :")
                st.markdown(f"**{st.session_state['description']}**")
                st.markdown("-------------------")
                # Sélecteur de langue
                lang_options = ["Français", "Anglais", "Espagnol", "Allemand", "Arabe", "Chinois", "Portugais"]
                selected_lang = st.selectbox("Choisissez la langue de traduction", lang_options)

                if selected_lang:
                    prompt = f"Traduis ce texte en {selected_lang} : {st.session_state['description']}"
                    translated_text = translate_text(prompt)
                    st.success(f"📌 Traduction en {selected_lang} :")
                    st.markdown(f"**{translated_text}**")
        else:
            st.warning("📤 Veuillez charger une image.")

# --- Traduction PDF ---
elif selection == "Traduction PDF":
    toast()
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: #339CFF;'>📄 LanguePro AI / Traduction PDF</h1></div>""", unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("📤 Téléversez un fichier PDF de moins de 5000 caractères", type=["pdf"])

    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox("Langue source", ["fr", "en", "es","zh-CN","ar", "de", "it", "pt"])
    with col2:
        target_lang = st.selectbox("Langue cible", ["fr", "en", "es", "de", "it", "pt","zh-CN","ar"])

    if uploaded_pdf:
        st.success("Fichier PDF chargé avec succès !")
        if st.button("Traduire le PDF"):
            try:
                # Lecture du contenu PDF
                pdf_bytes = uploaded_pdf.read()
                with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                    full_text = ""
                    for page in doc:
                        full_text += page.get_text()

                # Vérification de la longueur pour éviter l'erreur > 5000 caractères
                if len(full_text) > 5000:
                    st.error("🚫 Le contenu du PDF dépasse la limite de 5000 caractères.")
                    st.stop()

                # Traduction
                translated_text = translate_pdf_text(full_text, source_lang, target_lang)

                # Affichage
                st.markdown("### 📝 Texte traduit :")
                st.text_area("Texte traduit", value=translated_text, height=300)

                # Génération d’un PDF avec police Unicode
# Génération d’un PDF avec police Unicode
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)  # Assure-toi que ce fichier est dans le même dossier
                pdf.set_font('DejaVu', '', 12)

                for line in translated_text.split('\n'):
                    pdf.multi_cell(0, 10, line)

                # ✅ Exporter en tant que chaîne (format bytes)
                pdf_bytes = pdf.output(dest='S').encode('latin1')  # Convertir en bytes
                pdf_output = io.BytesIO(pdf_bytes)  # Stocker dans un buffer

                # ✅ Bouton de téléchargement
                st.download_button(
                    label="📥 Télécharger la traduction PDF",
                    data=pdf_output,
                    file_name="pdf_traduit.pdf",
                    mime="application/pdf"
                )


            except Exception as e:
                st.error(f"Erreur lors de l'ouverture ou la traduction du PDF : {e}")
    else:
        st.warning("Veuillez téléverser un fichier PDF pour commencer la traduction.")


elif selection == "Chatbot":
    toast()
    st.title("🤖 Chatbot IA - Propulsé par Groq + LLaMA 3")
    # Initialiser l'historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="Tu es un assistant IA utile, amical et précis.")
        ]
    # Afficher l'historique dans l'interface
    for msg in st.session_state.messages[1:]:  # ignorer le message système pour l'affichage
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)
    # Zone de saisie utilisateur
    if prompt := st.chat_input("Posez votre question ici..."):
        # Ajouter le message utilisateur
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)
        # Générer la réponse avec le modèle Groq
        with st.chat_message("assistant"):
            try:
                response = llm(st.session_state.messages)
                st.markdown(response.content)
                st.session_state.messages.append(response)
            except Exception as e:
                st.error(f"Erreur lors de l'appel au chatbot : {e}")


elif selection == "PDF to Audio":
    toast()
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: #339CFF;'>🔊 LanguePro AI/ PDF vers Audio</h1></div>""", unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("📤 Téléversez un fichier PDF à lire à haute voix", type=["pdf"], key="audio_pdf")

    if uploaded_pdf is not None:
        with fitz.open(stream=uploaded_pdf.read(), filetype="pdf") as doc:
            full_text = "".join([page.get_text() for page in doc])

        st.subheader("📑 Aperçu du texte extrait")
        st.text_area("Texte extrait du PDF", full_text, height=200)

        if st.button("🔊 Lire le PDF"):
            try:
                read_pdf_to_audio(full_text)

                st.success("✅ Audio généré avec succès !")
                st.audio("pdf_audio.mp3")

                with open("pdf_audio.mp3", "rb") as f:
                    st.download_button(
                        label="📥 Télécharger le fichier audio",
                        data=f,
                        file_name="pdf_audio.mp3",
                        mime="audio/mpeg"
                    )
            except Exception as e:
                st.error(f"❌ Erreur lors de la lecture du PDF : {e}")


# --- À propos ---
elif selection == "À propos":
    toast()
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: #339CFF;'>💡 À propos de LanguePro AI</h1></div>""", unsafe_allow_html=True)    

    c1,c2,c3,c4,c5,c6,c7,c8= st.columns(8)
    with c1:
        st.image("fran.jpg", width=75)
    with c2:
        st.image("ang.jpg", width=75)
    with c3:
        st.image("ar.jpg", width=75)  
    with c4:    
        st.image("esp.jpg", width=75)
    with c5:
        st.image("all.jpg", width=75)  
    with c6:    
        st.image("ita.jpg", width=75)
    with c7:
        st.image("por.jpg", width=75)
    with c8:
        st.image("ch.jpg", width=75)
    st.markdown("""
    <div class="box">
    <h3>🎯Objectif</h3>
    <p>Notre objectif est de rendre la technologie linguistique accessible à tous, en fournissant des outils de traduction, de synthèse vocale et d'analyse d'images.</p>
    <p>Nous croyons que la communication sans frontières est essentielle dans le monde d'aujourd'hui.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #B0E0E6; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">
            🌍 Benefice
        </h3>
        <p style="color: #34495e; font-size: 16px;">
            LanguePro AI démocratise l'apprentissage des langues grâce à l'intelligence artificielle. 
            Notre objectif est de briser les barrières linguistiques en proposant des traductions 
            précises et instantanées dans plus de 6 langues, accessibles à tous.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #B0E0E6; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">🤖 Technologies Avancées</h3>
        <p style="color: #34495e; font-size: 16px;">
            LanguePro AI utilise les meilleurs modèles de NLP (comme MarianMT, T5 ou M2M100 de Hugging Face) 
            pour offrir des traductions fiables. La synthèse vocale s’appuie sur les API Text-to-Speech (TTS) 
            les plus récentes pour restituer une voix naturelle.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #B0E0E6; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">🧰 Fonctionnalités Principales</h3>
        <ul style="color: #34495e; font-size: 16px;">
            <li>📘 Traduction instantanée texte (Français ↔ Espagnol ↔ Anglais ↔ Allemand ↔ Portuguais ↔ Italien)</li>
            <li>🔊 Synthèse vocale du texte traduit (voix naturelle)</li>
            <li>📄 Téléversement de fichiers texte ou PDF à traduire</li>
            <li>🗂️ Interface claire et professionnelle</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #B0E0E6; border-radius: 15px; padding: 20px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">🧰 Outils Utilisés</h3>
        <ul style="color: #34495e; font-size: 16px;">
            <li><strong>🤖 Hugging Face  :</strong> Pour les modèles de traduction Helsinki</li>
            <li><strong>🌐 LangChain :</strong> Pour l'intégration des modèles de langage et la gestion des flux de travail.</li>
            <li><strong>📚 LangSmith :</strong> Pour le monitoring et la traçabilité des actions.</li>
            <li><strong>🌐 Groq :</strong> Pour l'inférence rapide des modèles de langage.</li>
            <li><strong>📜 PyMuPDF :</strong> Pour la lecture et l'extraction de texte des fichiers PDF.</li>
            <li><strong>📜 FPDF :</strong> llama-4-scout-17b-16e-instruct.</li>
            <li><strong>🔊 gTTS / pyttsx3 / TTS :</strong> Pour la synthèse vocale multilingue.</li>
            <li><strong>📄 PDF / Fichiers texte :</strong> Téléversement et lecture de fichiers via <code>st.file_uploader</code>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #B0E0E6; border-radius: 15px; padding: 20px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">🔒 Sécurité et traçabilité</h3>
        <p style="color: #34495e; font-size: 16px;">Toutes les actions sont monitorées à l'aide de LangSmith pour : </p>
        <ul style="color: #34495e; font-size: 16px;">
            <li><strong>Améliorer la qualité de l'expérience</li>
            <li><strong>Détecter les erreurs</li>
            <li><strong>Analyser l’usage (anonymement).</li>
        </ul>
        <p style="color: #34495e; font-size: 16px;">Aucune information personnelle n’est collectée sans votre consentement.</p>
    </div>
    """, unsafe_allow_html=True)

    

elif selection == "Contact Us":
    toast()
    st.markdown("""<div style='text-align: center;'><h1 style='text-decoration: underline; color: #339CFF;'>📧 Contactez-nous</h1></div>""", unsafe_allow_html=True)    
    st.markdown("""
    <div style="background-color: #B0E0E6; border-radius: 15px; padding: 20px; 
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); margin: 20px 0;">
        <h3 style="color:#2c3e50; font-weight: bold;">📧 Contacter le concepteur</h3>

    </div>
    """, unsafe_allow_html=True)
    st.write("Si vous avez des questions ou suggestions, envoyez-nous un e-mail:")
    
    # Formulaire de contact
    email = st.text_input("Votre adresse e-mail")
    subject = st.text_input("Sujet")
    message = st.text_area("Message")
    if st.button("Envoyer"):
        if email and subject and message:
            success = send_email(SENDER_EMAIL, SENDER_PASSWORD, SENDER_EMAIL, f"De {email} : {subject}", message)
            if success:
                st.success("Message envoyé avec succès !")
            else:
                st.error("Échec de l'envoi.")
        else:
            st.warning("Tous les champs sont requis.")

    st.info("""Par Hiroshi Yewo / IA""")