import streamlit as st
import io, re, zipfile
import fitz  # PyMuPDF
import docx  # python-docx
import pptx  # python-pptx
import pandas as pd
import pytesseract
from PIL import Image
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from unidecode import unidecode
from openai import OpenAI
from fpdf import FPDF  # Pour générer le PDF

# --- CLIENT API NVIDIA ---
def get_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-..."
    )

# --- (toutes vos fonctions d'extraction et de traitement, inchangées) ---
# extract_text_from_pdf, extract_text_from_docx, ..., process_file, clean_text, split_text_into_chunks
# embed_chunks, build_faiss_index, search_best_chunks, generate_dynamic_prompt, generate_pdf, generate_docx

st.title('Résumé automatique de documents scientifiques')

# initialisation de session_state
if 'resume_text' not in st.session_state:
    st.session_state['resume_text'] = None

# Upload & options
ing = st.selectbox('Langue du résumé', ['Français','Anglais'])
lang = 'francais' if ing=='Français' else 'english'
preset = ['Contexte et objectifs','Verrous','Approche globale','Principaux résultats','Discussion','Perspectives']
chosen = st.multiselect('Sections à inclure', preset, default=['Contexte et objectifs','Principaux résultats','Perspectives'])
txt = st.text_input('Autres sections (séparées par virgules)','')
custom = [s.strip() for s in txt.split(',') if s.strip()]
sections = chosen + [s for s in custom if s not in chosen]

mode = st.radio('Mode', ['Fichier unique','Dossier (.zip)'])
files = st.file_uploader('Upload', type=['pdf','docx','pptx','xlsx','txt','jpg','jpeg','png','zip'],
                        accept_multiple_files=(mode.startswith('Dossier')))

# Bouton de génération
if st.button('Générer le résumé'):
    if not files:
        st.warning('Veuillez uploader au moins un fichier.')
        st.stop()

    uploaded = files if isinstance(files, list) else [files]
    txts = []
    for f in uploaded:
        if f is None:
            st.error('Fichier non valide détecté.')
            st.stop()
        f.seek(0)
        txts.append(process_file(f, f.name))

    full = "\n".join(txts)
    if not full.strip():
        st.error('Aucun texte extrait.')
        st.stop()

    clean = clean_text(full)
    chks = split_text_into_chunks(clean)
    with st.spinner('Embeddings et recherche...'):
        mdl = SentenceTransformer('all-MiniLM-L6-v2')
        emb = embed_chunks(chks, mdl)
        idx = build_faiss_index(np.array(emb))
        qemb = mdl.encode('objectif verrous méthodologie résultats perspectives', convert_to_numpy=True)
        ids = search_best_chunks(idx, qemb)
        ctx = "\n\n".join(chks[i] for i in ids if i < len(chks))
        prm = generate_dynamic_prompt(ctx, sections, lang)
        res = get_client().chat.completions.create(
            model='mistralai/mistral-small-24b-instruct',
            messages=[{'role':'user','content':prm}],
            temperature=0.2, top_p=0.7, max_tokens=2048, stream=True
        )
        out = ''
        for chunk in res:
            delta = chunk.choices[0].delta.content
            if delta:
                out += delta
    st.success('Résumé terminé.')
    st.session_state['resume_text'] = out  # On stocke le résumé

# Affichage et téléchargement uniquement si résumé généré
if st.session_state['resume_text']:
    out = st.session_state['resume_text']
    st.text_area('Résumé généré', out, height=300)

    # radio avec clé fixe : choix persistant
    download_format = st.radio(
        'Format de téléchargement',
        ['PDF', 'DOCX'],
        key='fmt_choice'
    )

    if download_format == 'PDF':
        st.download_button(
            'Télécharger en PDF',
            data=generate_pdf(out),
            file_name='resume.pdf',
            mime='application/pdf'
        )
    else:
        st.download_button(
            'Télécharger en DOCX',
            data=generate_docx(out),
            file_name='resume.docx',
            mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
