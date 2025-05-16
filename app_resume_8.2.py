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
        api_key="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"
    )

# --- EXTRACTION DE TEXTE ---
...  # (Garder toutes les fonctions d'extraction de texte existantes)

# --- PDF OUTPUT ---
def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 8, unidecode(line))
    return pdf.output(dest='S').encode('latin1')

# --- DOCX OUTPUT ---
def generate_docx(text):
    doc = docx.Document()
    for line in text.split('\n'):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()

# --- INTERFACE STREAMLIT ---
st.title('Résumé automatique de documents scientifiques')
st.markdown('Choisissez les sections (y compris personnalisées) et générez un résumé structuré.')

# Langue
ing = st.selectbox('Langue du résumé', ['Français','Anglais'])
lang = 'francais' if ing=='Français' else 'english'

# Sections prédéfinies
preset = ['Contexte et objectifs','Verrous','Approche globale','Principaux résultats','Discussion','Perspectives']
chosen = st.multiselect('Sections à inclure', preset, default=['Contexte et objectifs','Principaux résultats','Perspectives'])

# Sections perso
txt = st.text_input('Autres sections (séparées par virgules)','')
custom = [s.strip() for s in txt.split(',') if s.strip()]
sections = chosen + [s for s in custom if s not in chosen]

# Upload
mode = st.radio('Mode', ['Fichier unique','Dossier (.zip)'])
files = st.file_uploader('Upload', type=['pdf','docx','pptx','xlsx','txt','jpg','jpeg','png','zip'], accept_multiple_files=(mode.startswith('Dossier')))

if st.button('Générer le résumé'):
    if not files:
        st.warning('Veuillez uploader au moins un fichier.')
    else:
        # Extraction et traitement
        txts = [process_file(f, f.name) for f in (files if isinstance(files,list) else [files])]
        full = '\n'.join(txts)
        if not full.strip(): st.error('Aucun texte extrait.'); st.stop()
        clean = clean_text(full)
        st.text_area('Texte nettoyé', clean[:500]+'...', height=150)
        chks = split_text_into_chunks(clean)
        st.write(f'{len(chks)} chunks générés.')

        # Embeddings
        with st.spinner('Embeddings...'):
            mdl = SentenceTransformer('all-MiniLM-L6-v2')
            emb = embed_chunks(chks, mdl)
        idx = build_faiss_index(np.array(emb))

        # Recherche de contexte
        qemb = mdl.encode('objectif verrous méthodologie résultats perspectives', convert_to_numpy=True)
        ids = search_best_chunks(idx, qemb)
        ctx = '\n\n'.join(chks[i] for i in ids if i < len(chks))
        prm = generate_dynamic_prompt(ctx, sections, lang)

        # Appel LLM et affichage
        st.markdown('**Appel API...**')
        with st.spinner('Génération...'):
            try:
                res = get_client().chat.completions.create(
                    model='mistralai/mistral-small-24b-instruct',
                    messages=[{'role':'user','content':prm}],
                    temperature=0.2, top_p=0.7, max_tokens=2048, stream=True
                )
                out = ''
                ph = st.empty()
                for c in res:
                    if c.choices[0].delta.content:
                        out += c.choices[0].delta.content
                        ph.write(out)
                st.success('Résumé terminé.')

                # Choix du format et téléchargement
                download_format = st.radio('Format de téléchargement', ['PDF','DOCX'], index=0)
                if download_format == 'PDF':
                    st.download_button(
                        'Télécharger PDF', data=generate_pdf(out),
                        file_name='resume.pdf', mime='application/pdf'
                    )
                else:
                    st.download_button(
                        'Télécharger DOCX', data=generate_docx(out),
                        file_name='resume.docx', mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                    )

            except Exception as e:
                st.error(f'Erreur LLM: {e}')
