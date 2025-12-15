# Import komponen dalam telegram 
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update 

# komponen untuk membuat aplikasi bot dan handler
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler

# untuk membaca file CSV
import pandas as pd

# library untuk operasi matematika 
import numpy as np

# mengubah teks jadi vektor 
from sentence_transformers import SentenceTransformer

# database  vektor penyimpanan untuk mencari dokumen
import chromadb

# untuk async code di jupyter/colab
import nest_asyncio

#
from typing import Tuple, List, Dict

# Mencatat aktivitas bot
import logging

# data ke file CSV
import csv

# Mendapatkan waktu sekarang
from datetime import datetime

# Enable nested event loops if use colab
nest_asyncio.apply()

# Setup logging untuk tracking aktivitas bot
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__) # dipakai untuk mencetak log ke console

# Membuat header teks di console dan info bahwa bot sudah jalan
print("="*60)
print(" TELEGRAM BOT - IT SUPPORT SFL (with Evaluation)")
print("="*60)

# GLOBAL VARIABLES FOR EVALUATION
# untuk menyimpan semua data evaluasi
# digunakan untuk menghitung Precision dan MRR
evaluation_data = {
    'queries': [], # pertanyaan user
    'retrieved_docs': [], # daftar ID dokumen yang berhasil diambil bot (ranking) fungsinya Hitung MRR & precision
    'relevant_docs': [], # doc ID yang dianggap benar (ground truth) fungsinya membandingkan hasil retriev
    'feedback': [] # feedback yang membantu dan tidak membantu
}

# LOAD DATA & MODEL
print("\n Loading dataset...")

# Membaca file CSV dataset
try:
    df = pd.read_csv('datachatbot.csv', encoding='Windows-1252')
    print(f" Dataset loaded: {len(df)} rows")
except FileNotFoundError:
# Jika file tidak ditemukan, minta user upload (untuk Colab)
    print(" File datachatbot.csv tidak ditemukan!")
# jika pakai colab
    from google.colab import files
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df = pd.read_csv(filename, encoding='Windows-1252')
    print(f" Dataset loaded: {len(df)} rows")

# Load model
# ini akan mengubah teks menjadi vektor (embedding)
print("\n Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(" Model loaded!")

# Initialize ChromaDB
# database vektor yang menyimpan embedding dokumen
print("\n Initializing ChromaDB...")
client = chromadb.Client()
collection = client.create_collection(name="it_support_sfl_bot")

# Embed and store semua ke chroma db
print(f" Embedding {len(df)} documents...")
issues = df['issue'].tolist() # Mengubah kolom pertanyaan menjadi list
solutions = df['solve'].tolist() # Mengubah kolom solusi menjadi list
ids = [str(int(i)) for i in df['id'].tolist()] # Mengambil ID dokumen untuk kebutuhan

# Mengubah semua issue menjadi vektor (embedding)
embeddings = model.encode(issues, show_progress_bar=True)

# Menyimpan embedding ke ChromaDB dalam batch
print(" Storing in ChromaDB...")
batch_size = 100 # menambahkan data per 100 row biar tidak berat
for i in range(0, len(df), batch_size): # Loop untuk memasukkan data ke DB sedikit-sedikit
    end = min(i + batch_size, len(df)) 
    collection.add(
        embeddings=embeddings[i:end].tolist(), # Vektor embedding
        documents=issues[i:end], # isi pertanyaan user issue 
        metadatas=[{"solution": s} for s in solutions[i:end]], # menyimpan jawaban yang benar
        ids=ids[i:end] #menyimpan nomor id  Dokumen
    )

# memberikan info di console klo dataset udh ke load
print(f" {len(df)} documents stored in ChromaDB!")

# GUARDRAIL FILLTERING INPUT
KATA_KASAR = [
    'anjing', 'babi', 'bangsat', 'kampret', 'tolol', 
    'bodoh', 'goblok', 'asu','fuck', 'shit', 'bitch', 'idiot', 'stupid', 'bajingan', 'sialan'
]

def guardrail_check(text: str) -> Tuple[bool, str]:
    """Check for offensive words"""
    text_lower = text.lower()
    for kata in KATA_KASAR:
        if kata in text_lower:
            return False, " Mohon gunakan bahasa yang sopan. Pertanyaan Anda mengandung kata yang tidak pantas."
    return True, ""

# RETRIEVAL FUNCTION (dengan Top-K)

def semantic_search(query: str, k: int = 5):
    """Semantic search with top-k results"""
# Encode query menjadi vektor
    query_emb = model.encode([query])
# cari dokumen di chroma db yang paling mirip
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=k #ambil dokumen teratas
    )
    
    if results['ids'] and len(results['ids'][0]) > 0: # Cek apakah ada dokumen yang berhasil ditemukan.
        retrieved = []
        for i in range(len(results['ids'][0])): 
            retrieved.append({
                'id': results['ids'][0][i],
                'issue': results['documents'][0][i],
                'solution': results['metadatas'][0][i]['solution'],
                'distance': results['distances'][0][i], # (semakin kecil = semakin mirip)
                'rank': i + 1 # Rank ke berapa dia (peringkat)
            })
        return retrieved # Kembalikan hasil jika ada.
    return [] # Jika tidak ada hasil, kembalikan list kosong biar bot tidak error.

# EVALUATION METRICS
def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Calculate Precision@k"""
    if not retrieved_ids or not relevant_ids:
        return 0.0

# ambil k dokumen teratas
    retrieved_k = retrieved_ids[:k]
# menghitung  berapa banyak dokumen yang relevan
    relevant_retrieved = len(set(retrieved_k) & set(relevant_ids))
# mehitung precision
    precision = relevant_retrieved / k
    return precision

def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)"""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
# jika tidak ada dokumen relevan
    return 0.0

# Menghitung metrik evaluasi keseluruhan dari semua query
def calculate_overall_metrics() -> Dict:
    """Calculate overall Precision and MRR across all queries"""
    if not evaluation_data['queries']:
        return {'precision@1': 0, 'precision@3': 0, 'precision@5': 0, 'mrr': 0, 'total_queries': 0}
    
    precisions_1 = []
    precisions_3 = []
    precisions_5 = []
    mrrs = []
    
# Hitung metrik untuk setiap query 
    for retrieved, relevant in zip(evaluation_data['retrieved_docs'], evaluation_data['relevant_docs']):
        if retrieved and relevant:
            precisions_1.append(calculate_precision_at_k(retrieved, relevant, k=1))
            precisions_3.append(calculate_precision_at_k(retrieved, relevant, k=3))
            precisions_5.append(calculate_precision_at_k(retrieved, relevant, k=5))
            mrrs.append(calculate_mrr(retrieved, relevant))
# return rata rata dari semua query
    return {
        'precision@1': np.mean(precisions_1) if precisions_1 else 0,
        'precision@3': np.mean(precisions_3) if precisions_3 else 0,
        'precision@5': np.mean(precisions_5) if precisions_5 else 0,
        'mrr': np.mean(mrrs) if mrrs else 0,
        'total_queries': len(evaluation_data['queries'])
    }

# print summary di console contoh evaluation metric summary
def print_evaluation_summary():
    """Print evaluation summary to console"""
    metrics = calculate_overall_metrics()
    
    print("\n" + "="*60)
    print(" EVALUATION METRICS SUMMARY")
    print("="*60)
    print(f"Total Queries: {metrics['total_queries']}")
    print(f"Precision@1:   {metrics['precision@1']:.4f} ({metrics['precision@1']*100:.2f}%)")
    print(f"Precision@3:   {metrics['precision@3']:.4f} ({metrics['precision@3']*100:.2f}%)")
    print(f"Precision@5:   {metrics['precision@5']:.4f} ({metrics['precision@5']*100:.2f}%)")
    print(f"MRR:           {metrics['mrr']:.4f}")
    print("="*60 + "\n")

# TELEGRAM BOT HANDLERS

# digunakan ketika user ketik /start di telegram
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_msg = """
🤖 *Selamat datang di IT Support Chatbot SFL!*

Saya siap membantu Anda dengan masalah IT.

*Cara menggunakan:*
Langsung ketik pertanyaan Anda, contoh:
• Laptop tidak bisa connect wifi
• Cara reset password email
• Printer tidak bisa print

*Perhatian:*
Harap gunakan bahasa yang sopan dan profesional.

*Tips:*
Jelaskan masalah Anda dengan spesifik untuk hasil terbaik!

Ketik /help untuk melihat bantuan.
    """
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_msg = """
*Bantuan IT Support Chatbot*

*Perintah yang tersedia:*
/start - Memulai bot
/help - Menampilkan bantuan ini
/about - Informasi tentang bot

*Cara bertanya:*
Langsung ketik pertanyaan IT Anda tanpa perlu perintah khusus.

*Contoh pertanyaan:*
• "Laptop saya tidak bisa nyala"
• "Bagaimana cara reset password domain?"
• "Printer tidak terdeteksi"

Bot akan mencari solusi yang paling relevan dari database IT Support.
    """
    await update.message.reply_text(help_msg, parse_mode='Markdown')

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command"""
    about_msg = """
*Tentang IT Support Chatbot SFL*

*Perusahaan:* SFL
*Teknologi:*
• Sentence Transformer (all-MiniLM-L6-v2)
• ChromaDB Vector Store
• Semantic Retrieval

*Database:* 2100+ solusi IT
*Keamanan:* Dilengkapi guardrail filter

Dikembangkan untuk membantu karyawan menyelesaikan masalah IT dengan cepat dan efisien.
    """
    await update.message.reply_text(about_msg, parse_mode='Markdown')

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command - show evaluation metrics"""
    metrics = calculate_overall_metrics()
    
   
    await update.message.reply_text(stats_msg, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages"""
    user_message = update.message.text
    user_name = update.effective_user.first_name
    user_id = update.effective_user.id

    logger.info(f"User {user_name}: {user_message}")
    
    # Guardrail check
    is_safe, warning_msg = guardrail_check(user_message)
    if not is_safe:
        await update.message.reply_text(warning_msg)
        logger.warning(f"Blocked message from {user_name}: {user_message}")
        return
    
    # Show typing indicator
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    
    # Semantic search (top-5)
    results = semantic_search(user_message, k=5)
    
    if results:
        top_result = results[0]
        confidence = (1 - top_result['distance']) * 100

        # Filter pertanyaan tidak relevan
        if confidence < 0:
            await update.message.reply_text(
                "Mohon maaf, pertanyaan Anda kurang relevan dengan masalah IT Support.\n\n"
                "Saya hanya bisa membantu masalah IT seperti:\n"
                "• Masalah laptop/komputer\n"
                "• Printer tidak berfungsi\n"
                "• Koneksi internet/wifi\n"
                "• Password dan akun\n"
                "• Software/aplikasi\n\n"
                "Silakan tanyakan masalah IT Anda."
            )
            logger.info(f"Irrelevant query from {user_name}: confidence={confidence:.1f}%")
            return

        # Store for evaluation
        retrieved_ids = [r['id'] for r in results]
        evaluation_data['queries'].append(user_message)
        evaluation_data['retrieved_docs'].append(retrieved_ids)
        # Initially, we don't know the relevant doc, will be updated by feedback
        evaluation_data['relevant_docs'].append([top_result['id']])
        evaluation_data['feedback'].append(None)

        # Calculate current query metrics
        current_precision_1 = calculate_precision_at_k(retrieved_ids, [top_result['id']], k=1)
        current_precision_2 = calculate_precision_at_k(retrieved_ids, [top_result['id']], k=2)
        current_precision_3 = calculate_precision_at_k(retrieved_ids, [top_result['id']], k=3)

        # Print to console
        print("\n" + "="*60)
        print(f"QUERY: {user_message}")
        print(f"USER: {user_name} (ID: {user_id})")
        print("-"*60)
        print(f"Top Result ID: {top_result['id']}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Distance: {top_result['distance']:.4f}")
        print("-"*60)
        print(f"CURRENT QUERY METRICS:")
        print(f"   Precision@1: {current_precision_1:.4f} ({current_precision_1*100:.2f}%)")
        print(f"   Precision@2: {current_precision_2:.4f} ({current_precision_2*100:.2f}%)")
        print(f"   Precision@3: {current_precision_3:.4f} ({current_precision_3*100:.2f}%)")
        print("-"*60)
        print(f"Top-5 Retrieved IDs: {retrieved_ids}")
        print("="*60)
        
        # Print overall metrics
        print_evaluation_summary()

        response = f"""
*Solusi untuk masalah Anda:*

*Masalah:*
{top_result['issue']}

*Solusi:*
{top_result['solution']}

Confidence: {confidence:.1f}%

---
Apakah solusi ini membantu? Jika tidak, coba jelaskan masalah dengan lebih detail.
        """

        keyboard = [
            [
                InlineKeyboardButton("✅ Membantu", callback_data=f"helpful|{top_result['id']}"),
                InlineKeyboardButton("❌ Tidak membantu", callback_data=f"not_helpful|{top_result['id']}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
    
        await update.message.reply_text(response, parse_mode='Markdown', reply_markup=reply_markup)
        logger.info(f"Response sent to {user_name} (ID: {top_result['id']}, Confidence: {confidence:.1f}%)")

    else:
        await update.message.reply_text(
            "Maaf, saya tidak menemukan solusi yang tepat untuk masalah Anda.\n\n"
            "Coba:\n"
            "• Jelaskan masalah dengan lebih spesifik\n"
            "• Gunakan kata kunci yang berbeda\n"
            "• Hubungi IT Support langsung jika mendesak"
        )
        logger.info(f"No solution found for {user_name}")

async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle feedback from users"""
    query = update.callback_query
    await query.answer()

    feedback, doc_id = query.data.split("|")
    user_id = update.effective_user.id
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Simpan evaluasi ke CSV
    with open("evaluation_logs.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_id, doc_id, feedback])

    # Update evaluation data
    if evaluation_data['feedback']:
        evaluation_data['feedback'][-1] = feedback
        
        # Update relevant docs based on feedback
        if feedback == "helpful":
            # The retrieved doc was relevant
            evaluation_data['relevant_docs'][-1] = [doc_id]
        else:
            # The retrieved doc was NOT relevant (relevant doc is unknown)
            evaluation_data['relevant_docs'][-1] = []

    print("\n" + "="*60)
    print(f" FEEDBACK RECEIVED")
    print(f"Doc ID: {doc_id}")
    print(f"Feedback: {feedback}")
    print(f"User ID: {user_id}")
    print("="*60)
    
    # Print updated overall metrics
    print_evaluation_summary()

    if feedback == "helpful":
        msg = "Terima kasih! Senang bisa membantu! 😊"
    else:
        msg = "Terima kasih atas feedbacknya! Saya akan coba lebih baik lagi! 🙏"

    await query.edit_message_reply_markup(reply_markup=None)
    await query.message.reply_text(msg)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")

# ========================================
# MAIN FUNCTION
# ========================================
def main():
    """Run the bot"""
    TOKEN = "8025034942:AAHBy8UOc-AeSS7YcWQ5PWiNGqSelKTBR_U"
    
    if TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("\n" + "="*60)
        print("ERROR: Token bot belum diisi!")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("STARTING TELEGRAM BOT...")
    print("="*60)
    
    # Create application
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    # application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(feedback_handler))
    application.add_error_handler(error_handler)

    print("\nBot is running!")
    print("Buka Telegram dan cari bot Anda")
    print("Tekan Ctrl+C untuk stop bot")
    print("Ketik /stats di bot untuk melihat evaluasi")
    print("="*60 + "\n")
    
    # Run bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

# RUN BOT

if __name__ == "__main__":
    main()