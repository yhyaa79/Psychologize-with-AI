#!pip install -U 'g4f[all]'
#!pip install PyExecJS
#!pip install streamlit
#!pip install curl_cffi
#!pip install -U openai-whisper
#!pip install telebot
#!pip install pydub
#!sudo apt-get install ffmpeg
#!pip install bark



################# import Library

import datetime
import os
import certifi
import ssl
import g4f
from g4f.client import Client
import telebot
import time  
import whisper
import concurrent.futures
import re
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
import threading
from queue import Queue
from telebot import types
import json
import pandas as pd
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton



################ import paths

base_path = "/Users/yayhaeslami/Python/my_workspace/Psychology"

path_original = f"{base_path}/original.txt"  
path_summary = f"{base_path}/summary.txt"   
save_audio = f"{base_path}"  
path_psychologize = f"{base_path}/psychologize.txt"  
path_practice = f"{base_path}/practice.txt"  
path_chat_history = f"{base_path}/chat_history.json"
path_file_xlsx = 'psychology_sessions.xlsx'
#path_file_json = f'schedules/{user_id}_schedule.json'



################ make variables

text_storage = []
voice_storage = []
processing_queue = Queue()
lock = threading.Lock()
new_conversations = {}
callback_data_dict = {}




################ api telegram

API_TOKEN = "7419737791:AAElCmpsL3R3sJcuWCjjStiCoFZ-vq5wPug"
bot = telebot.TeleBot(API_TOKEN)





################ lode model voice to text and text to voice

#model = whisper.load_model("large")
#preload_models()





################ chatGPT 3.5 turbo

"""def chatGPT_35_turbo(input_text_chatGPT_35_turbo, text_chatGPT_35_turbo):
    os.environ['SSL_CERT_FILE'] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context
    text = f"{text_chatGPT_35_turbo}\n{input_text_chatGPT_35_turbo}"
    response = g4f.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": text}],
        stream=True
    )
    input_text_chatGPT_35_turbo = ""
    for message in response:
        input_text_chatGPT_35_turbo += message  

    return input_text_chatGPT_35_turbo
"""

def chatGPT_35_turbo(input_text_chatGPT_35_turbo, text_chatGPT_35_turbo):
    client = Client()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{text_chatGPT_35_turbo}\n{input_text_chatGPT_35_turbo}"}],
    )
    return response.choices[0].message.content







################ data today

def make_formatted_date():
    date = datetime.datetime.today()
    formatted_date = date.strftime("%Y/%m/%d")
    return formatted_date





################ voice to text

def voice_to_text(model, voice_for_save):
    result = model.transcribe(voice_for_save)
    print(result["text"])
    clear_ogg_files(base_path)
    return result["text"]




################# text to voice

def text_to_voice(text_prompt, save_audio):
    audio_array = generate_audio(text_prompt)
    
    #print(f"Saving audio to: {save_audio}")
    
    write_wav(save_audio, SAMPLE_RATE, audio_array)
    Audio(audio_array, rate=SAMPLE_RATE)







#################  read file

def read_file(file_path, date_texts):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    all_matches = []
    for date_text in date_texts:
        pattern = re.compile(rf'(\d_voice_{date_text}:"[^"]*")')
        matches = pattern.findall(content)
        all_matches.extend(matches)

    result = "\n".join([match.split(':', 1)[1].strip('"') for match in all_matches])

    return result







################# save file 

def save_file(date, teresultai_chatGPT_35_turbo, filename):
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
            next_number = 1
            for line in reversed(lines):
                line = line.strip()
                if line:
                    parts = line.split('_')
                    if parts and parts[0].isdigit():
                        last_number = int(parts[0])
                        next_number = last_number + 1
                        break
    except FileNotFoundError:
        next_number = 1

    separator = "..........................................................."
    with open(filename, "a") as file:
        file.write(f"{next_number}_{date}:\"{teresultai_chatGPT_35_turbo}\"\n{separator}\n")







################## save file original

def original(user_id, formatted_date, text_for_save):
    try:
        with open(path_original, "r") as file:
            lines = file.readlines()
            next_number = 1
            for line in reversed(lines):
                line = line.strip()
                if line:
                    parts = line.split('_')
                    if parts and parts[0].isdigit():
                        last_number = int(parts[0])
                        next_number = last_number + 1
                        break
    except FileNotFoundError:
        next_number = 1

    separator = "..........................................................."
    with open(path_original, "a") as file:
        file.write(f"{next_number}_{user_id}_{formatted_date}:\"{text_for_save}\"\n{separator}\n")

    return next_number





################## save file summary

def summary(summary_type ,date, text, path_summary):
    try:
        with open(path_summary, "r") as file:
            lines = file.readlines()
            next_number = 1
            for line in reversed(lines):
                line = line.strip()
                if line:
                    parts = line.split('_')
                    if parts and parts[0].isdigit():
                        last_number = int(parts[0])
                        next_number = last_number + 1
                        break
    except FileNotFoundError:
        next_number = 1
  
    separator = "..........................................................."
    with open(path_summary, "a") as file:
        file.write(f"{next_number}_{summary_type}_{date}:\"{text}\"\n{separator}\n")






################## extract information from file

def extract_information(next_number):

    with open(path_original, 'r') as file:
        lines = file.readlines()

    for index, line in enumerate(lines):
        if line.startswith(f"{next_number}_voice"):
            parts = line.split(':')
            if len(parts) < 2:
                continue

            header = parts[0]
            header_parts = header.split('_')
            if len(header_parts) < 3:
                continue

            summary_type = header_parts[1]
            date = header_parts[2]
            if len(header_parts) > 3 and header_parts[3] == 'summary':      
                break  

            input_text = parts[1].strip().strip('"')
            return summary_type, date, input_text

    return None, None, None





################## Delete ogg files

def clear_ogg_files(base_path):
    try:
        for filename in os.listdir(base_path):
            if filename.endswith('.ogg'):
                file_path = os.path.join(base_path, filename)
                
                if os.path.isfile(file_path):
                    os.remove(file_path)  
                         
    except Exception as e:
        print(f"An error clear_ogg_files: {e}")






################## add data to table

def add_and_save_to_table(df, user_id, formatted_date, identified_issues, conversation_summary, severity_level, 
                          treatment_recommendations, Practice_Table, file_path):
    new_data = {
        "User ID": user_id,
        "Datetime": formatted_date,
        "Identified Issues": identified_issues,
        "Conversation Summary": conversation_summary,
        "Severity Level": severity_level,
        "Treatment Recommendations": treatment_recommendations,
        "Practice Table": Practice_Table
    }
    
    new_row = pd.DataFrame([new_data])
    df = pd.concat([df, new_row], ignore_index=True)
    
    if file_path.endswith('.csv'):
        df.to_csv(file_path, index=False)

    elif file_path.endswith('.xlsx'):
        df.to_excel(file_path, index=False)

    elif file_path.endswith('.json'):
        df.to_json(file_path, orient='records', lines=True)






################## Save data to original file and summary file

def run_all_storage(text_for_save):
    try:
        formatted_date = make_formatted_date()
        next_number = original(formatted_date, text_for_save)
        summary_type, date, input_text = extract_information(next_number)

        if summary_type and date and input_text:
            text_chatGPT_35_turbo = "Read the following text and separate the important points of the text and return to English in the output."
            result_chatGPT_35_turbo_summary = chatGPT_35_turbo(input_text, text_chatGPT_35_turbo)
            summary(summary_type, date, result_chatGPT_35_turbo_summary, path_summary)  
        else:
            print("No information extracted.")

        clear_ogg_files(base_path)

    except Exception as e:
        print(f"An error run_all_storage: {e}")



#text_chatGPT_35_turbo = "Read the following text and separate the important points of the text and return to English in the output."






"""def run_all_storage_table(text_for_save, user_id):
    try:
        formatted_date = make_formatted_date()
        result_original_summary = original(user_id, formatted_date, text_for_save)


        identified_issues = chatGPT_35_turbo(text_for_save, "Please extract a brief and concise list of psychological, emotional, or behavioral problems from the text below without any additional explanation.")

        conversation_summary = chatGPT_35_turbo(text_for_save, "Please summarize the text below, including the key points, in a brief and concise manner.")

        severity_level = chatGPT_35_turbo(text_for_save, "Please tell me the level of the problem in terms of psychologist or analytical system in the following text without any additional explanation.")

        treatment_recommendations = chatGPT_35_turbo(text_for_save, "Please give me psychoanalytic offers based on the text below")

        Practice_Table = chatGPT_35_turbo(text_for_save, "Practice me with my conversation and my mental problems to improve my mental mood.")

        add_task_to_next_available_date(user_id, Practice_Table)


        df_excel = load_or_create_table(path_file_xlsx)
        add_and_save_to_table(df_excel, user_id, formatted_date, identified_issues, conversation_summary, severity_level, 
                              treatment_recommendations, Practice_Table, path_file_xlsx)

    except Exception as e:
        print(f"An error run_all_storage_table: {e}")
"""




################## Save data to original file and xlsx file and json file

def run_all_storage_table(text_for_save, user_id):
    try:
        formatted_date = make_formatted_date()
        result_original_summary = original(user_id, formatted_date, text_for_save)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(chatGPT_35_turbo, text_for_save, "Please extract a brief and concise list of psychological, emotional, or behavioral problems from the text below without any additional explanation."): 'identified_issues',
                executor.submit(chatGPT_35_turbo, text_for_save, "Please summarize the text below, including the key points, in a brief and concise manner."): 'conversation_summary',
                executor.submit(chatGPT_35_turbo, text_for_save, "Please tell me the level of the problem in terms of psychologist or analytical system in the following text without any additional explanation."): 'severity_level',
                executor.submit(chatGPT_35_turbo, text_for_save, "Please give me psychoanalytic offers based on the text below"): 'treatment_recommendations',
                executor.submit(chatGPT_35_turbo, text_for_save, "Practice me with my conversation and my mental problems to improve my mental mood."): 'Practice_Table',
            }

            results = {futures[future]: future.result() for future in concurrent.futures.as_completed(futures)}

        identified_issues = results['identified_issues']
        conversation_summary = results['conversation_summary']
        severity_level = results['severity_level']
        treatment_recommendations = results['treatment_recommendations']
        Practice_Table = results['Practice_Table']

        add_task_to_next_available_date(user_id, Practice_Table)


        df_excel = load_or_create_table(path_file_xlsx)
        add_and_save_to_table(df_excel, user_id, formatted_date, identified_issues, conversation_summary, severity_level, 
                              treatment_recommendations, Practice_Table, path_file_xlsx)

    except Exception as e:
        print(f"An error occurred in run_all_storage_table: {e}")




################## convert voice to text and run 'run_all_storage_table' 

def run_all_voice(voice_for_save, model, chat_id):
    try:
        text_for_save = voice_to_text(model, voice_for_save)
        if text_for_save:
            run_all_storage_table(text_for_save, chat_id)
        else:
            bot.send_message(chat_id, "Transcription resulted in empty text.")
        clear_ogg_files(base_path)

    except Exception as e:
        bot.send_message(chat_id, f"An error run_all_voice: {e}")




################## convert chat bot to text and run 'run_all_storage_table' 

def run_all_chat_bot(text_chat_bot, user_id):
    try:
        text_for_save = chatGPT_35_turbo(text_chat_bot, "The conversation is under my conversation with chat bot. Please summarize my conversation")  
        run_all_storage_table(text_for_save, user_id)

    except Exception as e:
        print(f"An error run_all_chat_bot: {e}")




################## psychologize 

def psychologize(text_for_save):
    try:

        input_text_chatGPT_35_turbo = chatGPT_35_turbo(text_for_save, "Read the following text and separate the important points of the text and return to English in the output.")

        text_chatGPT_35_turbo = "These are my words about my life, you can psychologize me and say what problems do I have in my life?Say in English."
        resultai_chatGPT_35_turbo_read_file = chatGPT_35_turbo(input_text_chatGPT_35_turbo, text_chatGPT_35_turbo)

        return resultai_chatGPT_35_turbo_read_file
    except Exception as e:
        print(f"An error psychologize: {e}")







"""def run_all_psychologize(path_summary, date_text, path_psychologize):
    try:

        input_text_chatGPT_35_turbo = read_file(path_summary, date_text)

        text_chatGPT_35_turbo = "These are my words about my life, you can psychologize me and say what problems do I have in my life?Say in English."
        resultai_chatGPT_35_turbo_read_file = chatGPT_35_turbo(input_text_chatGPT_35_turbo, text_chatGPT_35_turbo)


        save_file(date_text, resultai_chatGPT_35_turbo_read_file, path_psychologize)

    except Exception as e:
        print(f"An error run_all_psychologize: {e}")"""





################## give practice to psychologize

def practice(resultai_chatGPT_35_turbo_read_file):
    try:
        text_chatGPT_35_turbo_practice = "Can you give me a few ways and practice to improve my life according to my problems? Say in English. These are my problems:"
        resultai_chatGPT_35_turbo_practice = chatGPT_35_turbo(resultai_chatGPT_35_turbo_read_file, text_chatGPT_35_turbo_practice)
        
        return resultai_chatGPT_35_turbo_practice

    except Exception as e:
        return f"An error practice: {e}"





"""def run_all_practice(path_psychologize, date_text):
    try:

        input_text_chatGPT_35_turbo_practice = read_file(path_psychologize, date_text)

        text_chatGPT_35_turbo_practice = "Can you give me a few ways and practice to improve my life according to my problems? Say in English. These are my problems:"
        resultai_chatGPT_35_turbo_practice = chatGPT_35_turbo(input_text_chatGPT_35_turbo_practice, text_chatGPT_35_turbo_practice)
        print(resultai_chatGPT_35_turbo_practice)

        save_file(date_text, resultai_chatGPT_35_turbo_practice, path_practice)

    except Exception as e:
        print(f"An error run_all_practice: {e}")"""






################## Load chat history from file

def load_chat_history():
    try:
        with open(path_chat_history, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}



################## Save chat history to file

def save_chat_history():
    with open(path_chat_history, "w") as file:
        json.dump(chat_history, file)



chat_history = load_chat_history()






################## make xlsx file and load it

def load_or_create_table(file_path):
    if os.path.exists(file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, orient='records', lines=True)
    else:
        columns = ["User ID", "Datetime", "Identified Issues", "Conversation Summary", 
                   "Severity Level", "Treatment Recommendations", "Practice Table", "Additional Notes"]
        df = pd.DataFrame(columns=columns)
    return df




################## create table in 'xlsx' file

def filter_sessions_by_user_id(user_id):

    df = pd.read_excel(path_file_xlsx)
    filtered_df = df[df['User ID'] == user_id]
    return filtered_df




################## create 'datetime' and 'issues' in the table

def filter_datetime_issues_by_user_id(user_id):

    df = pd.read_excel(path_file_xlsx)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

    filtered_df = df[df['User ID'] == user_id]
    return filtered_df


def get_datetime_issues_by_user_id(user_id):
    filtered_df = filter_datetime_issues_by_user_id(user_id)
    Datetime_Identified_Issues = ""

    for index, row in filtered_df.iterrows():
        Datetime_Identified_Issues += f"Datetime:\n{row['Datetime'].strftime('%Y/%m/%d')}\n\nIssues:\n{row['Identified Issues']}\n...................................\n"

    return Datetime_Identified_Issues


def Datetime_Issues(user_id):
    try:
        datetime_issues_text = get_datetime_issues_by_user_id(user_id)
        if not datetime_issues_text:
            return "No issues found for the given user ID."

        result = chatGPT_35_turbo(datetime_issues_text, "Considering the information given to me, explain to me what the user has problems")
        return result

    except Exception as e:
        return f"An error occurred in Datetime_Issues: {e}"






################## create 'datetime' and 'Severity' in the table

import pandas as pd

def filter_datetime_Severity_by_user_id(user_id):

    df = pd.read_excel(path_file_xlsx)

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

    filtered_df = df[df['User ID'] == user_id]
    return filtered_df


def get_severity_datetime_by_user_id(user_id):
    filtered_df = filter_datetime_Severity_by_user_id(user_id)
    Severity_Datetime = ""

    for index, row in filtered_df.iterrows():
        Severity_Datetime += f"Datetime:\n{row['Datetime'].strftime('%Y/%m/%d')}\n\nSeverity:\n{row['Severity Level']}\n...................................\n"

    return Severity_Datetime


def Datetime_Severity(user_id):
    try:
        datetime_issues_text = get_severity_datetime_by_user_id(user_id)
        if not datetime_issues_text:
            return "No issues found for the given user ID."

        result = chatGPT_35_turbo(datetime_issues_text, "Explain the process of improving my mental mood on the basis of the following information and dates")
        return result

    except Exception as e:
        return f"An error occurred in Datetime_Severity: {e}"
    





################## create 'recommendations' in the table

def filter_treatment_recommendations_by_user_id(user_id):

    df = pd.read_excel(path_file_xlsx)

    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

    filtered_df = df[df['User ID'] == user_id]
    return filtered_df


def get_treatment_recommendations_by_user_id(user_id):
    filtered_df = filter_treatment_recommendations_by_user_id(user_id)
    Treatment_Recommendations = ""

    for index, row in filtered_df.iterrows():
        Treatment_Recommendations += f"{row['Treatment Recommendations']}\n\n"

    return Treatment_Recommendations


def treatment_recommendations(user_id):
    try:
        treatment_recommendations_text = get_treatment_recommendations_by_user_id(user_id)
        if not treatment_recommendations_text:
            return "No issues found for the given user ID."

        result = chatGPT_35_turbo(treatment_recommendations_text, "Depending on the information I gave, explain the suggestions and the way to treatment")
        return result

    except Exception as e:
        return f"An error occurred in treatment_recommendations: {e}"





################## convert text to masege

def send_long_message(chat_id, text):
    while len(text) > 4000:
        split_index = text.rfind('\n', 0, 4000)
        if split_index == -1:
            split_index = 4000  
        bot.send_message(chat_id, text[:split_index])
        text = text[split_index:].lstrip()  
    bot.send_message(chat_id, text)  





################## add text to 'json' file

def add_task_to_next_available_date(user_id, task):
    schedule = load_user_schedule(user_id)
    date = datetime.datetime.today().strftime("%Y/%m/%d")
    while date in schedule and len(schedule[date]) >= 1: 
        date = (datetime.datetime.strptime(date, "%Y/%m/%d") + datetime.timedelta(days=1)).strftime("%Y/%m/%d")
    if date not in schedule:
        schedule[date] = []
    schedule[date].append(task)
    save_user_schedule(user_id, schedule)

    return date  





#################################################### Function bot



"""def process_text_step(message):
    if message.text == 'Finish':
        combined_text = "\n".join(text_storage)
        text_storage.clear()  
        bot.send_message(message.chat.id, "Please wait...")  # Sending "Please wait..." message
        threading.Thread(target=process_text_in_background, args=(combined_text, message.chat.id)).start()
    else:
        text_storage.append(message.text)
        msg = bot.reply_to(message, "Please enter more text or press 'Finish':")
        bot.register_next_step_handler(msg, process_text_step)
"""



def process_text_in_background(combined_text, chat_id):
    with lock:
        run_all_storage_table(combined_text, chat_id)
        
        markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
        itembtn_psychologize = types.KeyboardButton('Psychologize')
        itembtn_reset = types.KeyboardButton('Reset')
        markup.add(itembtn_psychologize, itembtn_reset)
        bot.send_message(chat_id, "Processing completed. Choose an option:", reply_markup=markup)



def process_voice_in_background(combined_voice_path, model, chat_id):
    with lock:
        bot.send_message(chat_id, "Please wait...")
        run_all_voice(combined_voice_path, model, chat_id)
        markup = types.ReplyKeyboardMarkup(row_width=1, one_time_keyboard=True)
        itembtn_reset = types.KeyboardButton('Reset')
        markup.add(itembtn_reset)
        bot.send_message(chat_id, "Processing completed.", reply_markup=markup)


def process_psychologize_in_background(combined_text, chat_id):
    with lock:
        bot.send_message(chat_id, "Please wait...")  
        psychologize_output = psychologize(combined_text)
        bot.send_message(chat_id, psychologize_output)  
        markup = types.ReplyKeyboardMarkup(row_width=1, one_time_keyboard=True)
        itembtn_reset = types.KeyboardButton('Reset')
        markup.add(itembtn_reset)





 
 ####################################################  bot

@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(row_width=1, one_time_keyboard=True)
    itembtn_start = types.KeyboardButton('Start')
    markup.add(itembtn_start)
    bot.send_message(message.chat.id, "Press 'Start' to begin.", reply_markup=markup)


@bot.message_handler(func=lambda message: message.text == 'Start')
def show_options(message):
    markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
    itembtn1 = types.KeyboardButton('Chat')
    itembtn2 = types.KeyboardButton('Report')
    itembtn3 = types.KeyboardButton('Chat Bot')
    itembtn4 = types.KeyboardButton('Time schedule')
    itembtn_reset = types.KeyboardButton('Reset')
    markup.add(itembtn1, itembtn2, itembtn3, itembtn4, itembtn_reset)
    bot.send_message(message.chat.id, "Choose an option:", reply_markup=markup)



############## Option 1   'Chat'

@bot.message_handler(func=lambda message: message.text == 'Chat')
def option1_handler(message):
    markup = types.ReplyKeyboardMarkup(row_width=1, one_time_keyboard=True)
    itembtn_finish = types.KeyboardButton('Finish')
    markup.add(itembtn_finish)
    msg = bot.reply_to(message, "Please enter the text or send a voice message:", reply_markup=markup)
    bot.register_next_step_handler(msg, process_text_step)


def process_text_step(message):
    if message.text == 'Finish':
        combined_text = "\n".join([text for text in text_storage if text is not None])
        for voice_path in voice_storage:
            transcribed_text = voice_to_text(model, voice_path)
            if transcribed_text:
                combined_text += "\n" + transcribed_text
        text_storage.clear()  
        voice_storage.clear()  
        
        bot.send_message(message.chat.id, "Please wait...")  
        threading.Thread(target=process_text_in_background, args=(combined_text, message.chat.id)).start()
        
        bot.register_next_step_handler(message, process_psychologize_step, combined_text)

    elif message.content_type in ['voice', 'audio']:
        file_info = bot.get_file(message.voice.file_id if message.content_type == 'voice' else message.audio.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        voice_path = f"{message.message_id}.ogg"  
        with open(voice_path, 'wb') as new_file:
            new_file.write(downloaded_file)
        voice_storage.append(voice_path)
        msg = bot.reply_to(message, "Please enter more text or send more voice messages or press 'Finish':")
        bot.register_next_step_handler(msg, process_text_step)

    else:
        text_storage.append(message.text)
        msg = bot.reply_to(message, "Please enter more text or send more voice messages or press 'Finish':")
        bot.register_next_step_handler(msg, process_text_step)



def process_psychologize_step(message, combined_text):
    if message.text == 'Psychologize':
        bot.send_message(message.chat.id, "Please")
        threading.Thread(target=process_psychologize_in_background, args=(combined_text, message.chat.id)).start()
    elif message.text == 'Reset':
        send_welcome(message)
    else:
        bot.send_message(message.chat.id, "Invalid option, please choose 'Psychologize' or 'Reset'.")
        bot.register_next_step_handler(message, process_psychologize_step, combined_text)
        
def process_psychologize_in_background(combined_text, chat_id):
    with lock:
        bot.send_message(chat_id, "Please wait...") 
        psychologize_output = psychologize(combined_text)
        bot.send_message(chat_id, psychologize_output)  # Send the output of psychologize function
        
        # Add Practice and Reset buttons after psychologize process
        markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
        itembtn_practice = types.KeyboardButton('Practice')
        itembtn_reset = types.KeyboardButton('Reset')
        markup.add(itembtn_practice, itembtn_reset)
        bot.send_message(chat_id, "Psychologize completed. Choose an option:", reply_markup=markup)
        
        bot.register_next_step_handler_by_chat_id(chat_id, process_practice_step, psychologize_output)

def process_practice_step(message, psychologize_output):
    if message.text == 'Practice':
        bot.send_message(message.chat.id, "Please wait...")  # Sending "Please wait..." message
        threading.Thread(target=process_practice_in_background, args=(psychologize_output, message.chat.id)).start()
    elif message.text == 'Reset':
        send_welcome(message)
    else:
        bot.send_message(message.chat.id, "Invalid option, please choose 'Practice' or 'Reset'.")
        bot.register_next_step_handler(message, process_practice_step, psychologize_output)

def process_practice_in_background(psychologize_output, chat_id):
    with lock:
        try:
            practice_output = practice(psychologize_output)

            if not practice_output or not isinstance(practice_output, str):
                practice_output = "The practice function did not return a valid output."

            bot.send_message(chat_id, practice_output)

        except Exception as e:
            bot.send_message(chat_id, f"An error process_practice_in_background: {str(e)}")

        markup = types.ReplyKeyboardMarkup(row_width=1, one_time_keyboard=True)
        itembtn_reset = types.KeyboardButton('Reset')
        markup.add(itembtn_reset)
        bot.send_message(chat_id, "Practice completed. Choose an option:", reply_markup=markup)




####### Option 2       'Report'       

@bot.message_handler(func=lambda message: message.text == 'Report')
def Report_handler(message):
    markup = types.ReplyKeyboardMarkup(row_width=2, one_time_keyboard=True)
    itembtn1 = types.KeyboardButton('Show Table')
    itembtn2 = types.KeyboardButton('Issues')
    itembtn3 = types.KeyboardButton('Severity')
    itembtn4 = types.KeyboardButton('Recommendations')
    itembtn_reset = types.KeyboardButton('Reset')
    markup.add(itembtn1, itembtn2, itembtn3, itembtn4, itembtn_reset)
    bot.send_message(message.chat.id, "Choose an option under Option 2:", reply_markup=markup)




@bot.message_handler(func=lambda message: message.text == 'Show Table')
def show_table(message):
    user_id = message.from_user.id
    filtered_data = filter_sessions_by_user_id(user_id)  

    if filtered_data.empty:
        bot.send_message(message.chat.id, "No data found for your User ID.")
    else:
        for index, row in filtered_data.iterrows():
            response = (
                f"Datetime:\n {row['Datetime']}\n\n\n\n"
                f"Issues:\n {row['Identified Issues']}\n\n\n\n"
                f"Summary:\n {row['Conversation Summary']}\n\n\n\n"
                f"Severity:\n {row['Severity Level']}\n\n\n\n"
                f"Recommendations:\n {row['Treatment Recommendations']}\n\n\n\n"
                f"Practice:\n {row['Practice Table']}\n\n\n\n"
            )
            send_long_message(message.chat.id, response)



@bot.message_handler(func=lambda message: message.text == 'Issues')
def Issues_handler(message):
    user_id = message.chat.id  
    bot.send_message(user_id, "Please wait...")  

    result = Datetime_Issues(user_id)  

    if not result:
        result = "No response generated."

    bot.send_message(user_id, result)  



@bot.message_handler(func=lambda message: message.text == 'Severity')
def Severity_handler(message):
    user_id = message.chat.id  
    bot.send_message(user_id, "Please wait...") 

    result = Datetime_Severity(user_id) 
    if not result:
        result = "No response generated."

    bot.send_message(user_id, result)  



@bot.message_handler(func=lambda message: message.text == 'Recommendations')
def Recommendations_handler(message):
    user_id = message.chat.id  
    bot.send_message(user_id, "Please wait...")  

    result = Datetime_Issues(user_id) 

    if not result:
        result = "No response generated."

    bot.send_message(user_id, result)  







####### Option 3     'Chat Bot'

@bot.message_handler(func=lambda message: message.text == 'Chat Bot')
def chat_selected(message):
    bot.send_message(message.chat.id, "How can I help you?")

    bot.register_next_step_handler(message, chat_conversation_loop)

def chat_conversation_loop(message):
    user_id = str(message.chat.id)

    if user_id not in chat_history:
        chat_history[user_id] = []

    if user_id not in new_conversations:
        new_conversations[user_id] = [] 

    if message.text == 'Finish':
        finish_selected(message)
        return

    if message.content_type == 'voice':
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        voice_file_path = f"voice_message_{user_id}.ogg"
        with open(voice_file_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.send_message(message.chat.id, "Please wait...")
        input_text = voice_to_text(model, voice_file_path)
    else:
        input_text = message.text

    chat_history[user_id].append({"role": "user", "content": input_text})
    new_conversations[user_id].append({"role": "user", "content": input_text})

    bot.send_message(user_id, "Please wait...")

    context_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history[user_id]])
    output = chatGPT_35_turbo(input_text, context_text)

    chat_history[user_id].append({"role": "assistant", "content": output})
    new_conversations[user_id].append({"role": "assistant", "content": output})

    save_chat_history()

    bot.send_message(user_id, output)

    show_finish_option(message)
    bot.register_next_step_handler(message, chat_conversation_loop)

def show_finish_option(message):
    markup = types.ReplyKeyboardMarkup(row_width=1, one_time_keyboard=True)
    itembtn_finish = types.KeyboardButton('Finish')
    markup.add(itembtn_finish)
    bot.send_message(message.chat.id, ".", reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == 'Finish')
def finish_selected(message):
    user_id = str(message.chat.id)
    bot.send_message(message.chat.id, "Please wait...")

    if user_id in new_conversations:
        text_chat_bot = "\n".join([f"{entry['role']}: {entry['content']}" for entry in new_conversations[user_id]])
        run_all_chat_bot(text_chat_bot, user_id)
        new_conversations[user_id] = [] 

    bot.send_message(message.chat.id, "Chat ended. Type /reset to start over.")
    show_reset_option(message)

def show_reset_option(message):
    markup = types.ReplyKeyboardMarkup(row_width=1, one_time_keyboard=True)
    itembtn_reset = types.KeyboardButton('Reset')
    markup.add(itembtn_reset)
    bot.send_message(message.chat.id, "You can reset the bot:", reply_markup=markup)


def load_user_schedule(user_id):
    filename = f'{base_path}/schedules/{user_id}_schedule.json'
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}

def save_user_schedule(user_id, schedule):
    filename = f'{base_path}/schedules/{user_id}_schedule.json'
    os.makedirs('schedules', exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(schedule, file, indent=4)




####### Option 4     'Time schedule'

@bot.message_handler(func=lambda message: message.text == 'Time schedule')
def time_schedule_handler(message):
    user_id = message.from_user.id
    current_date = make_formatted_date()
    schedule = load_user_schedule(user_id)
    

    sorted_schedule = dict(sorted(schedule.items()))
    
    for date, tasks in sorted_schedule.items():
        for task in tasks:
            keyboard = types.InlineKeyboardMarkup()
            

            task_id = f'{date}_{tasks.index(task)}'
            callback_data_dict[task_id] = task 
            
            if date < current_date:
                yes_button = types.InlineKeyboardButton("Yes", callback_data=f'complete_task:{task_id}')
                no_button = types.InlineKeyboardButton("No", callback_data=f'reschedule_task:{task_id}')
                keyboard.add(yes_button, no_button)
                bot.send_message(message.chat.id, f"Task from {date}: {task}\nDid you complete this task?", reply_markup=keyboard)
            
            else:
                yes_button = types.InlineKeyboardButton("Yes", callback_data=f'complete_task:{task_id}')
                keyboard.add(yes_button)
                bot.send_message(message.chat.id, f"Task for {date}: {task}\nMark as complete?", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data.startswith('complete_task:'))
def complete_task_handler(call):
    user_id = call.from_user.id
    task_id = call.data.split(':')[1]
    task = callback_data_dict.get(task_id)
    date = task_id.split('_')[0]
    
    schedule = load_user_schedule(user_id)
    if task in schedule.get(date, []):
        schedule[date].remove(task)
        if not schedule[date]:
            del schedule[date]
    
    save_user_schedule(user_id, schedule)
    bot.answer_callback_query(call.id, "Task marked as complete and removed.")
    bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)

@bot.callback_query_handler(func=lambda call: call.data.startswith('reschedule_task:'))
def reschedule_task_handler(call):
    user_id = call.from_user.id
    task_id = call.data.split(':')[1]
    task = callback_data_dict.get(task_id)
    
    add_task_to_next_available_date(user_id, task)  
    bot.answer_callback_query(call.id, "Task rescheduled to the next available date.")

    old_date = task_id.split('_')[0]
    schedule = load_user_schedule(user_id)
    if task in schedule.get(old_date, []):
        schedule[old_date].remove(task)
        if not schedule[old_date]:
            del schedule[old_date]
    
    save_user_schedule(user_id, schedule)
    bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)





@bot.message_handler(func=lambda message: message.text == 'Reset')
def reset_handler(message):
    send_welcome(message)

bot.polling()





