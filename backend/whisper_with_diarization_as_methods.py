## Import statements ##
import whisper
# import pandas as pd # Commented out import since the method importing it is not in current use
import csv
import time
from datetime import datetime
import magic
from typing import Any, List, Optional
from pyannote.audio import Pipeline
from backend.merge_timestamps import diarize_text
from iso639 import Lang
import torch


def validate_audio_file(audio_file_path: str) -> bool:
    """
    This function utilizes calls from the python-magic-bin==0.4.14 library to check whether or not the file denoted
    by the path: audio_file_path is a valid audio file that can be interpreted by Whisper.

    Preconditions:
        - Audio file inputs are either .wav or .mp3. Whisper can process more audio file inputs, but the checking for
        other "types" of files has not been implemented yet
    """
    validate_audio_path_msg = magic.from_file(audio_file_path, mime=True)
    supported_file_extensions = {"mpeg", "mp4", "wav"} #mpeg include checks for mp3, mp4 includes checks for .mp4 and .m4a
    for file_ext in supported_file_extensions:
        if file_ext in validate_audio_path_msg:
            return True
    return False

def define_whisper_model(model_path: str):
    """
    This method downloads a Whisper model by loading in a .pt file in the directory
    specified by parameter model_path

    This essentilly "choses" the model used for transcription/translation

    For reference, here are possible Whisper model sizes:
    ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small',
    'medium.en', 'medium', 'large-v1', 'large-v2', 'large']

    NOTE: The file being locaed MUST be a .pt file

    Preconditions:
        - model_path is a valid path to a .pt file

    :param model_path: Local path of the Whisper model
    :return: A Whisper Model Object
    """
    # TODO: Need to handle cases where invalid model path is passed, need to add conditional checks for that
    # TODO #2: Need to determine what is the return type of the model?
    whisper_model = whisper.load_model(model_path)
    return whisper_model

def detecting_language(whisper_model: Any, audio_file_path: str) -> str:
    """
    This method takes in a Whisper Model instances, and an audio file with the path as specified by parameter
    audio_file_path.

    It then calls lower level code in Whisper to return a String with what languaeg Whisper automatically
    detects in the first approx 30 sec of the audio in the audio file.

    Throughout out this file, this language will be referenced as the "original language" or "autodetected language"
    :param whisper_model: Any
    :param audio_file_path: str
    :return: str

    ATTRIBUTION: The code in this method is credited to the official Whisper repo (https://github.com/openai/whisper)
    """
    audio = whisper.load_audio(audio_file_path)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)
    detected_lang_code = str(max(probs, key=probs.get))
    full_language = Lang(detected_lang_code).name # decoding the language code
    return full_language

def transcribe_audio(whisper_model: Any, audio_file_path: str, is_translate: Optional[bool] = False):
    """
    This method takes an audio file with the path as specified by parameter audio_file_path.
    It also takes a boolean is_translate. If true, we wish to translate the transcribed text to English.
    If this is false, then we transcribe the audio file based on the autodetected language

    It then passes in both of these variables to the whisper model's transcribe method.

    Finally, a list of segments containing the timestamps and transcribed/translated text is extracted

    :param audio_file_path: str
    :param is_translate: bool
    :return: Any

    Preconditions:
        - Audio file path is defined and links to a .wav file.
    """
    if is_translate:
        transcription = whisper_model.transcribe(audio=audio_file_path, task="translate", fp16=False, verbose=False)
    else:
        transcription = whisper_model.transcribe(audio=audio_file_path, fp16=False, verbose=False)

    return transcription

def retrieving_speaker_diaz(pipeline_file: str, audio_file_path: str):
    """
    This method initiaties a local version of the Version 3.1 Pyannote Speaker Diarization pipeline
    through reading from a provided config.yaml file

    It then returns speaker diarization from the audio file whose path is passed in (via argument "audio_file_path")

    Preconditions:
        - The path to the config.yaml file passed in exists AND correctly corresponds to the pipeline outlined above.

    Notes:
        - The pipeline should auto-detect the number of speakers in the file,
        but if you want to specify, can pass in addditional argument: num_speakers=2
    """
    speaker_diarization_pipeline = Pipeline.from_pretrained(pipeline_file)
    pipeline_result = speaker_diarization_pipeline(audio_file_path)
    return pipeline_result

def display_timestamps_speaker_and_text(whisper_result, speaker_diaz_result):
    """
    This function takes the Whisper transcription result (through argument whisper_result) and
    returns an object combinining timestamps, speaker identification and text

    Code from: https://github.com/yinruiqing/pyannote-whisper
    :param whisper_result:
    :param speaker_diaz_result:
    :return:
    """
    return diarize_text(whisper_result, speaker_diaz_result)

# def write_to_csv_both_lang_speakers(csv_headers: list[str], csv_file_path: str,
#                                     comb_result_autodetect, comb_result_eng):
#     """
#     This method attempts to make a CSV with the following format
#
#     Column 1: Timestamps (in HH:MM:SS) format (hour, minute, second)
#     Column 2: Speaker No (String denoting the identification of speakers)
#     Column 3: Column is named Text[Orig Lang] if the comb_result contains text
#     based on Whisper's autodetected language. Otherwise, it is named Text[Eng]
#
#     Preconditions:
#         - csv_headers contains only the column names listed above
#         - Valid combined eng and autodetected language objects are passed into this method
#         - The length of the Timestamps and Speaker No in the 2 CSV files are the same
#         - Likewise, the content of these 2 columns are assumed to be the same for each entry
#     """
#     autodetect_csv_content = gen_group_speakers_csv_content(comb_result_autodetect)
#     eng_csv_content = gen_group_speakers_csv_content(comb_result_eng)
#
#     with open(csv_file_path, "w") as comb_lang_csv_file:
#         comb_lang_csv_writer = csv.writer(comb_lang_csv_file)
#         comb_lang_csv_writer.writerow(csv_headers)  # Write the header row
#         for i in range(len(autodetect_csv_content)):
#             autodetect_csv_content[i].append(eng_csv_content[i][-1])
#             comb_lang_csv_writer.writerow(autodetect_csv_content[i])

def gen_group_speakers_csv_content(comb_result) -> List:
    """
    This method returns a List that contain the parsed information
    from object comb_result. This information is intended to be written into a CSV file
    where the speakers are GROUPED TOGETHER / CLUBBED TOGETHER.

    Eg, if the segments are as following:

    00:00:00 - 00:00:15 | Speaker 0 | "Hello."
    00:00:15 - 00:00:40 | Speaker 0 | "I am a cat"
    00:00:40 - 00:00:50 | Speaker 1 | "A cat?"
    00:00:50 - 00:00:55 | Speaker 0 | "Yes."

    Then the returned List should contain:

    [["00:00:00 - 00:00:40","Speaker 0", "Hello. I am a cat "],
    ["00:00:40 - 00:00:50","Speaker 1", "A cat?"],
    ["00:00:50 - 00:00:55","Speaker 0", "Yes."]]

    The generated CSV is expected to have the following format:

    00:00:00 - 00:00:40 | Speaker 0 | "Hello. I am a cat "
    00:00:40 - 00:00:50 | Speaker 1 | "A cat?"
    00:00:50 - 00:00:55 | Speaker 0 | "Yes."
    """
    curr_speaker = comb_result[0][1]  # Denotes the speaker that is currently "speaking" in the iteration
    initial_seg = comb_result[0][0]
    start_timestamp_as_time_obj = time.gmtime(float(initial_seg.start))
    beg_speaker_timestamp = time.strftime("%H:%M:%S", start_timestamp_as_time_obj)

    end_timestamp_as_time_obj = time.gmtime(float(initial_seg.end))
    end_speaker_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)

    speaker_text_seg = comb_result[0][2]

    csv_content = []

    for i in range(1, len(comb_result)):
        seg = comb_result[i][0]
        speaker = comb_result[i][1]  # Denotes the speaker that is currently "speaking" in the iteration

        if (speaker == curr_speaker) and i < len(comb_result) - 1:
            speaker_text_seg = speaker_text_seg + comb_result[i][2]
            end_timestamp_as_time_obj = time.gmtime(float(seg.end))
            end_speaker_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)

        elif (speaker == curr_speaker) and i == len(comb_result) - 1:
            speaker_text_seg = speaker_text_seg + comb_result[i][2]
            end_timestamp_as_time_obj = time.gmtime(float(seg.end))
            end_speaker_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)
            # In addition to the above, since we reached the end of the iteration, need to write to csv file
            row_to_write = []
            full_timestamp = beg_speaker_timestamp + "-" + end_speaker_timestamp
            row_to_write.append(full_timestamp)

            row_to_write.append(curr_speaker)

            row_to_write.append(speaker_text_seg)

            csv_content.append(row_to_write)

        else:
            # if reach here, speaker changed. This is where we write to csv file

            # Step 1: define list to write values into
            row_to_write = []

            # step 2: retrieve end timestamp value and write full timestamp value to row
            full_timestamp = beg_speaker_timestamp + "-" + end_speaker_timestamp
            row_to_write.append(full_timestamp)

            # step 3: write speaker into row
            row_to_write.append(curr_speaker)

            # step 4: write the text into row
            row_to_write.append(speaker_text_seg)

            # step 5: write entire row into csv
            csv_content.append(row_to_write)

            # step 6: change value of curr_speaker to speaker (officially "switching" the counter variable curr_speaker to reflect actual value from variable speaker)
            curr_speaker = speaker

            # step 7: modify the value of the beginning timestamp to reflect when the new speaker starts talking
            start_timestamp_as_time_obj = time.gmtime(float(seg.start))
            beg_speaker_timestamp = time.strftime("%H:%M:%S", start_timestamp_as_time_obj)

            # step 8: change end_speaker_timestamp value to be the most recent end timestamp (resetting value)
            end_timestamp_as_time_obj = time.gmtime(float(seg.end))
            end_speaker_timestamp = time.strftime("%H:%M:%S", end_timestamp_as_time_obj)

            # step 9: change value of speaker_text_seg to be the text seg corresponding to the speaker at this current time
            speaker_text_seg = comb_result[i][2]

            if i == len(comb_result) - 1:
                row_to_write = []
                full_timestamp = beg_speaker_timestamp + "-" + end_speaker_timestamp
                row_to_write.append(full_timestamp)
                row_to_write.append(speaker)
                row_to_write.append(speaker_text_seg)
                csv_content.append(row_to_write)
    return csv_content

def write_list_to_csv(list_of_csv_content: List[str], output_csv_path: str, output_csv_headers: List[str]) -> None:
    """
    This method writes a list of strings (which is the expected output from the method gen_group_speakers_csv_content
    into a CSV file with path defined by parameter output_csv_path
    :param list_of_csv_content:
    :return:None
    """
    with open(output_csv_path, "w") as comb_lang_csv_file:
        comb_lang_csv_writer = csv.writer(comb_lang_csv_file)
        comb_lang_csv_writer.writerow(output_csv_headers)  # Write the header row
        for i in range(len(list_of_csv_content)):
            comb_lang_csv_writer.writerow(list_of_csv_content[i])

# def merging_csv_files(orig_lang_csv_file: str, eng_lang_csv_file: str, comb_lang_csv_file: str):
#     """
#     This function merges 2 CSV files.
#     The 1st CSV file contains the following 3 column names:
#          - Timestamps
#          - Speaker No
#          - Text[Orig_lang]
#
#     The 2nd CSV file contains the following 3 column names:
#         - Timestamps
#         - Speaker No
#         - Text[Eng]
#
#     The return would be another csv file (with the path specified by parameter csv_file_path) which has
#     the following 4 column names
#          - Timestamps
#          - Speaker No
#          - Text[Orig_lang]
#          - Text[Eng]
#
#     Preconditions:
#         - The length of the Timestamps and Speaker No in the 2 CSV files are the same
#         - Likewise, the content of these 2 columns are assumed to be the same for each entry
#     """
#     first_csv_file_as_df = pd.read_csv(orig_lang_csv_file)
#     second_csv_file_as_df = pd.read_csv(eng_lang_csv_file)
#     eng_column = second_csv_file_as_df["Text[Eng]"]
#
#     # Creating a new csv file with path denoted by parameter comb_lang_csv_file
#     dframe1 = pd.DataFrame(first_csv_file_as_df["Timestamps"])
#     data1 = pd.concat([dframe1, first_csv_file_as_df["Speaker No"]], axis=1)
#     data2 = pd.concat([data1, first_csv_file_as_df["Text[Orig Lang]"]], axis=1)
#     data3 = pd.concat([data2, eng_column], axis=1)
#     data3.to_csv(comb_lang_csv_file, encoding='utf-8', index=False)

def main(process_selected: str, input_file: str, to_english_selection: bool, model_size_selection: str, destination_selection: str, diarize_model):
    # Step 1: Defining input audio path + defining CSV Headers
    input_audio_path = input_file # Insert audio file name and extension here (extensions can include: .mp3, .wav)

    if process_selected == "Transcription Only":
        output_csv_headers = ["Timestamps", "Speaker No", "Text[Orig Lang]"] # Insert your headers here by replacing values of empty strings. Eg: ["Timestamps", "Speaker No", "Text[Eng]"]
        output_format = "transcription"
    elif process_selected == "Translation Only":
        output_csv_headers = ["Timestamps", "Speaker No", "Text[Eng]"]
        output_format = "translation"
    else: # reaching here means: process_selected == 'Transcription + Translation Only'
        output_csv_headers = ["Timestamps", "Speaker No", "Text[Orig Lang]", "Text[Eng]"]
        output_format = "transcribe&translate"

    now = datetime.now()
    audio_path_last_backslash_index = input_file.rfind("/")
    audio_name = input_file[audio_path_last_backslash_index + 1:]
    output_csv_path = destination_selection + "/" + audio_name + "_" + output_format + str(now.hour) + str(now.minute) + ".csv" #TODO: There is a bug where inserting a ":" in between the hours and minutes causes an extra '\' to appear
    translate_to_english = to_english_selection    # True denotes that if audio file is not in english, you want to translate text to english. If False, text would be transcribed based on autodetected language from Whisper

    # Step 2: Check if audio file is in valid format
    is_valid_audio_file = validate_audio_file(input_audio_path)
    if (is_valid_audio_file):
        # Step 3: Defining whisper model
        loaded_whisper_model = define_whisper_model(model_size_selection)

        # Step 4: Processing and printing out detected language
        whisper_detect_lang = detecting_language(loaded_whisper_model, input_audio_path)
        print(f'Detected language in input audio file: {whisper_detect_lang}\n')

        print("Speaker diarization has started, in progress\n")
        diarize_model = diarize_model
        the_audio = whisper.load_audio(input_audio_path, 16000)
        audio_data = {
            'waveform': torch.from_numpy(the_audio[None, :]),
            'sample_rate': 16000
        }
        diarization_result = diarize_model(audio_data)
        print("Speaker diarization has completed\n")

        # Step 6: Running conditional checks. The code to run will differ based on whether detected language is ENG or not.
        if (process_selected == "Transcription Only"):
            print("Transcribing audio file\n")
            transcript_whisper_result = transcribe_audio(loaded_whisper_model, input_audio_path)
            transcript_final_result = display_timestamps_speaker_and_text(transcript_whisper_result,
                                                                             diarization_result)
            transcript_csv_content = gen_group_speakers_csv_content(transcript_final_result)
            print("Finished transcribing audio file. Writing output as a CSV file to destination...\n")
            write_list_to_csv(transcript_csv_content, output_csv_path, output_csv_headers)
            print("CSV file has been created. Process is complete\n")

        elif (process_selected == "Translation Only" or translate_to_english == "Yes"):
            print("Translating audio file to English\n")
            trans_whisper_result = transcribe_audio(loaded_whisper_model, input_audio_path, is_translate=True)
            trans_lang_final_result = display_timestamps_speaker_and_text(trans_whisper_result, diarization_result)
            trans_csv_content = gen_group_speakers_csv_content(trans_lang_final_result)
            print("Finished translating audio file to English. Writing output as a CSV file to destination...\n")
            write_list_to_csv(trans_csv_content, output_csv_path, output_csv_headers)
            print("CSV file has been created. Process is complete\n")

        else: #If reached here, then process_selected == "translate_+_transcribe"
            print("Transcribing audio file\n")

            transcript_whisper_result = transcribe_audio(loaded_whisper_model, input_audio_path)
            transcript_final_result = display_timestamps_speaker_and_text(transcript_whisper_result,
                                                                          diarization_result)
            transcript_csv_content = gen_group_speakers_csv_content(transcript_final_result)
            print("Done transcription\n")

            print("Now, translating audio file to English\n")
            trans_whisper_result = transcribe_audio(loaded_whisper_model, input_audio_path, is_translate=True)
            trans_lang_final_result = display_timestamps_speaker_and_text(trans_whisper_result, diarization_result)
            trans_csv_content = gen_group_speakers_csv_content(trans_lang_final_result)
            print("Done translation\n")

            print("Finished both transcription and translation. Writing output as a CSV file to destination...\n")


    else:
        print("Invalid file format. Please try again")
