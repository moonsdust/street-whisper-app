from backend import whisper_with_diarization_as_methods
import os
import magic
import typer
from PyInquirer import prompt
from rich import print as rprint
from pyannote.audio import Pipeline


app = typer.Typer()

def startup_ui(howtouse: bool = typer.Option(False, '-howtouse', help="How to use the tool"),
               credits: bool = typer.Option(False, '-credits', help="Credits")):
    """This function creates a UI based on the command given by the user."""
    if not howtouse and not credits:
        # When no option is passed in, the app will start
        rprint("[magenta]=============================[magenta]")
        rprint("[bold][underline]STREET Lab Whisper App[underline][bold]")
        rprint("[magenta]=============================[magenta]")
        authorization()
    if howtouse and not credits:
        # When -howtouse is used, it will display the help section
        howtouse_ui()
    if not howtouse and credits:
        # When -credits is used, it will display the credits section
        credits_ui()

def validate_path(input_path: str, is_intended_file: bool) -> bool:
    """
    This function checks whether the path specified by string: input_path
    is defined in the user's system.

    If is_intended_audio_file is true, then the expected path should point to a file.
    We can adjust the testing of the file_path based on this parameter
    """
    if (is_intended_file):
        # In this branch, check whether path points to a valid file
        # We test that this file is an audio file later on (in validate_audio_file)
        is_file = os.path.isfile(input_path)
        if is_file:
            return True
        else:
            return False
    else:
        # In this branch, check whether path points to a valid directory
        is_directory = os.path.isdir(input_path)
        if is_directory:
            return True
        else:
            return False

def validate_audio_file(audio_file_path: str) -> bool:
    """
    This function utilizes calls from the python-magic-bin==0.4.14 library to check whether or not the file denoted
    by the path: audio_file_path is a valid audio file that can be interpreted by Whisper.

    Preconditions:
        - Audio file inputs are either .wav or .mp3. Whisper can process more audio file inputs, but the checking for
        other "types" of files has not been implemented yet
    """
    try:
        validate_audio_path_msg = magic.from_file(audio_file_path, mime=True)
        supported_file_extensions = {"mpeg", "mp4", "wav", "webm", "flac", "ogg", "adts"}
        #Note: In above line, mpeg include checks for mpeg, mp3 and mpga. mp4 includes checks for .mp4 and .m4a
        # adts files can include some audio files disguised as mp3/mp4
        for file_ext in supported_file_extensions:
            if file_ext in validate_audio_path_msg:
                return True
        print("The file type is not supported by Whisper. If you think this is not the case, please contact the developers.")
        return False
    except: 
        print("There was an error reading in the name of the file because it contains a non UTF-8 character. Update the file name and try again.")
        return False

def authorization():
    """This function deals with access token authentication, catching errors, keyboard interruptions, and more."""
    access_token_prompt = [
        {
            'type': 'password',
            'message': 'What is your access token? An access token from Hugging Face and accepting \npyannote/speaker-diarization-3.1\'s user condition is needed to run the app.\nIf this is not your first time running the app and you have previously entered a\nvalid token, press enter on your keyboard.\n',
            'name': 'password'
        }
    ]
    potential_access_token = prompt(access_token_prompt)
    if (potential_access_token == {} or potential_access_token["password"].lower() == "exit"):
        return
    try:
        # Check token
        diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=str(potential_access_token["password"]))
        questions_ui(diarize_model)
        typer.Exit()
    except KeyError:
        # You reach here if you click on a selection in the prompt selection instead of
        # using your keyboard
        invalidKeyError  = [
            {
                'type': 'list',
                'name': 'key_error',
                'message': 'Please only use your keyboard and the enter key when making a selection.',
                'choices': [
                    {
                        'name': 'I would like to try again',
                    },
                    {
                        'name': 'Exit the app',
                    }
                ],
            }
        ]
        key_error = prompt(invalidKeyError)
        if key_error != {} and key_error["key_error"] == 'I would like to try again':
            authorization()
        else:
            typer.Exit()

    except KeyboardInterrupt:
        typer.Exit()
    except:
        invalidTokenPrompt = [
            {
                'type': 'list',
                'name': 'invalid_token',
                'message': 'This token is invalid. Do you want to try again?',
                'choices': [
                    {
                        'name': 'Yes',
                    },
                    {
                        'name': 'No',
                    }
                ],
            }
        ]
        invalid_token = prompt(invalidTokenPrompt)
        if invalid_token["invalid_token"] == 'Yes':
            authorization()
        else:
            typer.Exit()

def questions_ui(diarize_model):
    """This function asks prompts about the audio file that the user wants to translate
    or perform transcription on and conducts file checks. Afterwards, information
    is passed to backend."""
    rprint("[magenta]=============================[magenta]")
    rprint(f"You can now go offline.[bold]")
    rprint("[magenta]=============================[magenta]")
    # Process selection
    translation_transcription_prompt = [
        {
            'type': 'list',
            'name': 'process_selected',
            'message': 'What process do you want to run on your audio file?',
            'choices': [
                {
                    'name': 'Transcription Only',
                },
                {
                    'name': 'Translation Only',
                },
                {
                    'name': 'Transcription + Translation Only',
                },
                {
                    'name': 'Exit the app',
                }
            ],
        }
    ]
    process_selected = prompt(translation_transcription_prompt)
    if process_selected["process_selected"] == 'Exit the app':
        return
    # Input file
    rprint("[blue]=============================[blue]")
    rprint(f"[bold]Enter the absolute path to the audio file you want to do the \"{process_selected['process_selected']}\" process on:[bold]")

    # Check if audio file path is a valid path within system that CLI is running from

    while True:
        input_file = input()
        input_audio_path = input_file.strip()  # remove leading and trailing whitespace from the input path ONLY
        is_valid_audio_path = validate_path(input_audio_path, True)
        if is_valid_audio_path:
            break
        else:
            print("You entered an invalid audio path. Please try again:")

    # Check if the referenced audio file itself is one that Whisper can process
    is_valid_audio_file = validate_audio_file(input_audio_path)
    if not (is_valid_audio_file):
        return

    rprint("[blue]=============================[blue]")
    # Is Input File in English?
    to_eng_selection_prompt = [
        {
            'type': 'list',
            'name': 'to_english_selection',
            'message': 'Is your audio file in English?',
            'choices': [
                {
                    'name': 'Yes',
                },
                {
                    'name': 'No',
                },
                {
                    'name': 'Exit the app',
                },
            ],
        }
    ]
    to_english_selection = prompt(to_eng_selection_prompt)
    if to_english_selection["to_english_selection"] == 'Exit the app':
        return

    rprint("[blue]=============================[blue]")
    # Model size selection
    model_size_selection_prompt = [
        {
            'type': 'list',
            'name': 'model_size_selection',
            'message': 'What model size do you want to use on your audio file?',
            'choices': [
                {
                    'name': 'large-v2',
                },
                {
                    'name': 'small',
                },
                {
                    'name': 'medium',
                },
                {
                    'name': 'Exit the app',
                },
            ],
        }
    ]
    model_size_selection = prompt(model_size_selection_prompt)
    if model_size_selection["model_size_selection"] == 'Exit the app':
        return
    rprint("[blue]=============================[blue]")
    # Destination Folder
    rprint(f"[bold]Enter the absolute path to your destination folder:[bold]")

    # Check if dest file path is a valid path within system that CLI is running from
    while True:
        destination_selection = input()
        destination_selection = destination_selection.strip()  # remove leading and trailing whitespace
        is_valid_dest_path = validate_path(destination_selection, False)
        if is_valid_dest_path:
            break
        else:
            print("You entered an invalid destination path. Please try again:")

    rprint("[blue]=============================[blue]")
    questions_finished_prompt = [
            {
                'type': 'list',
                'name': 'questions_finished',
                'message': f'Do you want to start the process? If not, this app will quit.',
                'choices': [
                            {
                                'name': 'Yes',
                            },
                            {
                                'name': 'No',
                            },
                ],
            }
        ]
    questions_finished = prompt(questions_finished_prompt)
    if questions_finished["questions_finished"] == 'Yes':
        # Run process
        whisper_with_diarization_as_methods.main(process_selected["process_selected"], input_file, to_english_selection["to_english_selection"], model_size_selection["model_size_selection"], destination_selection, diarize_model)
    else:
        # Exit out of app
        typer.Exit()

#TODO: Need to complete this function later
# How to use (previously: help) section
# (note: this is different from the option --help, which list out all the options the user can use)
# Called by using the option -howtouse
def howtouse_ui():
    """This function creates the how to use section."""
    rprint("[magenta]=============================[magenta]")
    rprint("[bold][underline]STREET Lab Whisper App[underline][bold]")
    rprint("")
    rprint("[bold]How to Use[bold]")
    rprint("[magenta]=============================[magenta]")

# Credits section
# Called by using the option -credits
def credits_ui():
    """This function creates the credits section."""
    rprint("[magenta]=============================[magenta]")
    rprint("[bold][underline]STREET Lab Whisper App[underline][bold]")
    rprint("")
    rprint("[bold]Credits[bold]")
    rprint("This application was created by STREET Lab: https://www.streetlab.tech/ ")
    rprint("For details about the technologies and libraries used, visit the following repository: https://github.com/street-lab-tech/street-whisper-app")
    rprint("[magenta]=============================[magenta]")

if __name__ == "__main__":
    typer.run(startup_ui)
