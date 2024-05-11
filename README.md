It's amazing what a little inspiration can do for an idiot like me. This project came from a couple of comments I saw on LinkedIn 
and I figured this would be a good challenge to undertake. Here's the core of what's going on with this repository:

1) Download a YouTube video
2) Extract the audio
3) Transcribe the audio into text
4) If the audio file size is greater than 25 MB, break it into chunks and translate. Otherwise, just translate it
5) Assemble the chunks into a single text file
6) Use the text file as material for the underlying LLM
7) Give a synopsis of the text

I ran into multiple issues building this. First, PyDub is a bit of an asshole. Here are the steps I took to get it working:

1) https://www.ffmpeg.org/download.html
   a. I don't have a Mac, so you'll have to play around a bit with this part, but this link:
         https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
   b. Your experience may vary. I'm trying my best to muddle way through as I go and share what I know.
   
2) Put the /bin folder in your system path
   a. Type "path" in your search and click the icon that says "edit the system environment variables
   b. Click "Environment Variables..."
   c. There should be two areas to enter information. Select "New" for both
   d. Name the variable something like "ffmpeg" or whatever else you want
   e. Paste the path to your ffmpeg folder downloaded from step one
   f. Click "Ok"
   g. VS Code needs to be restarted to recognize the path
   
3) In your package manager (I use Anaconda), paste the following then update your index:
         conda install -c conda-forge pydub

Make sure to set your own file paths. I included plenty of print statements to help track progress. Further, if you don't have
Ollama, you'll stick with a GPT LLM. I've had success with Llama3 7B and Gemma 2B models. Ollama is surprisingly easy to use, 
so you know. Eventually, I'm going to dig into HuggingFace and its Transformer models, but that'll wait for another day.

Let me know if that doesn't work. I'm not sure how much help I can be, but I'll damn sure try. Whisper was kind of an asshole also. 
There seem to be multiple ways to go about doing it, but the way I set up my code was the winning ticket. Although I used "base" in
my model, there are more robust settings to use. As always, set your environment variables and use your damn API key. 

I'll check this again and make updates as needed. Until then, enjoy!
