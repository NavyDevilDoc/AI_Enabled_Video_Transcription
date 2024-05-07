It's amazing what a little inspiration can do for an idiot like me. This project came from a couple of comments I saw on LinkedIn and I figured this would be a good challenge to undertake. 
Here's the core of what's going on with this repository:

1) Download a YouTube video
2) Extract the audio
3) Transcribe the audio into text
4) If the audio file size is greater than 25 MB, break it into chunks and translate. Otherwise, just translate it
5) Assemble the chunks into a single text file
6) Use the text file as material for the underlying LLM
7) Give a synopsis of the text

I ran into multiple issues building this. First, PyDub is kind of an asshole. Here are the steps I took to get it working:
1) https://www.ffmpeg.org/download.html
2) Put the /bin folder in your system path
3) conda install -c conda-forge pydub

Let me know if that doesn't work. I'm not sure how much help I can be, but I'll damn sure try. Whisper was kind of an asshole also. 
There seem to be multiple ways to go about doing it, but the way I set up my code was the winning ticket. Although I used "base" in
my model, there are more robust settings to use. As always, set your own environment variables and use your own damn API key. 

I'll check this again and make updates as needed. Until then, enjoy!
