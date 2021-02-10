"""
Progress Transcript - allow stdout to terminal and captured in a file too
                      with the ability to log a progress bar when triggered by 'actions'

IMPORTANT: In a Jupyter notebook, the text and progress bars for later cells will
           show up after the first cell where progress_transcript is started. The
           transcript can be stopped and started again after a particular cell,
           if output is desired there.

Purpose:   Rather than logging every updated line of the progress bar or none, log only the
           last line, i.e., final outcome of the progress bar.
         
Usage:
    # first notebook cell (if a notebook is used)
    import progress_transcript
    progress_transcript.start('logfile.log')

    print('code without a progress bar here')

    # another notebook cell (if a notebook is used)
    # if output is desired after this cell, instead of the first
    # then optionally stop and start progress_transcript again
    #   progress_transcript.stop()
    #   progress_transcript.start('logfile.log')
    print('ACTION:START_PROGRESS_BAR')     # this is not printed or logged
    print('code with a progress bar here')
    print('ACTION:STOP_PROGRESS_BAR')      # this is not printed or logged
    
    progress_transcript.stop()
    print('code after logging')
"""
# Written by André Carrington
# Adapted from Transcript, written by Amith Koujalgi and Brian Burns
# The original version is one of the last posts by Brian Burns here:
# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

import sys

an_obj = []

class Progress_Transcript(object):

    def __init__(self, filename):
        self.terminal     = sys.stdout
        self.logfile      = open(filename, "a")
        self.progress     = False
        self.latest_line  = ''
        
    def write(self, message):
        if message == 'ACTION:START_PROGRESS_BAR':
            self.progress = True
            return
            
        if message == 'ACTION:STOP_PROGRESS_BAR':
            self.logfile.write(self.latest_line)
            self.progress     = False
            self.latest_line  = ''
            return

        if self.progress and len(message)>0:
            lines = message.splitlines()
            if len(lines[-1])>0:
                self.terminal.write(message)
                self.latest_line = lines[-1]
            elif len(lines)>1:
                self.terminal.write(message)
                self.latest_line = lines[-2]
            else:
                # message is just '\n', ignore it
                pass
        else:
            self.terminal.write(message)
            self.logfile.write(message)
                    
    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass

def start(filename):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Progress_Transcript(filename) 

def stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
