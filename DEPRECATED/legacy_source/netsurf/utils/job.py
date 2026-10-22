# Basic imports
import os

# Hashlib
import hashlib

# Log utils 
from . import log

# Pandas 
import pandas as pd

# Import netsurf
import netsurf

""" Check if a given pid is running """
def is_pid_running(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


""" Get process from a given job """
def get_process_from_job(job_progress_file, job_pid_file, verbose = True):
    
    is_running = False

    """ First pid """
    pid = None
    if os.path.exists(job_pid_file):
        # Check if the pid is running
        try:
            with open(job_pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if pid is running
            is_running = is_pid_running(pid)

        except Exception as e:
            if verbose: log._error(f"An error occurred while trying to get the pid from the job: {e}")

    else:
        job_pid_file = None
        if verbose: log._error(f"The job pid file '{job_pid_file}' does not exist.")
    
    # Okay, if the pid is running, then let's try to open the progress file to see if we can manage to get the progress out of it
    """ Now progress, if pid is running """
    progress = 0.0
    pseudostate = 'error'
    if os.path.exists(job_progress_file):
        try:
            with open(job_progress_file, 'r') as f:
                # The progress file contains two numbers separated by a comma, the first is the pseudostatus and the second is the progress
                pseudo = f.read().strip().split(',')
                progress = float(pseudo[1])
                pseudostate = str(pseudo[0])

                # If progress >= 1., we can force the pseudostate to be 'completed'
                if progress >= 1.0: pseudostate = "completed"
            
            # We don't care that much about that pseudo status/progress because in reality it might happen that this is a run
            # that was initialized outside of the gui, which means that it started with different combos of TMR/BEr, etc.
            # still, we can pass this on to the gui to show the progress
        except Exception as e:
            if verbose: log._error(f"An error occurred while trying to get the progress from the job: {e}")
    else:
        job_progress_file = None
        if verbose: log._error(f"The job progress file '{job_progress_file}' does not exist.")
    
    # If it's not running ...
    if not is_running:
        # ... but we have a pid file, then we can delete it 
        if job_pid_file is not None:
            try:
                os.remove(job_pid_file)
                if verbose: log._log(f"Deleted pid file '{job_pid_file}'")
            except Exception as e:
                if verbose: log._error(f"An error occurred while trying to delete the pid file: {e}")
            job_pid_file = None
            pid = None

        # Now, if in the progress file it said that the job was not completed, something happened. Like an error.
        # This run is incomplete
        if progress is not None and progress < 1.0 and pseudostate != "completed":
            pseudostate = "incomplete"

    # Finally, if progress is None, set it to 0
    if progress is None: progress = 0.0
        
    return is_running, pseudostate, progress, pid, job_pid_file, job_progress_file


# Get nodus jobs 
def get_nodus_jobs_for_config(config = {}):
    # Get the netsurf nodus db connector 
    nodus_db = netsurf.nodus_db
    # Job manager obj
    job_manager = nodus_db.job_manager

    # Get all jobs 
    jobs = job_manager.get_jobs()

    # Turn jobs into a dataframe with columns:
    """
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            nodus_session_id TEXT NOT NULL,
            parent_caller TEXT NOT NULL,
            job_name TEXT,
            status TEXT DEFAULT 'waiting',  -- 'waiting', 'running', 'complete', 'failed'
            timestamp TEXT NOT NULL,
            completion_time TEXT,
            log_path TEXT,
            pid TEXT DEFAULT NULL,
            config TEXT
    """
    #['job_id', 'nodus_session_id', 'parent_caller', 'job_name', 'status', 'completion_time', 'log_path', 'pid', 'config', 'command', 'priority', 'script_path']
    nodus_columns = ['job_id', 'nodus_session_id', 'parent_caller', 'job_name', 'status', 'timestamp', 'completion_time', 'log_path', 'pid', 'config', 'command', 'priority', 'script_path']
    jobs_df = pd.DataFrame(jobs, columns = nodus_columns)

    # Now filter only those that have the same config
    if len(config) > 0:
        jobs_df = jobs_df[jobs_df['config'] == str(config)]
    
    return jobs_df

def run_jobs_with_nodus(commands):
    # Get the netsurf nodus db connector 
    nodus_db = netsurf.nodus_db
    # Job manager obj
    jm = nodus_db.job_manager

    # First we need to get and run the parents 
    # Get unique parents 
    g = commands.groupby('parent')
    # Loop thru parent groups
    for parent, df in g:
        # Parent job hash
        hash = hashlib.md5(parent.encode()).hexdigest()

        # Create job for parent
        job_id_parent, job_parent = jm.create_job(
            name=f"model_training_{hash}",
            parent_caller="netsurf",
            job_type="command",
            command = parent
        )

        # Now for the rest of the commands that have this as a parent, this job_id is a dependency
        for i, row in df.iterrows():
            cmd = row['command']
            hash = hashlib.md5(cmd.encode()).hexdigest()
            # Create job
            job_id, job = jm.create_job(
                name=f"model_training_{hash}",
                parent_caller="netsurf",
                job_type="command",
                command = cmd,
                dependencies = [job_id_parent]
            )
            
    return True
