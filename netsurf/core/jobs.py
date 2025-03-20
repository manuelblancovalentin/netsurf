""" Define any objects/functions used to communicate with the OS """

# Basic imports 
import os 

# Import utils
from netsurf import utils

""" Class to define a JOB that will be run (or that has been run) """
class RunJob:
    def __init__(self, dir, name = 'job', status = None, **kwargs):
        # Set job directory
        self.dir = dir
        self.name = name
        self.status = status

        # Init some vars
        self.log = None
        self.sh = None

        # Now search for job files 
        self.job_files = self.get_job_files(**kwargs)

        # Parse job files
        self.pseudostatus, self.is_running, self.progress, self.pid = self.parse_job_files(**kwargs)
        
    """ Representation """
    def __repr__(self, tab = 0, print_files = True, print_header = True, dir = None):
        tabs = '  ' * tab
        if dir is None:
            dir = self.dir
        d = f' @ {dir}' if len(dir) > 0 else ''
        msg = f'{tabs}[RunJob]: {self.name}{d}'
        msg += f'\n{tabs}    Status: {self.pseudostatus}'
        msg += f'\n{tabs}    Running: {self.is_running}'
        msg += f'\n{tabs}       PID: {self.pid}' if self.pid is not None else ''
        msg += f'\n{tabs}    Progress: {self.progress*100:3.1f}%'
        if print_files:
            for job_file, (job_file_path, job_file_index) in self.job_files.items():
                if job_file_path is not None:
                    msg += f'\n{tabs}    {job_file} @ .../{os.path.basename(job_file_path)}'
        return msg
    
    """ Get job files """
    def get_job_files(self, next = False, verbose = True):
        # Init expected file paths (and get latest ones)
        job_progress_file, jpfi = utils.get_latest_suffix_for_file(os.path.join(self.dir, 'job.progress'), next = next, return_index = True, verbose = False, next_if_empty = True)
        job_log_file, jlfi = utils.get_latest_suffix_for_file(os.path.join(self.dir, 'job.log'), next = next, return_index = True, verbose = False, next_if_empty = True)
        job_sh_file, jsfi = utils.get_latest_suffix_for_file(os.path.join(self.dir, 'job.sh'), next = next, return_index = True, verbose = False, next_if_empty = True)
        job_pid_file, jifi = utils.get_latest_suffix_for_file(os.path.join(self.dir, 'job.pid'), next = next, return_index = True, verbose = False, next_if_empty = True)

        job_files = {'job.progress': (job_progress_file, jpfi), 
                     'job.log': (job_log_file, jlfi), 
                     'job.sh': (job_sh_file, jsfi), 
                     'job.pid': (job_pid_file, jifi)}

        return job_files
        

    """ Parse job files """
    def parse_job_files(self, **kwargs):
        # First, get the PID/progress/status
        job_progress_file = self.job_files['job.progress'][0]
        job_pid_file = self.job_files['job.pid'][0]
        is_running, pseudostatus, progress, pid, \
            job_pid_file, job_progress_file = self.get_job_progress(job_progress_file, job_pid_file, **kwargs)

        # Check pseudostatus
        if pseudostatus == 'completed':
            # We don't care, just overwrite the progress file, 
            # if the pid exists, delete it, and if the log exists, return it. That's it.
            # Overwrite progress file
            with open(job_progress_file, 'w') as f:
                f.write('complete,1')
            # Delete pid file if exists
            if utils.path_exists(job_pid_file):
                os.remove(job_pid_file)
            # Make sure pid is None
            job_pid_file = None
            pid = None
            progress = 1.0
            is_running = False
        
        # Check if log file exists
        job_log_file = self.job_files['job.log'][0]
        if not utils.path_exists(job_log_file):
            job_log_file = None
            self.log = None
        else:
            # Read log file
            with open(job_log_file, 'r') as f:
                job_log = f.read()
            # Set log 
            self.log = job_log
        
        # Check if sh file exists
        job_sh_file = self.job_files['job.sh'][0]
        if not utils.path_exists(job_sh_file):
            job_sh_file = None
            self.sh = None
        else:
            # Read sh file
            with open(job_sh_file, 'r') as f:
                job_sh = f.read()
            # Set sh 
            self.sh = job_sh

        # Update job files
        self.job_files = self.get_job_files(next = False)

        # Return inferred status
        return pseudostatus, is_running, progress, pid

    """ Get job progress """
    def get_job_progress(self, job_progress_file, job_pid_file, **kwargs):
        return utils.get_process_from_job(job_progress_file, job_pid_file, **kwargs)

    # """ Fix job files """
    # def fix_job_files(self, status):
        
    #     # Check status
    #     if status == 'completed':
    #         # We don't care, just overwrite the progress file, 
    #         # if the pid exists, delete it, and if the log exists, return it. That's it.
    #         # Overwrite progress file
    #         with open(job_progress_file, 'w') as f:
    #             f.write('complete,1')
    #         # Delete pid file if exists
    #         if utils.path_exists(job_pid_file):
    #             os.remove(job_pid_file)
    #         job_pid_file = None
    #         pid = None
    #         progress = 1.0
    #         is_running = False

    #     elif status == 'incomplete':
    #         # This is a bit trickier. First let's check if the pid file is there
    #         is_running, status, progress, pid, job_pid_file, job_progress_file = self.get_job_progress(job_progress_file, job_pid_file)

    #     # Check if log file exists
    #     if not utils.path_exists(job_log_file):
    #         job_log_file = None
    #     # Check if sh file exists
    #     if not utils.path_exists(job_sh_file):
    #         job_sh_file = None

    #     return is_running, status, progress, pid, job_progress_file, job_log_file, job_sh_file, job_pid_file

        